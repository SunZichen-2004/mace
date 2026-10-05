"""Pure edge cluster expansion: geometry -> bond density -> SO(2) products -> energy.

No atomic hidden states, atomic tensor products, or message passing. Signed
magnetic orders have independent real radial weights. Only the real m=0
components of the cluster expansion enter the edge-energy readout.
"""

import itertools
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from e3nn import o3

from mace.modules.blocks import AtomicEnergiesBlock, ScaleShiftBlock
from mace.modules.radial import PolynomialCutoff
from mace.modules.utils import get_outputs, prepare_graph
from mace.tools.scatter import scatter_sum


def bond_frames(bond_axis: torch.Tensor) -> torch.Tensor:
    """Rotation F with F @ bond_axis = e_z.

    The tangent axes are the branchless orthonormal basis of Duff et al. (2017).
    Its denominator ``sign(n_z) + n_z`` never vanishes, so the second derivative
    stays finite at the poles. A spherical-coordinate frame does not.
    """
    axis = torch.nn.functional.normalize(bond_axis, dim=-1)
    sign = torch.where(
        axis[..., 2] >= 0, torch.ones_like(axis[..., 2]), -torch.ones_like(axis[..., 2])
    )
    scale = -1.0 / (sign + axis[..., 2])
    mixed = axis[..., 0] * axis[..., 1] * scale
    tangent_x = torch.stack(
        (
            1.0 + sign * axis[..., 0].square() * scale,
            sign * mixed,
            -sign * axis[..., 0],
        ),
        dim=-1,
    )
    tangent_y = torch.stack(
        (mixed, sign + axis[..., 1].square() * scale, -axis[..., 1]),
        dim=-1,
    )
    return torch.stack((tangent_x, tangent_y, axis), dim=-2)


def _complex(values: torch.Tensor) -> torch.Tensor:
    return torch.complex(values, torch.zeros_like(values))


def _mlp(in_features: int, hidden: int, out_features: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, out_features),
    )


class CylindricalBasis(torch.nn.Module):
    """Bessel(rho), configurable longitudinal basis, and angular embedding.

    Bessel(z) includes the extra continuous signed coordinate z/r_max.
    Legendre includes both even and odd degrees, without that extra coordinate.
    smooth_theta_embedding=False gives the pure phase exp(i m theta).
    When enabled, Cartesian polynomials implement (rho/r_max)**|m| exp(i m theta)
    without evaluating atan2, including on the bond axis.
    """

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        m_max: int,
        z_basis: str = "bessel",
        smooth_theta_embedding: bool = False,
        bond_neighbor_union: bool = False,
    ):
        super().__init__()
        z_basis = z_basis.lower()
        if z_basis not in ("bessel", "legendre"):
            raise ValueError("z_basis must be bessel or legendre")
        if z_basis == "legendre" and num_radial < 2:
            raise ValueError(
                "Legendre z encoding needs at least two basis functions to include an odd degree"
            )
        self.z_basis = z_basis
        self.num_radial = num_radial
        self.m_max = m_max
        self.smooth_theta_embedding = bool(smooth_theta_embedding)
        self.num_features = 2 * num_radial + (1 if z_basis == "bessel" else 0)
        self.register_buffer("r_max", torch.tensor(float(r_max)))
        # Union neighborhoods can reach |z| < r_max + r_ij/2 <= 1.5*r_max.
        self.register_buffer(
            "z_scale",
            torch.tensor(float(r_max) * (1.5 if bond_neighbor_union else 1.0)),
        )
        self.register_buffer(
            "bessel_weights", math.pi / r_max * torch.arange(1, num_radial + 1)
        )
        self.register_buffer("prefactor", torch.tensor(math.sqrt(2.0 / r_max)))
        self.register_buffer("orders", torch.arange(-m_max, m_max + 1))

    def _bessel_squared(self, distance_squared: torch.Tensor) -> torch.Tensor:
        """Even analytic Bessel function, with finite second derivatives at zero."""
        argument_squared = distance_squared * self.bessel_weights.square()
        small = argument_squared < 1e-4
        argument = torch.sqrt(
            torch.where(small, torch.ones_like(argument_squared), argument_squared)
        )
        series = (
            1
            - argument_squared / 6
            + argument_squared.square() / 120
            - argument_squared**3 / 5040
            + argument_squared**4 / 362880
        )
        sinc = torch.where(small, series, torch.sinc(argument / math.pi))
        return self.prefactor * self.bessel_weights * sinc

    def bessel(self, distance: torch.Tensor) -> torch.Tensor:
        return self._bessel_squared(distance.square())

    def legendre(self, height: torch.Tensor) -> torch.Tensor:
        coordinate = height / self.z_scale
        polynomials = [torch.ones_like(coordinate), coordinate]
        for degree in range(1, self.num_radial - 1):
            polynomials.append(
                (
                    (2 * degree + 1) * coordinate * polynomials[-1]
                    - degree * polynomials[-2]
                )
                / (degree + 1)
            )
        return torch.cat(polynomials[: self.num_radial], dim=-1)

    def forward(self, local_vectors: torch.Tensor):
        rho_squared = local_vectors[:, :2].square().sum(dim=-1, keepdim=True)
        height = local_vectors[:, 2:3]
        radial = self._bessel_squared(rho_squared)
        if self.z_basis == "bessel":
            features = torch.cat(
                (radial, self.bessel(height), height / self.r_max), dim=-1
            )
        else:
            features = torch.cat((radial, self.legendre(height)), dim=-1)
        if self.smooth_theta_embedding:
            transverse = (
                torch.complex(local_vectors[:, 0], local_vectors[:, 1]) / self.r_max
            )
            # Repeated multiplication keeps first and second derivatives finite at zero.
            powers = [torch.ones_like(transverse)]
            for _ in range(self.m_max):
                powers.append(powers[-1] * transverse)
            positive = torch.stack(powers, dim=-1)
            embedding = torch.cat((positive[:, 1:].flip(-1).conj(), positive), dim=-1)
        else:
            on_axis = rho_squared.squeeze(-1) == 0
            x = torch.where(
                on_axis, torch.ones_like(local_vectors[:, 0]), local_vectors[:, 0]
            )
            y = torch.where(
                on_axis, torch.zeros_like(local_vectors[:, 1]), local_vectors[:, 1]
            )
            theta = torch.atan2(y, x)
            angle = theta.unsqueeze(-1) * self.orders
            embedding = torch.complex(torch.cos(angle), torch.sin(angle))
        return features, embedding


class EdgeClusterExpansion(torch.nn.Module):
    """A_ij=sum_k g_ij,k, followed by higher-order SO(2) products."""

    def __init__(
        self,
        num_channels,
        num_density_channels,
        m_max,
        nu_max,
        num_bessel,
        element_embedding_dim,
        environment_basis_dim,
    ):
        super().__init__()
        self.num_density_channels = num_density_channels
        self.num_orders = 2 * m_max + 1
        self.nu_max = nu_max
        self.env_radial_mlp = _mlp(
            environment_basis_dim + 3 * element_embedding_dim,
            64,
            num_density_channels * self.num_orders,
        )
        identity = torch.eye(num_channels, num_density_channels)
        # Independent real channel maps at each signed magnetic order.
        self.output_mix = torch.nn.Parameter(
            identity.expand(self.num_orders, -1, -1).clone()
        )
        self.density_mix = torch.nn.Parameter(
            torch.eye(num_channels).expand(nu_max - 1, self.num_orders, -1, -1).clone()
        )
        orders = torch.arange(-m_max, m_max + 1)
        left, right = torch.meshgrid(
            torch.arange(self.num_orders), torch.arange(self.num_orders), indexing="ij"
        )
        total = orders[left] + orders[right]
        keep = total.abs() <= m_max
        self.register_buffer("conv_left", left[keep])
        self.register_buffer("conv_right", right[keep])
        self.register_buffer("conv_out", total[keep] + m_max)

    def _environment_radial(self, geometry: Dict[str, torch.Tensor]) -> torch.Tensor:
        """g^{nm}=f_cut R_nm times the selected angular embedding; R is independent for each signed m."""
        features = torch.cat(
            (geometry["env_features"], geometry["env_species"]), dim=-1
        )
        weights = self.env_radial_mlp(features).reshape(
            features.shape[0], self.num_density_channels, self.num_orders
        )
        weights = weights * geometry["env_cutoff"].unsqueeze(-1)
        return _complex(weights) * geometry["env_phase"].unsqueeze(1)

    @staticmethod
    def _mix_orders(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("qca,eaq->ecq", _complex(weight), values)

    def _so2_product(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        paired = left.index_select(-1, self.conv_left) * right.index_select(
            -1, self.conv_right
        )
        return scatter_sum(paired, self.conv_out, dim=2, dim_size=self.num_orders)

    def forward(self, geometry: Dict[str, torch.Tensor]):
        cylindrical = self._environment_radial(geometry)
        bond_feature_narrow = scatter_sum(
            cylindrical,
            geometry["bond_index"],
            dim=0,
            dim_size=geometry["number_of_edges"],
        )
        bond_feature = self._mix_orders(bond_feature_narrow, self.output_mix)
        correlations = [bond_feature]
        for order in range(1, self.nu_max):
            mixed = self._mix_orders(bond_feature, self.density_mix[order - 1])
            correlations.append(self._so2_product(correlations[-1], mixed))
        return bond_feature_narrow, torch.cat(correlations, dim=1)


class MECE(torch.nn.Module):
    """MACE-compatible pure Edge Cluster Expansion energy model.

    The historical MECE name remains available to the training CLI. hidden_irreps
    supplies only the scalar channel count; num_interactions is accepted for the
    MACE interface but there is exactly one cluster expansion, without updates.
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        heads: Optional[List[str]] = None,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        num_density_channels: Optional[int] = None,
        element_embedding_dim: Optional[int] = None,
        bond_neighbor_union: bool = False,
        bond_neighbor_middle: Optional[bool] = None,
        z_basis: str = "bessel",
        smooth_theta_embedding: bool = False,
    ):
        super().__init__()
        if bond_neighbor_middle is None:
            bond_neighbor_middle = not bond_neighbor_union
        if bool(bond_neighbor_union) == bool(bond_neighbor_middle):
            raise ValueError(
                "Choose exactly one of bond_neighbor_union and bond_neighbor_middle"
            )
        if r_max <= 0 or num_bessel < 1 or max_ell < 0 or correlation < 1:
            raise ValueError(
                "Require r_max>0, num_bessel>=1, max_ell>=0, correlation>=1"
            )
        self.heads = list(heads) if heads is not None else ["Default"]
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        channels = self.hidden_irreps[0].mul
        self.num_density_channels = int(
            channels if num_density_channels is None else num_density_channels
        )
        self.element_embedding_dim = int(
            channels if element_embedding_dim is None else element_embedding_dim
        )
        if self.num_density_channels < 1 or self.element_embedding_dim < 1:
            raise ValueError("Channel and embedding dimensions must be positive")
        self.num_bessel = int(num_bessel)
        self.num_polynomial_cutoff = int(num_polynomial_cutoff)
        self.l_max = self.q_max = int(max_ell)
        self.nu_max = int(correlation)
        self.bond_neighbor_union = bool(bond_neighbor_union)
        self.bond_neighbor_middle = bool(bond_neighbor_middle)
        self.z_basis = z_basis.lower()
        self.smooth_theta_embedding = bool(smooth_theta_embedding)
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(float(r_max)))
        self.register_buffer("num_interactions", torch.tensor(1, dtype=torch.int64))
        self.register_buffer(
            "avg_num_neighbors", torch.tensor(float(avg_num_neighbors))
        )
        self.element_embedding = torch.nn.Linear(
            num_elements, self.element_embedding_dim, bias=False
        )
        self.cylindrical_basis = CylindricalBasis(
            r_max,
            num_bessel,
            self.q_max,
            z_basis=self.z_basis,
            smooth_theta_embedding=self.smooth_theta_embedding,
            bond_neighbor_union=self.bond_neighbor_union,
        )
        self.cutoff = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.edge_cluster_expansion = EdgeClusterExpansion(
            channels,
            self.num_density_channels,
            self.q_max,
            self.nu_max,
            num_bessel,
            self.element_embedding_dim,
            self.cylindrical_basis.num_features,
        )
        self.environment_readout = _mlp(
            self.nu_max * channels, channels, len(self.heads)
        )
        self.bond_readout = _mlp(
            num_bessel + 2 * self.element_embedding_dim, channels, len(self.heads)
        )
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def _candidate_images(self, positions, atom_ids, midpoints, cell, pbc):
        """Enumerate periodic atom images covering all bond search regions.

        Image shifts are integer topology; Cartesian translations retain cell
        derivatives for stress. Distinct images of the same atom remain distinct.
        """
        if not bool(pbc.any()):
            return atom_ids, positions[atom_ids]
        inverse = torch.linalg.pinv(cell.detach())
        atom_fractional = positions[atom_ids].detach() @ inverse
        center_fractional = midpoints.detach() @ inverse
        # A union of endpoint balls lies within 1.5*r_max of the midpoint.
        radius = self.r_max.detach() * (1.5 if self.bond_neighbor_union else 1.0)
        padding = radius * torch.linalg.vector_norm(inverse, dim=0)
        low = torch.floor(
            center_fractional.amin(0) - atom_fractional.amax(0) - padding
        ).to(torch.int64)
        high = torch.ceil(
            center_fractional.amax(0) - atom_fractional.amin(0) + padding
        ).to(torch.int64)
        ranges = [
            range(int(low[d]), int(high[d]) + 1) if bool(pbc[d]) else (0,)
            for d in range(3)
        ]
        unit_shifts = positions.new_tensor(list(itertools.product(*ranges)))
        images = positions[atom_ids].unsqueeze(0) + (unit_shifts @ cell).unsqueeze(1)
        return atom_ids.repeat(unit_shifts.shape[0]), images.reshape(-1, 3)

    def _geometry(
        self,
        positions,
        edge_index,
        vectors,
        lengths,
        batch,
        node_attrs,
        cell=None,
        pbc=None,
    ):
        sender, receiver = edge_index[0], edge_index[1]
        number_of_edges = receiver.numel()
        # MACE vectors are r_receiver-r_sender+shift. Anchor the image bond at i.
        endpoint_i = positions[receiver]
        endpoint_j = endpoint_i - vectors
        midpoint = 0.5 * (endpoint_i + endpoint_j)
        frames = bond_frames(-vectors)
        element_features = self.element_embedding(node_attrs)
        bond_features = torch.cat(
            (
                self.cylindrical_basis.bessel(lengths),
                element_features[receiver],
                element_features[sender],
            ),
            dim=-1,
        )
        parts_bond, parts_atom, parts_local, parts_cutoff = [], [], [], []
        number_of_graphs = int(batch.max()) + 1 if batch.numel() else 0
        cells = cell.reshape(-1, 3, 3) if cell is not None else None
        periodic = pbc.reshape(-1, 3) if pbc is not None else None
        for graph in range(number_of_graphs):
            edges = torch.nonzero(batch[receiver] == graph, as_tuple=True)[0]
            if edges.numel() == 0:
                continue
            atoms = torch.nonzero(batch == graph, as_tuple=True)[0]
            graph_pbc = (
                periodic[graph]
                if periodic is not None
                else torch.zeros(3, dtype=torch.bool, device=positions.device)
            )
            graph_cell = (
                cells[graph] if cells is not None else positions.new_zeros(3, 3)
            )
            candidate_ids, candidates = self._candidate_images(
                positions,
                atoms,
                midpoint[edges],
                graph_cell,
                graph_pbc,
            )
            relative = candidates.unsqueeze(0) - midpoint[edges].unsqueeze(1)
            distance_i = torch.linalg.vector_norm(
                candidates.unsqueeze(0) - endpoint_i[edges].unsqueeze(1), dim=-1
            )
            distance_j = torch.linalg.vector_norm(
                candidates.unsqueeze(0) - endpoint_j[edges].unsqueeze(1), dim=-1
            )
            not_endpoint = (distance_i > 1e-10) & (distance_j > 1e-10)
            if self.bond_neighbor_union:
                inside = (distance_i < self.r_max) | (distance_j < self.r_max)
                cutoff_distance = torch.minimum(distance_i, distance_j)
            else:
                cutoff_distance = torch.linalg.vector_norm(relative, dim=-1)
                inside = cutoff_distance < self.r_max
            row, candidate = torch.nonzero(inside & not_endpoint, as_tuple=True)
            bond_index = edges[row]
            local = torch.matmul(
                frames[bond_index], relative[row, candidate].unsqueeze(-1)
            ).squeeze(-1)
            parts_bond.append(bond_index)
            parts_atom.append(candidate_ids[candidate])
            parts_local.append(local)
            parts_cutoff.append(cutoff_distance[row, candidate].unsqueeze(-1))
        empty_index = receiver.new_empty(0)
        bond_index = torch.cat(parts_bond) if parts_bond else empty_index
        bond_neighbor_atom_index = torch.cat(parts_atom) if parts_atom else empty_index
        local_vectors = (
            torch.cat(parts_local) if parts_local else positions.new_empty(0, 3)
        )
        cutoff_distance = (
            torch.cat(parts_cutoff) if parts_cutoff else positions.new_empty(0, 1)
        )
        env_features, env_phase = self.cylindrical_basis(local_vectors)
        env_species = torch.cat(
            (
                element_features[receiver[bond_index]],
                element_features[sender[bond_index]],
                element_features[bond_neighbor_atom_index],
            ),
            dim=-1,
        )
        return {
            "receiver": receiver,
            "sender": sender,
            "bond_features": bond_features,
            "bond_cutoff": self.cutoff(lengths),
            "bond_index": bond_index,
            "bond_neighbor_atom_index": bond_neighbor_atom_index,
            "env_features": env_features,
            "env_species": env_species,
            "env_phase": env_phase,
            "env_cutoff": self.cutoff(cutoff_distance),
            "number_of_edges": number_of_edges,
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        del compute_atomic_stresses
        if lammps_mliap:
            raise NotImplementedError(
                "Pure ECE requires positions for bond environments"
            )
        context = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )
        # prepare_graph writes strained positions and shifts back into data.
        positions = data["positions"]
        cell = context.cell.reshape(-1, 3, 3)
        if compute_virials or compute_stress or compute_displacement:
            strain = 0.5 * (
                context.displacement + context.displacement.transpose(-1, -2)
            )
            cell = cell + cell @ strain
        geometry = self._geometry(
            positions,
            data["edge_index"],
            context.vectors,
            context.lengths,
            data["batch"],
            data["node_attrs"],
            cell,
            context.pbc,
        )
        bond_feature_narrow, bond_feature = self.edge_cluster_expansion(geometry)
        # Real m=0 projects the SO(2)-invariant products to reflection-even scalars.
        zero_modes = bond_feature[:, :, self.q_max].real
        edge_outputs = self.environment_readout(zero_modes) * self.bond_readout(
            geometry["bond_features"]
        )
        edge_outputs = edge_outputs * geometry["bond_cutoff"]
        receiver = geometry["receiver"]
        edge_heads = context.node_heads[receiver].to(torch.int64)
        edge_range = torch.arange(receiver.numel(), device=positions.device)
        edge_scale = torch.atleast_1d(self.scale_shift.scale).to(positions.dtype)[
            edge_heads
        ]
        # Both orientations are in the MACE graph: each contributes half.
        directed_energy = 0.5 * edge_outputs[edge_range, edge_heads] * edge_scale
        edge_graph = scatter_sum(
            directed_energy, data["batch"][receiver], dim=0, dim_size=context.num_graphs
        )
        atom_edge = scatter_sum(
            directed_energy, receiver, dim=0, dim_size=positions.shape[0]
        )
        atom_shift = torch.atleast_1d(self.scale_shift.shift).to(positions.dtype)[
            context.node_heads
        ]
        interaction_energy = edge_graph + scatter_sum(
            atom_shift, data["batch"], dim=0, dim_size=context.num_graphs
        )
        # Keep a differentiable zero even for graphs without bonds.
        interaction_energy = interaction_energy + positions.sum() * 0.0
        node_reference = self.atomic_energies_fn(data["node_attrs"])[
            context.num_atoms_arange, context.node_heads
        ]
        reference_energy = scatter_sum(
            node_reference, data["batch"], dim=0, dim_size=context.num_graphs
        ).to(positions.dtype)
        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=interaction_energy,
            positions=context.positions,
            displacement=context.displacement,
            vectors=context.vectors,
            cell=context.cell,
            pbc=context.pbc,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )
        return {
            "energy": reference_energy + interaction_energy,
            "node_energy": node_reference + atom_edge + atom_shift,
            "interaction_energy": interaction_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "atomic_virials": None,
            "atomic_stresses": None,
            "displacement": context.displacement,
            "bond_feature_narrow": bond_feature_narrow,
            "bond_feature": bond_feature,
            "edge_energy": directed_energy,
        }
