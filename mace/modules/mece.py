"""Bond-centered SO(2) density expansion with SO(3)-equivariant atomic states.

The construction follows ``mace/note/mece.md``. Each directed bond rotates
neighbor states into its local frame, builds a cylindrical one-particle density,
forms higher orders by an SO(2) product of that density, and rotates the message
back to the global frame. Neighbor tuples are never enumerated: every sum over
environment atoms or bonds is a scatter.

Radial embeddings follow the note: R_{nμ} is an MLP on Bessel(ρ) and the
longitudinal encoding Z(z), optionally concatenated with element one-hots.
The geometric basis is g^{nμ} = f_env R_{nμ} e^{iμθ}. One-particle features
and the scatter to A live in a narrow width C_φ; O expands A to node width C
only after Σ_k. Each interaction owns its own MLP.
"""

import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.radial import PolynomialCutoff
from mace.modules.utils import get_outputs, prepare_graph
from mace.tools.scatter import scatter_sum


def _rotation_angles(rotation: torch.Tensor):
    """Euler angles of ``rotation`` in the e3nn Y-X-Y convention, without a determinant assert."""
    y_axis = torch.nn.functional.normalize(
        rotation @ rotation.new_tensor([0.0, 1.0, 0.0]), dim=-1
    )
    alpha, beta = o3.xyz_to_angles(y_axis)
    removed = (
        o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        @ rotation
    )
    gamma = torch.atan2(removed[..., 0, 2], removed[..., 0, 0])
    return alpha, beta, gamma


def _wigner_blocks(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, l_max: int) -> torch.Tensor:
    """Block-diagonal real Wigner matrix for l = 0..l_max. Shape [..., (l_max+1)^2, (l_max+1)^2]."""
    dim = (l_max + 1) ** 2
    blocks = alpha.new_zeros(alpha.shape + (dim, dim))
    start = 0
    for ell in range(l_max + 1):
        width = 2 * ell + 1
        blocks[..., start : start + width, start : start + width] = o3.wigner_D(
            ell, alpha, beta, gamma
        )
        start += width
    return blocks


def _wigner_from_rotation(rotation: torch.Tensor, l_max: int) -> torch.Tensor:
    """Real Wigner matrix of a rotation, as a polynomial in the matrix entries.

    Euler angles are a singular chart: ``acos`` on a frame axis has an infinite
    derivative at the poles, and force training differentiates through that chart
    twice. ``D^1 = R`` and each higher block is the Clebsch-Gordan contraction of
    ``D^{l-1}`` with ``R``, which stays smooth.
    """
    flat = rotation.reshape(-1, 3, 3)
    dim = (l_max + 1) ** 2
    blocks = flat.new_zeros(flat.shape[0], dim, dim)
    blocks[:, 0, 0] = 1
    if l_max == 0:
        return blocks.reshape(rotation.shape[:-2] + (dim, dim))
    blocks[:, 1:4, 1:4] = flat
    previous = flat
    start = 4
    for ell in range(2, l_max + 1):
        width = 2 * ell + 1
        coupling = o3.wigner_3j(ell - 1, 1, ell).to(device=flat.device, dtype=flat.dtype)
        coupling = coupling / coupling[..., 0].norm().clamp_min(1e-12)
        dim_i, dim_j, dim_p = coupling.shape
        left = torch.matmul(
            previous.transpose(-1, -2), coupling.reshape(dim_i, dim_j * dim_p)
        ).reshape(-1, previous.shape[-1], dim_j, dim_p)
        left = left.permute(0, 1, 3, 2).reshape(-1, previous.shape[-1] * dim_p, dim_j)
        mid = torch.matmul(left, flat).reshape(-1, previous.shape[-1], dim_p, flat.shape[-1])
        current = torch.matmul(
            mid.permute(0, 2, 1, 3).reshape(-1, dim_p, previous.shape[-1] * flat.shape[-1]),
            coupling.reshape(previous.shape[-1] * flat.shape[-1], dim_p),
        )
        blocks[:, start : start + width, start : start + width] = current
        previous = current
        start += width
    return blocks.reshape(rotation.shape[:-2] + (dim, dim))


def bond_frames(bond_axis: torch.Tensor) -> torch.Tensor:
    """Rotation F with F @ bond_axis = e_z.

    The tangent axes are the branchless orthonormal basis of Duff et al. (2017).
    Its denominator ``sign(n_z) + n_z`` never vanishes, so the second derivative
    stays finite at the poles. A spherical-coordinate frame does not.
    """
    axis = torch.nn.functional.normalize(bond_axis, dim=-1)
    sign = torch.where(axis[..., 2] >= 0, torch.ones_like(axis[..., 2]), -torch.ones_like(axis[..., 2]))
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


def _so2_fourier_matrix(l_max: int) -> torch.Tensor:
    """Complex change of basis from e3nn real irreps to magnetic order m = -l..l.

    Row ``m + l`` of each l block is a left eigenvector of a bond-axis rotation,
    so a transverse gauge change by gamma multiplies that coefficient by exp(i m gamma).
    """
    dim = (l_max + 1) ** 2
    matrix = torch.zeros(dim, dim, dtype=torch.complex128)
    offset = 0
    for ell in range(l_max + 1):
        width = 2 * ell + 1
        epsilon = 1e-6
        plus = o3.matrix_z(torch.tensor(epsilon, dtype=torch.float64))
        minus = o3.matrix_z(torch.tensor(-epsilon, dtype=torch.float64))
        alpha_plus, beta_plus, gamma_plus = _rotation_angles(plus)
        alpha_minus, beta_minus, gamma_minus = _rotation_angles(minus)
        generator = (
            o3.wigner_D(ell, alpha_plus, beta_plus, gamma_plus)
            - o3.wigner_D(ell, alpha_minus, beta_minus, gamma_minus)
        ) / (2 * epsilon)
        eigenvalues, eigenvectors = torch.linalg.eig(generator.transpose(0, 1).to(torch.complex128))
        taken = torch.zeros(width, dtype=torch.bool)
        rows = []
        for magnetic in range(-ell, ell + 1):
            distance = (eigenvalues - 1j * magnetic).abs()
            distance = distance.masked_fill(taken, 1e9)
            choice = torch.argmin(distance)
            taken[choice] = True
            vector = eigenvectors[:, choice]
            vector = vector / vector.abs().norm().clamp_min(1e-12)
            rows.append(vector)
        matrix[offset : offset + width, offset : offset + width] = torch.stack(rows, dim=0)
        offset += width
    return matrix


def _complex(real: torch.Tensor) -> torch.Tensor:
    return torch.complex(real, torch.zeros_like(real))


def _mix_channels(weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """weight [M, out, in], values [N, in, M] -> [N, out, M]."""
    mixed = torch.matmul(values.permute(2, 0, 1), weight.transpose(-1, -2))
    return mixed.permute(1, 2, 0)


def _zero_linear(in_features: int, out_features: int) -> torch.nn.Linear:
    layer = torch.nn.Linear(in_features, out_features, bias=False)
    torch.nn.init.zeros_(layer.weight)
    return layer


def _truncated_eye(out_features: int, in_features: int) -> torch.Tensor:
    """Rectangular identity used to init C -> C_φ and C_φ -> C maps."""
    weight = torch.zeros(out_features, in_features, dtype=torch.get_default_dtype())
    rank = min(out_features, in_features)
    weight[:rank, :rank] = torch.eye(rank, dtype=torch.get_default_dtype())
    return weight


def _diagonal_product_weight(num_channels: int) -> torch.Tensor:
    """Init full TP mix so B_γ = Σ_{αβ} W_{γαβ} B_α Â_β starts as channelwise (γ=α=β)."""
    weight = torch.zeros(
        num_channels, num_channels, num_channels, dtype=torch.get_default_dtype()
    )
    index = torch.arange(num_channels)
    weight[index, index, index] = 1.0
    return weight


class _EnergyMLP(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.SiLU(),
            _zero_linear(hidden_features, out_features),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _safe_bessel(distance: torch.Tensor, weights: torch.Tensor, prefactor: torch.Tensor) -> torch.Tensor:
    """MACE Bessel values, with the sinc limit so a zero radius stays finite."""
    return prefactor * weights * torch.sinc((distance * weights) / math.pi)


class CylindricalBasis(torch.nn.Module):
    """Fixed inputs to R_{nμ}: Bessel(ρ), Bessel(|z|), signed z/r_max, and phase.

    The learnable map R = MLP(Bessel(ρ), Z(z), ...) is MECEInteraction.env_radial_mlp.
    """

    def __init__(self, r_max: float, num_radial: int, q_max: int, cutoff_order: int):
        super().__init__()
        self.q_max = q_max
        self.cutoff = PolynomialCutoff(r_max=r_max, p=cutoff_order)
        weights = (
            math.pi
            / r_max
            * torch.linspace(1.0, num_radial, num_radial, dtype=torch.get_default_dtype())
        )
        self.register_buffer("bessel_weights", weights)
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "prefactor", torch.tensor(math.sqrt(2.0 / r_max), dtype=torch.get_default_dtype())
        )

    def forward(self, local_vectors: torch.Tensor):
        rho = torch.linalg.vector_norm(local_vectors[..., :2], dim=-1, keepdim=True)
        height = local_vectors[..., 2:3]
        radius = torch.linalg.vector_norm(local_vectors, dim=-1, keepdim=True)
        cutoff = self.cutoff(radius)
        rho_bessel = _safe_bessel(rho, self.bessel_weights, self.prefactor) * cutoff
        height_bessel = _safe_bessel(height.abs(), self.bessel_weights, self.prefactor) * cutoff
        signed_height = height / self.r_max.to(local_vectors.dtype) * cutoff
        features = torch.cat((rho_bessel, height_bessel, signed_height), dim=-1)
        transverse = torch.complex(local_vectors[..., 0], local_vectors[..., 1]) / self.r_max.to(
            local_vectors.dtype
        )
        orders = torch.arange(self.q_max + 1, device=local_vectors.device)
        phase = transverse.unsqueeze(-1) ** orders
        phase = torch.where(orders == 0, torch.ones_like(phase), phase)
        return features, phase, cutoff


class MECEInteraction(torch.nn.Module):
    """One bond-density interaction: one-particle projection, SO(2) products, pair path, gated atomic update."""

    def __init__(
        self,
        num_channels: int,
        num_density_channels: int,
        l_max: int,
        q_max: int,
        nu_max: int,
        num_bessel: int,
        num_elements: int,
        num_paths: int,
        radial_out: int = 64,
        full_self_tensor_product: bool = False,
        nonlinear_B: bool = False,
        mix_l_after_rotation: bool = False,
        so2_full_tp: bool = False,
        so2_lowrank_tp: bool = False,
        so2_tp_rank: int = 8,
    ):
        super().__init__()
        if num_density_channels < 1:
            raise ValueError("num_density_channels must be positive")
        if so2_full_tp and so2_lowrank_tp:
            raise ValueError("SO2fulltp and SO2lowranktp are mutually exclusive")
        self.num_channels = num_channels
        self.num_density_channels = num_density_channels
        self.l_max = l_max
        self.q_max = q_max
        self.nu_max = nu_max
        self.full_self_tensor_product = bool(full_self_tensor_product)
        self.nonlinear_B = bool(nonlinear_B)
        self.mix_l_after_rotation = bool(mix_l_after_rotation)
        self.so2_full_tp = bool(so2_full_tp)
        self.so2_lowrank_tp = bool(so2_lowrank_tp)
        self.so2_tp_rank = int(so2_tp_rank)
        irrep_dim = (l_max + 1) ** 2
        environment_basis = 2 * num_bessel + 1
        self.bond_radial_mlp = FullyConnectedNet(
            [num_bessel + 2 * num_elements, 64, 64, 64, radial_out],
            act=torch.nn.functional.silu,
        )
        # R_{nμ} lives in the narrow density width C_φ.
        self.env_radial_mlp = FullyConnectedNet(
            [environment_basis + 3 * num_elements, 64, 64, 64, num_density_channels * (q_max + 1)],
            act=torch.nn.functional.silu,
        )
        # U maps node channels into density width. Default: one matrix per l.
        # mix_l_after_rotation: at fixed m, mix ell >= |m| of one parity.
        if self.mix_l_after_rotation:
            num_paths = self._init_fixed_m_mix(num_channels, num_density_channels)
        else:
            self.channel_mix = torch.nn.Parameter(
                _truncated_eye(num_density_channels, num_channels)
                .expand(l_max + 1, -1, -1)
                .clone()
            )
        if self.so2_full_tp:
            # W^{(|m|,|μ|)}_{a b n}: full C_φ×C_φ→C_φ map, shared across sign flips.
            num_keys = (l_max + 1) * (q_max + 1)
            weight = torch.zeros(
                num_keys,
                num_density_channels,
                num_density_channels,
                num_density_channels,
                dtype=torch.get_default_dtype(),
            )
            index = torch.arange(num_density_channels)
            weight[:, index, index, index] = 1.0
            self.phi_tp_weight = torch.nn.Parameter(weight)
        elif self.so2_lowrank_tp:
            if self.so2_tp_rank < 1:
                raise ValueError("so2_tp_rank must be positive")
            rank = self.so2_tp_rank
            self.phi_left_mix = torch.nn.Parameter(
                _truncated_eye(rank, num_density_channels)
                .expand(l_max + 1, -1, -1)
                .clone()
            )
            self.phi_right_mix = torch.nn.Parameter(
                _truncated_eye(rank, num_density_channels)
                .expand(q_max + 1, -1, -1)
                .clone()
            )
            self.phi_out_mix = torch.nn.Parameter(
                _truncated_eye(num_density_channels, rank)
                .expand(q_max + 1, -1, -1)
                .clone()
            )
        else:
            self.path_weight = torch.nn.Parameter(
                torch.randn(num_density_channels, num_paths, dtype=torch.get_default_dtype())
                / math.sqrt(num_density_channels)
            )
        # O: [q_max+1, C, C_φ] expands A back to node width after Σ_k.
        self.output_mix = torch.nn.Parameter(
            _truncated_eye(num_channels, num_density_channels)
            .expand(q_max + 1, -1, -1)
            .clone()
        )
        if nu_max > 1:
            self.density_mix = torch.nn.Parameter(
                torch.eye(num_channels, dtype=torch.get_default_dtype())
                .expand(nu_max - 1, q_max + 1, -1, -1)
                .clone()
            )
            if self.full_self_tensor_product:
                # W^{(ν)}_{γαβ}: project the C×C channel TP back to width C.
                self.product_mix = torch.nn.Parameter(
                    _diagonal_product_weight(num_channels)
                    .expand(nu_max - 1, -1, -1, -1)
                    .clone()
                )
        self.message_weight = torch.nn.Parameter(
            torch.randn(irrep_dim, num_channels, nu_max * num_channels, dtype=torch.get_default_dtype())
            / math.sqrt(nu_max * num_channels)
        )
        self.update_mix = torch.nn.Parameter(
            torch.eye(num_channels, dtype=torch.get_default_dtype()).expand(l_max + 1, -1, -1).clone()
        )
        if self.nonlinear_B:
            # Per correlation order: mix channels within each |q|, then SO(2) gate from q=0.
            self.nonlinear_mix = torch.nn.Parameter(
                torch.eye(num_channels, dtype=torch.get_default_dtype())
                .expand(nu_max, q_max + 1, -1, -1)
                .clone()
            )
            self.b_invariant_mlp = torch.nn.ModuleList()
            self.b_gate_mlp = torch.nn.ModuleList()
            for _ in range(nu_max):
                invariant = torch.nn.Sequential(
                    torch.nn.Linear(num_channels, num_channels),
                    torch.nn.SiLU(),
                    torch.nn.Linear(num_channels, num_channels),
                )
                torch.nn.init.zeros_(invariant[-1].weight)
                torch.nn.init.zeros_(invariant[-1].bias)
                gate = torch.nn.Sequential(
                    torch.nn.Linear(num_channels, num_channels),
                    torch.nn.SiLU(),
                    torch.nn.Linear(num_channels, num_channels * q_max if q_max > 0 else num_channels),
                )
                torch.nn.init.zeros_(gate[-1].weight)
                torch.nn.init.constant_(gate[-1].bias, 2.0)
                self.b_invariant_mlp.append(invariant)
                self.b_gate_mlp.append(gate)
        bond_input = 2 * num_channels + radial_out
        self.bond_mlp = torch.nn.Sequential(
            torch.nn.Linear(bond_input, num_density_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(num_density_channels, num_density_channels),
        )
        torch.nn.init.zeros_(self.bond_mlp[-1].weight)
        torch.nn.init.constant_(self.bond_mlp[-1].bias, 2.0)
        self.pair_mlp = torch.nn.Sequential(
            torch.nn.Linear(bond_input, num_channels),
            torch.nn.SiLU(),
            _zero_linear(num_channels, (l_max + 1) * num_channels),
        )
        invariant_size = num_channels * (l_max + 2)
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(invariant_size, num_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(num_channels, num_channels * (l_max + 1)),
        )
        torch.nn.init.zeros_(self.gate_mlp[-1].weight)
        torch.nn.init.constant_(self.gate_mlp[-1].bias, 1.0)
        self.scalar_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_channels, num_channels),
            torch.nn.SiLU(),
            _zero_linear(num_channels, num_channels),
        )

    def _init_fixed_m_mix(self, num_channels: int, num_density_channels: int) -> int:
        """Real U^{(|m|, p)} mixing ell >= |m| with (-1)^ell = p. +m and -m share U."""
        self.fixed_m_mix = torch.nn.ParameterList()
        self._fixed_m_abs = []
        self._fixed_m_parity = []
        group_keys = []
        eye = _truncated_eye(num_density_channels, num_channels)
        for abs_m in range(self.l_max + 1):
            for parity in (0, 1):
                ells = [
                    ell
                    for ell in range(abs_m, self.l_max + 1)
                    if ell % 2 == parity
                ]
                if not ells:
                    continue
                weight = eye.new_zeros(num_density_channels, len(ells), num_channels)
                for slot, _ell in enumerate(ells):
                    weight[:, slot, :] = eye
                slot_index = len(self.fixed_m_mix)
                self.fixed_m_mix.append(torch.nn.Parameter(weight))
                # Component layout is ell^2 + (m + ell), m = -ell..ell.
                plus = torch.tensor(
                    [ell * ell + abs_m + ell for ell in ells], dtype=torch.int64
                )
                minus = torch.tensor(
                    [ell * ell - abs_m + ell for ell in ells], dtype=torch.int64
                )
                self.register_buffer(f"fixed_m_plus_{slot_index}", plus)
                self.register_buffer(f"fixed_m_minus_{slot_index}", minus)
                self._fixed_m_abs.append(abs_m)
                self._fixed_m_parity.append(parity)
                group_keys.append((parity, abs_m))
        path_parity = []
        path_m = []
        path_mu = []
        path_q = []
        present = set(group_keys)
        for parity, abs_m in group_keys:
            signed_orders = (0,) if abs_m == 0 else (-abs_m, abs_m)
            for signed_m in signed_orders:
                if (parity, abs(signed_m)) not in present:
                    continue
                for mu_index, mu in enumerate(range(-self.q_max, self.q_max + 1)):
                    order = signed_m + mu
                    if abs(order) <= self.q_max:
                        path_parity.append(parity)
                        path_m.append(signed_m + self.l_max)
                        path_mu.append(mu_index)
                        path_q.append(order + self.q_max)
        path_m_tensor = torch.tensor(path_m, dtype=torch.int64)
        path_mu_tensor = torch.tensor(path_mu, dtype=torch.int64)
        self.register_buffer(
            "fixed_m_path_parity", torch.tensor(path_parity, dtype=torch.int64)
        )
        self.register_buffer("fixed_m_path_m", path_m_tensor)
        self.register_buffer("fixed_m_path_mu", path_mu_tensor)
        self.register_buffer(
            "fixed_m_path_q", torch.tensor(path_q, dtype=torch.int64)
        )
        self.register_buffer(
            "fixed_m_path_abs_m", (path_m_tensor - self.l_max).abs()
        )
        self.register_buffer(
            "fixed_m_path_abs_mu", (path_mu_tensor - self.q_max).abs()
        )
        return len(path_q)

    def _apply_fixed_m_mix(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Map [P, C, (l+1)^2] complex coefficients to [P, C_φ, 2, 2 l_max+1]."""
        number_of_m = 2 * self.l_max + 1
        mixed = coefficients.new_zeros(
            coefficients.shape[0],
            self.num_density_channels,
            2,
            number_of_m,
        )
        for index, weight in enumerate(self.fixed_m_mix):
            abs_m = self._fixed_m_abs[index]
            parity = self._fixed_m_parity[index]
            plus = getattr(self, f"fixed_m_plus_{index}")
            transformed = torch.einsum(
                "oec,pec->po",
                _complex(weight),
                coefficients.index_select(-1, plus).permute(0, 2, 1),
            )
            mixed[:, :, parity, self.l_max + abs_m] = transformed
            if abs_m == 0:
                continue
            minus = getattr(self, f"fixed_m_minus_{index}")
            transformed = torch.einsum(
                "oec,pec->po",
                _complex(weight),
                coefficients.index_select(-1, minus).permute(0, 2, 1),
            )
            mixed[:, :, parity, self.l_max - abs_m] = transformed
        return mixed

    def _project_phi(
        self,
        mixed: torch.Tensor,
        cylindrical: torch.Tensor,
        path_feature: torch.Tensor,
        path_mu: torch.Tensor,
        path_q: torch.Tensor,
        path_abs_m: torch.Tensor,
        path_abs_mu: torch.Tensor,
        abs_order: torch.Tensor,
        num_orders: int,
    ) -> torch.Tensor:
        """Build narrow A_{ij,a q} from atomic features x and cylindrical basis g."""
        path_count = max(int(path_q.numel()), 1)
        scale = 1.0 / math.sqrt(path_count)
        if self.so2_full_tp:
            atomic = mixed.index_select(-1, path_feature)
            radial = cylindrical.index_select(-1, path_mu)
            keys = path_abs_m * (self.q_max + 1) + path_abs_mu
            weight = _complex(self.phi_tp_weight[keys])
            product = torch.einsum("pabn,ebp,enp->eap", weight, atomic, radial)
            return scatter_sum(product, path_q, dim=2, dim_size=num_orders) * scale
        if self.so2_lowrank_tp:
            atomic = mixed.index_select(-1, path_feature)
            radial = cylindrical.index_select(-1, path_mu)
            left = torch.einsum(
                "prc,ecp->erp", _complex(self.phi_left_mix[path_abs_m]), atomic
            )
            right = torch.einsum(
                "prc,ecp->erp", _complex(self.phi_right_mix[path_abs_mu]), radial
            )
            compressed = scatter_sum(
                left * right, path_q, dim=2, dim_size=num_orders
            ) * scale
            return torch.einsum(
                "qor,erq->eoq", _complex(self.phi_out_mix[abs_order]), compressed
            )
        product = mixed.index_select(-1, path_feature) * cylindrical.index_select(
            -1, path_mu
        )
        product = product * _complex(self.path_weight)
        return scatter_sum(product, path_q, dim=2, dim_size=num_orders) * scale

    def _mix_orders(self, values: torch.Tensor, weight: torch.Tensor, abs_order: torch.Tensor) -> torch.Tensor:
        selected = _complex(weight[abs_order])
        return _mix_channels(selected, values)

    def _nonlinear_b(
        self, values: torch.Tensor, basis: Dict[str, torch.Tensor], correlation_index: int
    ) -> torch.Tensor:
        """SO(2) gated nonlinearity on B_{αq}: mix channels, gate |q|>0 from MLP(q=0)."""
        mixed = self._mix_orders(values, self.nonlinear_mix[correlation_index], basis["abs_order"])
        zero_index = self.q_max
        invariants = mixed[:, :, zero_index].real
        invariant_update = self.b_invariant_mlp[correlation_index](invariants)
        output = mixed.clone()
        output[:, :, zero_index] = torch.complex(invariants + invariant_update, torch.zeros_like(invariants))
        if self.q_max == 0:
            return output
        gates = torch.sigmoid(self.b_gate_mlp[correlation_index](invariants))
        gates = gates.reshape(values.shape[0], self.num_channels, self.q_max)
        abs_order = basis["abs_order"]
        # |q| = 0 stays ungated; ±q share gates[..., |q|-1].
        positive = abs_order > 0
        gate_index = (abs_order[positive] - 1).to(torch.int64)
        gated = mixed[:, :, positive] * gates[:, :, gate_index]
        output[:, :, positive] = gated
        return output

    def _so2_product(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        basis: Dict[str, torch.Tensor],
        product_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SO(2) convolution over orders. Channelwise or full C×C TP then project."""
        left_pairs = left.index_select(-1, basis["conv_left"])
        right_pairs = right.index_select(-1, basis["conv_right"])
        if product_weight is None:
            paired = left_pairs * right_pairs
        else:
            # T_{αβ} = B_α Â_β, then B'_γ = Σ_{αβ} W_{γαβ} T_{αβ}.
            number_of_edges, _, number_of_pairs = left_pairs.shape
            out_channels, left_channels, right_channels = product_weight.shape
            weight = _complex(product_weight)
            # mid[e,γ,β,p] = Σ_α W[γ,α,β] left[e,α,p]
            mid = torch.matmul(
                weight.permute(0, 2, 1).reshape(out_channels * right_channels, left_channels),
                left_pairs.permute(1, 0, 2).reshape(left_channels, number_of_edges * number_of_pairs),
            ).reshape(out_channels, right_channels, number_of_edges, number_of_pairs)
            mid = mid.permute(2, 0, 1, 3)
            paired = (mid * right_pairs.unsqueeze(1)).sum(dim=2)
        return (
            scatter_sum(paired, basis["conv_out"], dim=2, dim_size=basis["num_orders"])
            / basis["num_orders"]
        )

    def _bond_radial(self, geometry: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = torch.cat((geometry["bond_bessel"], geometry["bond_species"]), dim=-1)
        return self.bond_radial_mlp(features) * geometry["bond_cutoff"]

    def _environment_radial(self, geometry: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build g^{nμ} = f_env R_{nμ} e^{iμθ} with R = MLP(Bessel(ρ), Z(z), species)."""
        features = torch.cat((geometry["env_features"], geometry["env_species"]), dim=-1)
        # R_{nμ} for μ >= 0; negative orders are filled by conjugation below.
        weights = self.env_radial_mlp(features)
        weights = weights.reshape(features.shape[0], self.num_density_channels, self.q_max + 1)
        weights = weights * geometry["env_cutoff"].unsqueeze(-1)
        positive = _complex(weights) * geometry["env_phase"].unsqueeze(1)
        if self.q_max == 0:
            return positive
        negative = positive[..., 1:].flip(-1).conj()
        return torch.cat((negative, positive), dim=-1)

    def forward(self, node_state: torch.Tensor, geometry: Dict[str, torch.Tensor], basis: Dict[str, torch.Tensor]):
        receiver = geometry["receiver"]
        sender = geometry["sender"]
        bond_radial = self._bond_radial(geometry)
        endpoint = torch.cat(
            (node_state[:, :, 0][receiver], node_state[:, :, 0][sender], bond_radial),
            dim=-1,
        )
        # Narrow density A_{ij,a q} before the post-sum expand O.
        narrow_density = node_state.new_zeros(
            receiver.shape[0],
            self.num_density_channels,
            basis["num_orders"],
            dtype=basis["fourier"].dtype,
        )
        bond_index = geometry["bond_index"]
        if bond_index.numel() > 0:
            gathered = node_state[geometry["atom_index"]]
            triplet_rotation = geometry["triplet_rotation"][bond_index]
            local_state = torch.matmul(gathered, triplet_rotation.transpose(-1, -2))
            coefficients = torch.matmul(
                torch.complex(local_state, torch.zeros_like(local_state)),
                basis["fourier"].transpose(-1, -2),
            )
            # x = U ~h, width C_φ.
            bond_gate = torch.sigmoid(self.bond_mlp(endpoint))
            cylindrical = self._environment_radial(geometry)
            if self.mix_l_after_rotation:
                mixed = self._apply_fixed_m_mix(coefficients)
                mixed = mixed * bond_gate[bond_index].view(
                    bond_index.shape[0], self.num_density_channels, 1, 1
                )
                number_of_m = 2 * self.l_max + 1
                mixed = mixed.reshape(
                    mixed.shape[0], self.num_density_channels, 2 * number_of_m
                )
                path_feature = (
                    self.fixed_m_path_parity * number_of_m + self.fixed_m_path_m
                )
                path_mu = self.fixed_m_path_mu
                path_q = self.fixed_m_path_q
                path_abs_m = self.fixed_m_path_abs_m
                path_abs_mu = self.fixed_m_path_abs_mu
            else:
                mixed = _mix_channels(
                    _complex(self.channel_mix[basis["l_of_component"]]), coefficients
                )
                mixed = mixed * bond_gate[bond_index].unsqueeze(-1)
                path_feature = basis["path_lm"]
                path_mu = basis["path_mu"]
                path_q = basis["path_q"]
                path_abs_m = basis["path_abs_m"]
                path_abs_mu = basis["path_abs_mu"]
            projected = self._project_phi(
                mixed,
                cylindrical,
                path_feature,
                path_mu,
                path_q,
                path_abs_m,
                path_abs_mu,
                basis["abs_order"],
                basis["num_orders"],
            )
            narrow_density = scatter_sum(
                projected, bond_index, dim=0, dim_size=receiver.shape[0]
            )

        narrow_density = narrow_density / geometry["normalizer"]
        # A' = O A, expand to node width C after Σ_k.
        density = self._mix_orders(narrow_density, self.output_mix, basis["abs_order"])
        if self.nonlinear_B:
            density = self._nonlinear_b(density, basis, 0)
        correlations = [density]
        order_count = 1
        while order_count < self.nu_max:
            mixed_density = self._mix_orders(
                density, self.density_mix[order_count - 1], basis["abs_order"]
            )
            product_weight = (
                self.product_mix[order_count - 1] if self.full_self_tensor_product else None
            )
            convolved = self._so2_product(
                correlations[-1], mixed_density, basis, product_weight=product_weight
            )
            if self.nonlinear_B:
                convolved = self._nonlinear_b(convolved, basis, order_count)
            correlations.append(convolved)
            order_count += 1
        stacked = torch.cat(correlations, dim=1)
        valid = basis["order_valid"].to(stacked.dtype)
        gathered_orders = stacked.index_select(-1, basis["component_order"]) * valid
        message = _mix_channels(_complex(self.message_weight), gathered_orders)
        pair = self.pair_mlp(endpoint).reshape(receiver.shape[0], self.l_max + 1, self.num_channels)
        pair_message = torch.matmul(pair.transpose(1, 2), basis["m0_scatter"])
        message = message + torch.complex(pair_message, torch.zeros_like(pair_message))
        local_real = torch.matmul(message, basis["fourier_inv"].transpose(-1, -2)).real
        global_message = torch.matmul(local_real, geometry["rotation"])
        aggregated = scatter_sum(
            global_message, receiver, dim=0, dim_size=node_state.shape[0]
        ) / geometry["normalizer"]
        mixed_update = _mix_channels(self.update_mix[basis["l_of_component"]], aggregated)
        squared = scatter_sum(
            node_state.square(), basis["l_of_component"], dim=2, dim_size=self.l_max + 1
        )
        invariants = torch.cat(
            (node_state[:, :, 0], torch.sqrt(squared + 1e-8).flatten(1)), dim=-1
        )
        gate = torch.sigmoid(self.gate_mlp(invariants)).reshape(
            node_state.shape[0], self.num_channels, self.l_max + 1
        )
        updated = node_state + gate.index_select(-1, basis["l_of_component"]) * mixed_update
        scalars = updated[:, :, 0] + self.scalar_mlp(updated[:, :, 0])
        updated = torch.cat((scalars.unsqueeze(-1), updated[:, :, 1:]), dim=-1)
        return updated, stacked.real, bond_radial


class MECE(torch.nn.Module):
    """MACE-compatible bond-centered potential. The forward consumes a MACE batch dict."""

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
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        num_longitudinal: int = 4,
        num_density_channels: Optional[int] = None,
        full_self_tensor_product: bool = False,
        nonlinear_B: bool = False,
        mix_l_after_rotation: bool = False,
        so2_full_tp: bool = False,
        so2_lowrank_tp: bool = False,
        so2_tp_rank: int = 8,
        include_node_energy: bool = True,
        include_edge_energy: bool = True,
    ):
        super().__init__()
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        self.include_node_energy = include_node_energy
        self.include_edge_energy = include_edge_energy
        self.full_self_tensor_product = bool(full_self_tensor_product)
        self.nonlinear_B = bool(nonlinear_B)
        self.mix_l_after_rotation = bool(mix_l_after_rotation)
        self.so2_full_tp = bool(so2_full_tp)
        self.so2_lowrank_tp = bool(so2_lowrank_tp)
        self.so2_tp_rank = int(so2_tp_rank)
        if self.so2_full_tp and self.so2_lowrank_tp:
            raise ValueError("SO2fulltp and SO2lowranktp are mutually exclusive")
        hidden_irreps = o3.Irreps(hidden_irreps)
        angular = [ir.l for _, ir in hidden_irreps]
        if angular != list(range(hidden_irreps.lmax + 1)):
            raise ValueError("MECE hidden irreps must contain each l from 0 to l_max exactly once")
        channels = hidden_irreps[0].mul
        if any(mul != channels for mul, _ in hidden_irreps):
            raise ValueError("every MECE irrep must have the same channel count")
        if num_density_channels is None:
            num_density_channels = channels
        if num_density_channels < 1:
            raise ValueError("num_density_channels must be positive")
        self.num_density_channels = int(num_density_channels)
        self.l_max = hidden_irreps.lmax
        self.q_max = min(max_ell, self.l_max)
        self.nu_max = int(correlation)
        self.num_bessel = num_bessel
        self.num_polynomial_cutoff = num_polynomial_cutoff
        self.hidden_irreps = hidden_irreps
        irrep_dim = (self.l_max + 1) ** 2
        num_orders = 2 * self.q_max + 1
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("num_interactions", torch.tensor(num_interactions, dtype=torch.int64))
        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
        component_l = []
        component_m = []
        for ell in range(self.l_max + 1):
            for magnetic in range(-ell, ell + 1):
                component_l.append(ell)
                component_m.append(magnetic)
        l_of_component = torch.tensor(component_l, dtype=torch.int64)
        m_of_component = torch.tensor(component_m, dtype=torch.int64)
        orders = torch.arange(-self.q_max, self.q_max + 1)
        path_lm, path_mu = torch.meshgrid(
            torch.arange(irrep_dim), torch.arange(num_orders), indexing="ij"
        )
        path_q = m_of_component[path_lm] + orders[path_mu]
        path_keep = path_q.abs() <= self.q_max
        kept_lm = path_lm[path_keep]
        kept_mu = path_mu[path_keep]
        conv_left, conv_right = torch.meshgrid(
            torch.arange(num_orders), torch.arange(num_orders), indexing="ij"
        )
        conv_sum = orders[conv_left] + orders[conv_right]
        conv_keep = conv_sum.abs() <= self.q_max
        m0_scatter = torch.zeros(self.l_max + 1, irrep_dim, dtype=torch.get_default_dtype())
        m0_index = torch.tensor(
            [sum(2 * ell + 1 for ell in range(level)) + level for level in range(self.l_max + 1)],
            dtype=torch.int64,
        )
        m0_scatter[torch.arange(self.l_max + 1), m0_index] = 1
        fourier = _so2_fourier_matrix(self.l_max)
        complex_dtype = torch.complex64 if torch.get_default_dtype() == torch.float32 else torch.complex128
        self.register_buffer("fourier", fourier.to(complex_dtype))
        self.register_buffer("fourier_inv", torch.linalg.inv(fourier).to(complex_dtype))
        self.register_buffer("l_of_component", l_of_component)
        self.register_buffer("path_lm", kept_lm)
        self.register_buffer("path_mu", kept_mu)
        self.register_buffer("path_q", (path_q[path_keep] + self.q_max).to(torch.int64))
        self.register_buffer("path_abs_m", m_of_component[kept_lm].abs())
        self.register_buffer("path_abs_mu", orders[kept_mu].abs())
        self.register_buffer("conv_left", conv_left[conv_keep])
        self.register_buffer("conv_right", conv_right[conv_keep])
        self.register_buffer("conv_out", (conv_sum[conv_keep] + self.q_max).to(torch.int64))
        self.register_buffer("abs_order", orders.abs().to(torch.int64))
        self.register_buffer(
            "component_order", (m_of_component + self.q_max).clamp(0, num_orders - 1).to(torch.int64)
        )
        self.register_buffer("order_valid", (m_of_component.abs() <= self.q_max).to(torch.get_default_dtype()))
        self.register_buffer("m0_scatter", m0_scatter)
        self.num_orders = num_orders

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=o3.Irreps(f"{num_elements}x0e"),
            irreps_out=o3.Irreps(f"{channels}x0e"),
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.products = torch.nn.ModuleList(
            [
                CylindricalBasis(
                    r_max=r_max,
                    num_radial=num_bessel,
                    q_max=self.q_max,
                    cutoff_order=num_polynomial_cutoff,
                )
            ]
        )
        radial_out = 64
        self.interactions = torch.nn.ModuleList(
            [
                MECEInteraction(
                    num_channels=channels,
                    num_density_channels=self.num_density_channels,
                    l_max=self.l_max,
                    q_max=self.q_max,
                    nu_max=self.nu_max,
                    num_bessel=num_bessel,
                    num_elements=num_elements,
                    num_paths=int(path_keep.sum()),
                    radial_out=radial_out,
                    full_self_tensor_product=self.full_self_tensor_product,
                    nonlinear_B=self.nonlinear_B,
                    mix_l_after_rotation=self.mix_l_after_rotation,
                    so2_full_tp=self.so2_full_tp,
                    so2_lowrank_tp=self.so2_lowrank_tp,
                    so2_tp_rank=self.so2_tp_rank,
                )
                for _ in range(num_interactions)
            ]
        )
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        self.scale_shift = ScaleShiftBlock(scale=atomic_inter_scale, shift=atomic_inter_shift)
        edge_size = radial_out + 2 * channels + self.nu_max * channels
        node_readouts = [
            _zero_linear(channels, len(heads))
            if layer < num_interactions
            else _EnergyMLP(channels, channels, len(heads))
            for layer in range(num_interactions + 1)
        ]
        edge_readouts = [
            _zero_linear(edge_size, len(heads))
            if layer < num_interactions - 1
            else _EnergyMLP(edge_size, channels, len(heads))
            for layer in range(num_interactions)
        ]
        self.readouts = torch.nn.ModuleList(node_readouts + edge_readouts)
        self.num_node_readouts = num_interactions + 1

    def _basis(self) -> Dict[str, torch.Tensor]:
        return {
            "fourier": self.fourier,
            "fourier_inv": self.fourier_inv,
            "l_of_component": self.l_of_component,
            "path_lm": self.path_lm,
            "path_mu": self.path_mu,
            "path_q": self.path_q,
            "path_abs_m": self.path_abs_m,
            "path_abs_mu": self.path_abs_mu,
            "conv_left": self.conv_left,
            "conv_right": self.conv_right,
            "conv_out": self.conv_out,
            "abs_order": self.abs_order,
            "component_order": self.component_order,
            "order_valid": self.order_valid,
            "m0_scatter": self.m0_scatter,
            "num_orders": self.num_orders,
        }

    def _geometry(self, positions, edge_index, shifts, batch, vectors, lengths, node_attrs):
        receiver = edge_index[1]
        sender = edge_index[0]
        number_of_atoms = positions.shape[0]
        number_of_edges = receiver.shape[0]
        safe_lengths = lengths.clamp_min(1e-6)
        bond_bessel, _ = self.radial_embedding(
            safe_lengths, node_attrs, edge_index, self.atomic_numbers
        )
        bond_cutoff = self.radial_embedding.cutoff_fn(safe_lengths)
        bond_species = torch.cat((node_attrs[sender], node_attrs[receiver]), dim=-1)
        bond_axis = -vectors / safe_lengths
        frames = bond_frames(bond_axis)
        rotation = _wigner_from_rotation(frames, self.l_max)
        midpoint = 0.5 * (positions[receiver] + positions[sender] - shifts)
        atom_ids = torch.arange(number_of_atoms, device=positions.device)
        same_graph = batch[receiver].unsqueeze(1).eq(batch.unsqueeze(0))
        not_receiver = atom_ids.unsqueeze(0).ne(receiver.unsqueeze(1))
        not_sender = atom_ids.unsqueeze(0).ne(sender.unsqueeze(1))
        displacement = positions.unsqueeze(0) - midpoint.unsqueeze(1)
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        inside = distance.lt(self.r_max) & distance.gt(0)
        bond_index, atom_index = (same_graph & not_receiver & not_sender & inside).nonzero(as_tuple=True)
        local_vectors = torch.matmul(
            frames[bond_index], displacement[bond_index, atom_index].unsqueeze(-1)
        ).squeeze(-1)
        env_features, env_phase, env_cutoff = self.products[0](local_vectors)
        env_species = torch.cat(
            (
                node_attrs[atom_index],
                node_attrs[receiver][bond_index],
                node_attrs[sender][bond_index],
            ),
            dim=-1,
        )
        return {
            "receiver": receiver,
            "sender": sender,
            "bond_bessel": bond_bessel,
            "bond_species": bond_species,
            "bond_cutoff": bond_cutoff,
            "rotation": rotation,
            "triplet_rotation": rotation,
            "bond_index": bond_index,
            "atom_index": atom_index,
            "env_features": env_features,
            "env_phase": env_phase,
            "env_cutoff": env_cutoff,
            "env_species": env_species,
            "normalizer": self.avg_num_neighbors.clamp_min(1.0).to(positions.dtype),
            "number_of_edges": number_of_edges,
        }

    def _head(self, values: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        return values[torch.arange(values.shape[0], device=values.device), heads]

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
        del compute_atomic_stresses, lammps_mliap
        context = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )
        positions = context.positions
        number_of_graphs = context.num_graphs
        node_heads = context.node_heads.to(torch.int64)
        node_range = context.num_atoms_arange.to(torch.int64)
        node_reference = self.atomic_energies_fn(data["node_attrs"])[node_range, node_heads]
        reference_energy = scatter_sum(
            node_reference, data["batch"], dim=0, dim_size=number_of_graphs
        ).to(positions.dtype)
        geometry = self._geometry(
            positions,
            data["edge_index"],
            data["shifts"],
            data["batch"],
            context.vectors,
            context.lengths,
            data["node_attrs"],
        )
        embedded = self.node_embedding(data["node_attrs"])
        padding = embedded.new_zeros(
            embedded.shape[0], embedded.shape[1], (self.l_max + 1) ** 2 - 1
        )
        node_state = torch.cat((embedded.unsqueeze(-1), padding), dim=-1)
        basis = self._basis()
        node_outputs = [self.readouts[0](node_state[:, :, 0])]
        edge_outputs = []
        for layer_index, interaction in enumerate(self.interactions):
            scalars = node_state[:, :, 0]
            node_state, correlation_real, bond_radial = interaction(node_state, geometry, basis)
            edge_features = torch.cat(
                (
                    bond_radial,
                    scalars[geometry["receiver"]],
                    scalars[geometry["sender"]],
                    correlation_real[:, :, self.q_max],
                ),
                dim=-1,
            )
            edge_outputs.append(self.readouts[self.num_node_readouts + layer_index](edge_features))
            node_outputs.append(self.readouts[layer_index + 1](node_state[:, :, 0]))

        node_sum = torch.stack(node_outputs, dim=0).sum(dim=0)
        node_interaction = self.scale_shift(self._head(node_sum, node_heads), node_heads)
        edge_sum = torch.stack(edge_outputs, dim=0).sum(dim=0)
        edge_heads = node_heads[geometry["receiver"]]
        edge_scale = torch.atleast_1d(self.scale_shift.scale).to(edge_sum.dtype)[edge_heads]
        directed_energy = 0.5 * self._head(edge_sum, edge_heads) * edge_scale
        edge_graph = scatter_sum(
            directed_energy, data["batch"][geometry["receiver"]], dim=0, dim_size=number_of_graphs
        )
        node_edge = scatter_sum(
            directed_energy, geometry["receiver"], dim=0, dim_size=positions.shape[0]
        )
        node_graph = scatter_sum(node_interaction, data["batch"], dim=0, dim_size=number_of_graphs)
        interaction_energy = node_interaction.new_zeros(number_of_graphs)
        node_energy = node_reference
        if self.include_node_energy:
            interaction_energy = interaction_energy + node_graph
            node_energy = node_energy + node_interaction
        if self.include_edge_energy:
            interaction_energy = interaction_energy + edge_graph
            node_energy = node_energy + node_edge
        total_energy = reference_energy + interaction_energy
        forces, virials, stress, hessian, edge_forces, _ = get_outputs(
            energy=interaction_energy,
            positions=positions,
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
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": interaction_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": None,
            "atomic_stresses": None,
            "displacement": context.displacement,
            "hessian": hessian,
            "node_feats": node_state.flatten(1),
        }
