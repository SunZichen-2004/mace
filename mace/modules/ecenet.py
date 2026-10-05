"""MACE-compatible wrapper around the standalone ECENet package.

Uses the MACE ``AtomicData`` batch (DataLoader edges / positions / batch index)
and the same ``prepare_graph`` + ``get_outputs`` force path as MECE, so training
recipes are directly comparable. The core energy model is imported from
``/u/sunzc/work/ecenet`` and evaluated via ``forward_batch_multi``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from e3nn import o3

# Prefer an already-installed ``ecenet``; otherwise use the working copy.
try:
    from ecenet.model import ECENet as ECENetCore
except ImportError:  # pragma: no cover
    _ECENET_ROOT = Path("/u/sunzc/work/ecenet")
    if _ECENET_ROOT.is_dir() and str(_ECENET_ROOT) not in sys.path:
        sys.path.insert(0, str(_ECENET_ROOT))
    from ecenet.model import ECENet as ECENetCore

from mace.modules.blocks import AtomicEnergiesBlock, ScaleShiftBlock
from mace.modules.utils import get_outputs, prepare_graph
from mace.tools.scatter import scatter_sum


class ECENet(torch.nn.Module):
    """Bond-centered ECENet energy model with a MACE training interface."""

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
        embed_dim: Optional[int] = None,
        n_layers: int = 2,
        n_mp: int = 1,
        n_max: int = 4,
        r_cut_neighbor: Optional[float] = None,
        self_tp: bool = False,
        self_tp_full: bool = False,
        self_tp_nu_max: Optional[int] = None,
        so2_linear: bool = False,
        element_film: bool = True,
        cutoff_type: str = "cosine",
        activation: str = "silu",
    ):
        super().__init__()
        if heads is None:
            heads = ["Default"]
        self.heads = list(heads)
        # Accepted for MACE factory / checkpoint parity; ECENet has its own radial ACE.
        self.num_bessel = int(num_bessel)
        self.num_polynomial_cutoff = int(num_polynomial_cutoff)
        hidden_irreps = o3.Irreps(hidden_irreps)
        if embed_dim is None:
            embed_dim = hidden_irreps[0].mul
        l_max = int(max_ell)
        if hidden_irreps.lmax != l_max:
            # Prefer explicit max_ell (ACE SH order); hidden_irreps only supplies C.
            pass
        if self_tp_nu_max is None:
            self_tp_nu_max = int(correlation)
        if r_cut_neighbor is None:
            r_cut_neighbor = min(float(r_max), 4.0)
        if r_cut_neighbor > float(r_max) + 1e-12:
            raise ValueError(
                f"r_cut_neighbor ({r_cut_neighbor}) must be <= r_max ({r_max}); "
                "build the MACE neighbour list with r_max covering both cutoffs"
            )

        self.l_max = l_max
        self.embed_dim = int(embed_dim)
        self.n_layers = int(n_layers)
        self.n_mp = int(n_mp)
        self.n_max = int(n_max)
        self.self_tp = bool(self_tp) or bool(so2_linear)
        self.self_tp_full = bool(self_tp_full)
        self.self_tp_nu_max = int(self_tp_nu_max)
        self.so2_linear = bool(so2_linear)
        self.hidden_irreps = hidden_irreps
        self.num_elements = int(num_elements)

        self.register_buffer(
            "atomic_numbers", torch.tensor(list(atomic_numbers), dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(float(r_max), dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "r_cut_edge",
            torch.tensor(float(r_max), dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "r_cut_neighbor",
            torch.tensor(float(r_cut_neighbor), dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "num_interactions",
            torch.tensor(int(num_interactions), dtype=torch.int64),
        )
        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(float(avg_num_neighbors), dtype=torch.get_default_dtype()),
        )

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

        self.core = ECENetCore(
            n_types=num_elements,
            r_cut_edge=float(r_max),
            r_cut_neighbor=float(r_cut_neighbor),
            l_max=l_max,
            n_max=self.n_max,
            embed_dim=self.embed_dim,
            n_layers=self.n_layers,
            n_mp=self.n_mp,
            cutoff_type=cutoff_type,
            activation=activation,
            element_film=element_film,
            self_tp=self.self_tp,
            self_tp_full=self.self_tp_full,
            self_tp_nu_max=self.self_tp_nu_max,
            so2_linear=self.so2_linear,
        )
        # E0s come from AtomicEnergiesBlock (MACE --E0s); freeze the core offset.
        with torch.no_grad():
            self.core.atomic_energy.zero_()
        self.core.atomic_energy.requires_grad_(False)

    def _types_from_node_attrs(self, node_attrs: torch.Tensor) -> torch.Tensor:
        return node_attrs.to(dtype=torch.float64).argmax(dim=-1).to(dtype=torch.long)

    def _topology_from_mace_edges(
        self,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        lengths: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
    ):
        """Split the MACE batch edge list into per-graph ECENet topologies.

        ``r_max`` edges are filtered down to ``r_cut_edge`` / ``r_cut_neighbor``.
        Indices are returned in the local (per-structure) atom numbering expected
        by ``ECENetCore.forward_batch_multi``.
        """
        src, dst = edge_index[0], edge_index[1]
        lengths = lengths.reshape(-1)
        r_edge = float(self.r_cut_edge.item())
        r_nb = float(self.r_cut_neighbor.item())
        # MACE already drops self-edges; keep a tiny floor for safety.
        edge_ok = (lengths < r_edge) & (lengths > 1e-10)
        nb_ok = (lengths < r_nb) & (lengths > 1e-10)

        n_graphs = int(ptr.numel() - 1)
        topology = []
        for graph in range(n_graphs):
            in_graph = batch[src] == graph
            offset = int(ptr[graph].item())
            e_mask = in_graph & edge_ok
            n_mask = in_graph & nb_ok
            ei = src[e_mask] - offset
            ej = dst[e_mask] - offset
            she = shifts[e_mask]
            nb_src = src[n_mask] - offset
            nb_dst = dst[n_mask] - offset
            shn = shifts[n_mask]
            topology.append((ei, ej, she, nb_src, nb_dst, shn))
        return topology

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

        node_reference = self.atomic_energies_fn(data["node_attrs"])[
            node_range, node_heads
        ]
        reference_energy = scatter_sum(
            node_reference, data["batch"], dim=0, dim_size=number_of_graphs
        ).to(positions.dtype)

        types = self._types_from_node_attrs(data["node_attrs"])
        ptr = data["ptr"]
        positions_list = [
            positions[int(ptr[g]) : int(ptr[g + 1])] for g in range(number_of_graphs)
        ]
        types_list = [
            types[int(ptr[g]) : int(ptr[g + 1])] for g in range(number_of_graphs)
        ]
        topology = self._topology_from_mace_edges(
            edge_index=data["edge_index"],
            shifts=data["shifts"],
            lengths=context.lengths,
            batch=data["batch"],
            ptr=ptr,
        )

        # Interaction energies only (core E0s are frozen at zero).
        interaction_raw = self.core.forward_batch_multi(
            positions_list, types_list, topology=topology
        ).to(dtype=positions.dtype)

        graph_heads = (
            data["head"].to(torch.int64)
            if "head" in data
            else torch.zeros(number_of_graphs, dtype=torch.int64, device=positions.device)
        )
        interaction_energy = self.scale_shift(interaction_raw, graph_heads)
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
        node_energy = node_reference  # E0 only; ECENet is edge-centred
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
            "node_feats": None,
        }
