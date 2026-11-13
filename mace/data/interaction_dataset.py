"""Opt-in HDF5 samples containing a complex and all frozen isolated fragments."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from mace.data.hdf5_dataset import HDF5Dataset, unpack_value
from mace.data.utils import Configuration
from mace.tools.torch_geometric.batch import Batch


@dataclass
class InteractionSample:
    graphs: List
    target: Optional[float] = None
    weight: float = 1.0
    force_target: Optional[np.ndarray] = None
    fragment_atom_indices: Optional[np.ndarray] = None


class InteractionDataset(Dataset):
    def __init__(self, dataset, keep_total_loss=False, include_forces=False):
        if not isinstance(dataset, HDF5Dataset):
            raise ValueError(
                "Interaction-energy training requires Configuration HDF5 input"
            )
        self.dataset = dataset
        self.keep_total_loss = keep_total_loss
        self.include_forces = include_forces

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ds = self.dataset
        graph = ds[index]
        group = ds.file[
            f"config_batch_{index // ds.batch_size}/config_{index % ds.batch_size}"
        ]
        meta = group.get("metadata")
        if meta is None or "molecule_group_index" not in meta:
            if "synthetic" in str(unpack_value(group["config_type"][()])):
                raise ValueError("Synthetic sample is missing molecule_group_index")
            return InteractionSample([graph])
        if (
            any(
                key not in meta
                for key in ("molecule_charges", "molecule_multiplicities")
            )
            or "interaction_energy" not in meta.attrs
        ):
            raise ValueError(
                "Interaction sample requires fragment charges, multiplicities and interaction_energy"
            )
        props = group["properties"]
        if np.any(group["pbc"][:]):
            raise ValueError("Interaction training requires nonperiodic samples")
        for name in ("external_field", "fermi_level"):
            if name in props and np.any(props[name][()]):
                raise ValueError(
                    "Interaction training requires zero external field and fermi level"
                )
        ids = meta["molecule_group_index"][:]
        charges, spins = meta["molecule_charges"][:], meta["molecule_multiplicities"][:]
        numbers, positions = group["atomic_numbers"][:], group["positions"][:]
        n = len(charges)
        if (
            n < 2
            or ids.shape != (len(numbers),)
            or spins.shape != (n,)
            or not np.array_equal(np.unique(ids), np.arange(n))
        ):
            raise ValueError(
                "Fragment groups must be disjoint, nonempty and cover all atoms with IDs 0..K-1"
            )
        if (
            not np.isfinite(charges).all()
            or not np.isfinite(spins).all()
            or np.any(charges != np.round(charges))
            or np.any(spins < 1)
            or np.any(spins != np.round(spins))
            or not np.isclose(charges.sum(), props["total_charge"][()])
        ):
            raise ValueError(
                "Invalid fragment charges/multiplicities or inconsistent total charge"
            )
        target, weight = float(meta.attrs["interaction_energy"]), float(graph.weight)
        if not np.isfinite(target):
            raise ValueError("Nonfinite interaction-energy target")
        if not self.keep_total_loss:
            graph.weight = torch.zeros_like(graph.weight)
        graphs = [graph]
        heads = ds.kwargs.get("heads", ["Default"])
        head = (
            unpack_value(props["head"][()])
            if "head" in props
            else ds.kwargs.get("head", "Default")
        )
        isolated_forces = (
            meta["isolated_DFT_forces"][:]
            if "isolated_DFT_forces" in meta
            else np.zeros_like(positions)
        )
        for k in range(n):
            mask = ids == k
            fragment_props = dict(
                energy=(
                    float(meta["molecule_isolated_energies"][k])
                    if "molecule_isolated_energies" in meta
                    else 0.0
                ),
                forces=isolated_forces[mask],
                total_charge=float(charges[k]),
                total_spin=float(spins[k]),
                head=head,
                external_field=np.zeros(3),
                fermi_level=0.0,
            )
            # Optional labels are for diagnostics only. Auxiliary graphs have
            # no ordinary loss; moments are translated to the coordinate origin.
            if "molecule_dipoles" in meta and "molecule_centers" in meta:
                center, mu = meta["molecule_centers"][k], meta["molecule_dipoles"][k]
                fragment_props["dipole"] = mu + charges[k] * center
                if "molecule_quadrupoles" in meta:
                    shift = (
                        np.outer(mu, center)
                        + np.outer(center, mu)
                        + charges[k] * np.outer(center, center)
                    )
                    fragment_props["quadrupole"] = (
                        meta["molecule_quadrupoles"][k]
                        + shift
                        - np.eye(3) * np.trace(shift) / 3
                    )
            config = Configuration(
                numbers[mask],
                positions[mask],
                fragment_props,
                property_weights={key: 0.0 for key in group["property_weights"]},
                cell=np.zeros((3, 3)),
                pbc=(False, False, False),
                weight=0.0,
                head=head,
                config_type="interaction_auxiliary",
            )
            graphs.append(
                ds.atomic_dataclass.from_config(
                    config, z_table=ds.z_table, cutoff=ds.r_max, heads=heads
                )
            )
        force_target, inverse = None, None
        if self.include_forces:
            if "interaction_forces" in meta:
                force_target = meta["interaction_forces"][:]
            elif "isolated_DFT_forces" in meta:
                force_target = props["forces"][:] - isolated_forces
            else:
                raise ValueError(
                    "Interaction-force training requires interaction_forces or isolated_DFT_forces"
                )
            if (
                force_target.shape != positions.shape
                or not np.isfinite(force_target).all()
            ):
                raise ValueError("Invalid interaction-force labels")
            order = np.concatenate([np.flatnonzero(ids == k) for k in range(n)])
            inverse = np.empty(len(ids), dtype=np.int64)
            inverse[order] = np.arange(len(ids))
        return InteractionSample(graphs, target, weight, force_target, inverse)


def wrap_interaction_dataset(dataset, keep_total_loss=False, include_forces=False):
    if isinstance(dataset, ConcatDataset):
        return ConcatDataset(
            [
                wrap_interaction_dataset(d, keep_total_loss, include_forces)
                for d in dataset.datasets
            ]
        )
    return InteractionDataset(dataset, keep_total_loss, include_forces)


def collate_interactions(samples):
    graphs, base_graphs, base_nodes = [], [], []
    indices, coefficients, sample_ids, targets, sizes, weights = [], [], [], [], [], []
    force_full, force_isolated, force_targets, force_weights = [], [], [], []
    node_offset = 0
    for sample in samples:
        start = len(graphs)
        base_graphs.append(start)
        base_nodes.extend(range(node_offset, node_offset + sample.graphs[0].num_nodes))
        if sample.target is not None:
            count = len(sample.graphs)
            indices.extend(range(start, start + count))
            coefficients.extend([1.0] + [-1.0] * (count - 1))
            sample_ids.extend([len(targets)] * count)
            targets.append(sample.target)
            sizes.append(sample.graphs[0].num_nodes)
            weights.append(sample.weight)
        if sample.force_target is not None:
            n_atoms = sample.graphs[0].num_nodes
            if sample.fragment_atom_indices is None:
                raise ValueError(
                    "Interaction-force sample is missing the fragment atom mapping"
                )
            force_full.extend(range(node_offset, node_offset + n_atoms))
            force_isolated.extend(
                (node_offset + n_atoms + sample.fragment_atom_indices).tolist()
            )
            force_targets.extend(sample.force_target.tolist())
            force_weights.extend([sample.weight] * n_atoms)
        graphs.extend(sample.graphs)
        node_offset += sum(g.num_nodes for g in sample.graphs)
    batch = Batch.from_data_list(graphs)
    for name, values in (
        ("base_graphs", base_graphs),
        ("base_nodes", base_nodes),
        ("indices", indices),
        ("sample_ids", sample_ids),
    ):
        batch["interaction_" + name] = torch.tensor(values, dtype=torch.long)
    for name, values in (
        ("coefficients", coefficients),
        ("targets", targets),
        ("num_atoms", sizes),
        ("weights", weights),
    ):
        batch["interaction_" + name] = torch.tensor(
            values, dtype=torch.get_default_dtype()
        )
    batch.interaction_force_full = torch.tensor(force_full, dtype=torch.long)
    batch.interaction_force_isolated = torch.tensor(force_isolated, dtype=torch.long)
    batch.interaction_force_targets = torch.tensor(
        force_targets, dtype=torch.get_default_dtype()
    ).reshape(-1, 3)
    batch.interaction_force_weights = torch.tensor(
        force_weights, dtype=torch.get_default_dtype()
    )
    return batch


class InteractionDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["collate_fn"] = collate_interactions
        super().__init__(*args, **kwargs)
