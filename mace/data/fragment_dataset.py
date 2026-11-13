"""Read disjoint fragment charge/spin labels without creating auxiliary graphs."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from mace.data.hdf5_dataset import HDF5Dataset, unpack_value
from mace.tools.torch_geometric.batch import Batch


@dataclass
class FragmentSample:
    graph: object
    ids: np.ndarray
    charges: np.ndarray
    spins: np.ndarray


class FragmentDataset(Dataset):
    def __init__(self, dataset):
        if not isinstance(dataset, HDF5Dataset):
            raise ValueError(
                "Fragment charge/spin training requires Configuration HDF5 input"
            )
        self.dataset = dataset

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
                raise ValueError("Synthetic sample is missing fragment groups")
            return FragmentSample(
                graph, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
            )
        if "molecule_charges" not in meta:
            raise ValueError("Fragment supervision requires molecule_charges")
        ids = meta["molecule_group_index"][:]
        charges = meta["molecule_charges"][:]
        if "molecule_spins" in meta:
            spins = meta["molecule_spins"][:]  # Signed 2Sz, not multiplicity.
        elif "molecule_multiplicities" in meta:
            multiplicities = meta["molecule_multiplicities"][:]
            if np.any(multiplicities < 1):
                raise ValueError("Fragment multiplicities must be positive")
            spins = multiplicities - 1
        else:
            raise ValueError(
                "Fragment supervision requires molecule_spins or molecule_multiplicities"
            )
        n = len(charges)
        if (
            n == 0
            or charges.shape != (n,)
            or spins.shape != (n,)
            or ids.shape != (graph.num_nodes,)
            or not np.array_equal(np.unique(ids), np.arange(n))
        ):
            raise ValueError(
                "Fragment groups must be nonempty, disjoint, and cover all atoms with IDs 0..K-1"
            )
        if (
            not np.isfinite(charges).all()
            or not np.isfinite(spins).all()
            or np.any(charges != np.round(charges))
            or np.any(spins != np.round(spins))
        ):
            raise ValueError(
                "Fragment charge and signed 2Sz targets must be finite integers"
            )
        if not np.isclose(charges.sum(), float(graph.total_charge), rtol=0, atol=1e-8):
            raise ValueError("Fragment charges do not sum to system charge")
        if not np.isclose(spins.sum(), float(graph.total_spin) - 1, rtol=0, atol=1e-8):
            raise ValueError(
                "Fragment spins do not sum to system 2Sz; supply signed molecule_spins for oppositely aligned fragments"
            )
        return FragmentSample(graph, ids.astype(np.int64), charges, spins)


def wrap_fragment_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        return ConcatDataset([wrap_fragment_dataset(d) for d in dataset.datasets])
    return FragmentDataset(dataset)


def collate_fragments(samples):
    batch = Batch.from_data_list([sample.graph for sample in samples])
    nodes, ids, charges, spins, weights = [], [], [], [], []
    atom_offset = fragment_offset = 0
    for sample in samples:
        n = len(sample.charges)
        if n:
            nodes.extend(range(atom_offset, atom_offset + sample.graph.num_nodes))
            ids.extend((sample.ids + fragment_offset).tolist())
            charges.extend(sample.charges.tolist())
            spins.extend(sample.spins.tolist())
            weights.extend([float(sample.graph.weight)] * n)
        atom_offset += sample.graph.num_nodes
        fragment_offset += n
    for name, values in (("node_indices", nodes), ("ids", ids)):
        batch["fragment_" + name] = torch.tensor(values, dtype=torch.long)
    for name, values in (
        ("charge_targets", charges),
        ("spin_targets", spins),
        ("weights", weights),
    ):
        batch["fragment_" + name] = torch.tensor(
            values, dtype=torch.get_default_dtype()
        )
    return batch


class FragmentDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["collate_fn"] = collate_fragments
        super().__init__(*args, **kwargs)
