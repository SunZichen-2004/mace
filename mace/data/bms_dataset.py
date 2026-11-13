"""Read the original BMS25 shards without putting MM atoms in the MACE graph."""

import fcntl
import gzip
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import shutil
import warnings

import h5py
import numpy as np
import torch

from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.batch import Batch


def unpack_bms_shard(path):
    path = Path(path)
    if path.suffix == ".h5":
        return path
    target = path.parent / "uncompressed" / path.stem
    target.parent.mkdir(exist_ok=True)
    with open(str(target) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            required = max(20 * 1024**3, 8 * path.stat().st_size)
            if shutil.disk_usage(target.parent).free < required:
                raise OSError(f"Insufficient scratch space to unpack {path.name}")
            partial = target.with_suffix(".h5.part")
            with gzip.open(path, "rb") as source, partial.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
            partial.replace(target)
    return target


class BMSDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        directory,
        atomic_numbers,
        cutoff,
        split="train",
        seed=1,
        split_manifest=None,
    ):
        self.files = sorted(
            (
                p
                for p in Path(directory).glob("*_*.h5*")
                if p.name.endswith((".h5", ".h5.gz"))
                if "TOPOLOGIES" not in p.name
            ),
            key=lambda p: (not p.name.startswith("MONOPEPTIDES_0."), p.stat().st_size),
        )
        if not self.files:
            raise ValueError(f"No BMS25 structure shards in {directory}")
        self.z_table = AtomicNumberTable(list(atomic_numbers))
        self.cutoff, self.split, self.seed = cutoff, split, seed
        self.epoch = 0
        self.split_manifest = None if split_manifest is None else Path(split_manifest)
        if self.split_manifest is not None:
            manifest = json.loads(self.split_manifest.read_text())
            if manifest["policy"] != "amp-bms-jacs-2026-random-1pct-v1":
                raise ValueError("Unknown BMS25 split policy")
            if split not in ("train", "valid", "test"):
                raise ValueError(f"Unknown BMS25 split: {split}")
            self.files = [Path(directory) / item["file"] for item in manifest["files"]]
            for path, item in zip(self.files, manifest["files"]):
                if path.stat().st_size != item["bytes"]:
                    raise ValueError(f"BMS25 shard differs from split manifest: {path}")
            self.split_count = manifest["counts"][split]

    def _indexed_records(self):
        root = self.split_manifest.parent
        records = np.load(root / "records.npy", mmap_mode="r")
        indices = np.load(root / f"{self.split}.npy", mmap_mode="r")
        if self.split == "train":
            indices = np.random.default_rng(self.seed + self.epoch).permutation(indices)
        self.epoch += 1
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            indices = indices[worker.id :: worker.num_workers]
        allowed = {path.name for path in self.files}
        manifest = json.loads(self.split_manifest.read_text())
        files = [
            self.split_manifest.parent / item["file"] for item in manifest["files"]
        ]
        with ExitStack() as stack:
            handles = {}
            directory = self.files[0].parent
            for index in indices:
                record = records[index]
                filename = files[int(record["file"])].name
                if filename not in allowed:
                    continue
                if filename not in handles:
                    handles[filename] = stack.enter_context(
                        h5py.File(directory / filename, "r")
                    )
                name = record["path"].decode()
                yield filename, name, handles[filename][name]

    def _legacy_records(self):
        """Keep the former split available for auditing preexisting control runs."""
        worker = torch.utils.data.get_worker_info()
        files = (
            self.files
            if worker is None
            else self.files[worker.id :: worker.num_workers]
        )
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for path in files:
            with h5py.File(unpack_bms_shard(path), "r") as handle:
                systems = list(handle)
                rng.shuffle(systems)
                for system in systems:
                    key = f"{self.seed}:{path.name.rsplit('_', 1)[0]}:{system}".encode()
                    validation = int(hashlib.sha256(key).hexdigest()[:8], 16) % 20 == 0
                    if validation != (self.split == "valid"):
                        continue
                    conformers = list(handle[system])
                    rng.shuffle(conformers)
                    for conformation in conformers:
                        name = f"{system}/{conformation}"
                        yield path.name, name, handle[name]

    def __iter__(self):
        records = (
            self._legacy_records()
            if self.split_manifest is None
            else self._indexed_records()
        )
        for filename, name, group in records:
            graph = self._graph_from_group(filename, name, group)
            if graph is not None:
                yield graph

    def _graph_from_group(self, filename, name, group):
        MM_shape = group["coords_mm"].shape
        if group["grad_mm"].shape != MM_shape or group["charges_mm"].shape != (
            MM_shape[0],
        ):
            warnings.warn(
                f"Skipping inconsistent BMS25 record {filename}:{name}: "
                f"coords_mm={MM_shape}, grad_mm={group['grad_mm'].shape}, "
                f"charges_mm={group['charges_mm'].shape}"
            )
            return None
        scale = 96.48533212331002  # kJ/mol per eV
        properties = {
            "energy": float(group["e_qmmm"][()]) / scale,
            "forces": -group["grad_qmmm"][()] / scale,
            "total_charge": float(group["charge"][()]),
            "total_spin": 1.0,
        }
        config = Configuration(
            atomic_numbers=group["Z"][()],
            positions=group["coords_qm"][()],
            properties=properties,
            property_weights={},
            pbc=np.zeros(3, dtype=bool),
            cell=np.zeros((3, 3)),
        )
        graph = AtomicData.from_config(config, self.z_table, self.cutoff)
        dtype = graph.positions.dtype
        graph.MM_coordinate = torch.as_tensor(group["coords_mm"][()], dtype=dtype)
        graph.MM_charges = torch.as_tensor(group["charges_mm"][()], dtype=dtype)
        graph.MM_forces = torch.as_tensor(-group["grad_mm"][()] / scale, dtype=dtype)
        graph.sample_id = f"{filename}:{name}"
        return graph


def collate_bms(graphs):
    batch = Batch.from_data_list(graphs, follow_batch=["MM_coordinate"])
    batch.MM_batch = batch.MM_coordinate_batch
    return batch


class BMSBatches(torch.utils.data.IterableDataset):
    """Bound batches by graph and electrostatic pair counts for large proteins."""

    def __init__(self, dataset, batch_size, max_edges=0, max_pairs=0):
        self.dataset = dataset
        self.batch_size, self.max_edges, self.max_pairs = (
            batch_size,
            max_edges,
            max_pairs,
        )

    def __iter__(self):
        graphs, edges, pairs = [], 0, 0
        for graph in self.dataset:
            next_edges = graph.edge_index.shape[1]
            n = len(graph.positions)
            next_pairs = n * (len(graph.MM_coordinate) + n)
            if graphs and (
                len(graphs) >= self.batch_size
                or (self.max_edges and edges + next_edges > self.max_edges)
                or (self.max_pairs and pairs + next_pairs > self.max_pairs)
            ):
                yield graphs
                graphs, edges, pairs = [], 0, 0
            graphs.append(graph)
            edges += next_edges
            pairs += next_pairs
        if graphs:
            yield graphs


class BMSIndexedDataset(torch.utils.data.Dataset):
    """Random access to a fixed paper split; HDF5 files are opened per worker."""

    def __init__(
        self, directory, atomic_numbers, cutoff, split_manifest, split="train"
    ):
        self.dataset = BMSDataset(
            directory,
            atomic_numbers,
            cutoff,
            split=split,
            split_manifest=split_manifest,
        )
        self.records = self.indices = None
        self.handles = {}

    def __len__(self):
        return self.dataset.split_count

    def __getitem__(self, index):
        if self.records is None:
            root = self.dataset.split_manifest.parent
            self.records = np.load(root / "records.npy", mmap_mode="r")
            self.indices = np.load(root / f"{self.dataset.split}.npy", mmap_mode="r")
        record = self.records[self.indices[index]]
        path = self.dataset.files[int(record["file"])]
        if path.name not in self.handles:
            self.handles[path.name] = h5py.File(path, "r")
        name = record["path"].decode()
        graph = self.dataset._graph_from_group(
            path.name, name, self.handles[path.name][name]
        )
        if graph is None:
            raise ValueError(
                f"Invalid record in verified BMS25 split: {path.name}:{name}"
            )
        return graph


class BMSDistributedBatchSampler(torch.utils.data.Sampler):
    """Partition global batches across ranks without duplicating or dropping frames."""

    def __init__(self, size, batch_size, rank, world_size, seed=1, shuffle=True):
        if size < world_size:
            raise ValueError("A BMS25 split must contain at least one frame per rank")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.shuffle = seed, shuffle
        self.epoch, self.skip_batches = 0, 0

    def __len__(self):
        batch_size = self.batch_size * self.world_size
        count, remainder = divmod(self.size, batch_size)
        return count + int(remainder >= self.world_size or count == 0)

    def __iter__(self):
        indices = np.arange(self.size)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        batch_size = self.batch_size * self.world_size
        start, step = 0, 0
        while start < self.size:
            end = min(start + batch_size, self.size)
            if self.size - end < self.world_size:
                end = self.size
            if step >= self.skip_batches:
                yield indices[start:end][self.rank :: self.world_size].tolist()
            start, step = end, step + 1


def collate_bms_graphs(graphs):
    return graphs


def collate_bms_microbatches(graphs, batch_size, max_edges=0, max_pairs=0):
    """Collate in the worker to avoid transferring thousands of individual tensors."""
    return [
        collate_bms(items)
        for items in BMSBatches(graphs, batch_size, max_edges, max_pairs)
    ]
