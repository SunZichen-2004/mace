"""Unified MACE-POLAR-2 training records and AtomicData graphs.

All persistent arrays, manifests and sources are on Scratch. This is a dataset
adapter, not a training launcher. Native 140M LMDBs are normalized on read.
"""

import json, os, zlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
import h5py, lmdb

ROOT = Path(
    os.environ.get(
        "POLAR2_DATA_ROOT", "/dais/fs/scratch/sunzc/MACE-POLAR-2/training_data_v2"
    )
)
BOHR = 0.529177210903
EV_TO_KJMOL = 96.48533212331002
PROPERTY_SHAPES = {
    "energy": (),
    "dipole": (3,),
    "quadrupole": (3, 3),
    "spin_dipole": (3,),
    "spin_quadrupole": (3, 3),
    "polarizability": (3, 3),
}


def unpack(x):
    if isinstance(x, dict) and "__ndarray__" in x:
        shape, dtype, data = x["__ndarray__"]
        return np.array(data, dtype=dtype).reshape(shape)
    return np.asarray(x)


def six_to_matrix(v):
    x, y, z, xy, xz, yz = v
    return np.array([[x, xy, xz], [xy, y, yz], [xz, yz, z]])


def traceless(q):
    return q - np.eye(3) * np.trace(q) / 3


def translate_moments(mu, q, charge, origin):
    return mu + charge * origin, traceless(
        q
        + np.outer(mu, origin)
        + np.outer(origin, mu)
        + charge * np.outer(origin, origin)
    )


class UnifiedDataset:
    """Map-style dataset. split='train' by default; explicit test/all for evaluation.

    graph=False returns numpy records. graph=True returns existing MACE AtomicData.
    Geometries are centered at ORCA COM, using the recorded 4M origin when
    available and the ORCA mass table otherwise. Molecular moments and spin
    labels are expressed in that same frame; missing targets are masked.
    """

    def __init__(
        self,
        name,
        split="train",
        graph=True,
        cutoff=5.0,
        atomic_numbers=None,
        root=None,
        heads=None,
    ):
        self.name = name
        self.split = split
        self.graph = graph
        self.cutoff = cutoff
        self.atomic_numbers = (
            list(range(1, 84)) if atomic_numbers is None else list(atomic_numbers)
        )
        self.root = Path(root) if root else ROOT
        self.orca_masses = (
            json.loads((self.root / "orca_masses.json").read_text())
            if (self.root / "orca_masses.json").exists()
            else {}
        )
        self.catalog = json.loads((self.root / "catalog.json").read_text())
        self.spec = self.catalog["datasets"][name]
        cache_root = os.environ.get("POLAR2_NODE_CACHE")
        self.node_cache = None
        if cache_root and name in os.environ.get("POLAR2_CACHE_SOURCES", "spice_alpha,bms25").split(","):
            manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
            if not manifest.get("complete"):
                raise ValueError("POLAR-2 node cache has not completed verification")
            self.node_cache = manifest["files"]
        self.head = self.spec["head"]
        self.heads = heads if heads is not None else self.catalog["heads"]
        self.pid = os.getpid()
        self.cache = OrderedDict()
        self.index_cache = {}
        self.spin_dataset = None
        self.shards = self.spec["shards"]
        self.lengths = []
        if split not in self.spec["splits"]:
            raise ValueError(
                f'{name}: split {split} unavailable; available={self.spec["splits"]}'
            )
        for s in self.shards:
            self.lengths.append(s["counts"][split])
        self.ends = np.cumsum(self.lengths, dtype=np.int64)

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        d = dict(self.__dict__)
        d["cache"] = OrderedDict()
        d["index_cache"] = {}
        d["spin_dataset"] = None
        return d

    def close(self):
        for h in self.cache.values():
            h.close()
        self.cache.clear()
        self.index_cache.clear()
        if self.spin_dataset is not None:
            self.spin_dataset.close()
            self.spin_dataset = None

    def __del__(self):
        try:
            self.close()
        except (AttributeError, TypeError):
            pass

    def _cached_path(self, path):
        if self.node_cache is None:
            return path
        entry = self.node_cache.get(str(path))
        if entry is None or not entry.get("verified"):
            raise ValueError(f"Missing verified POLAR-2 node-cache entry: {path}")
        return entry["path"]

    def _handle(self, path, kind):
        path = self._cached_path(path)
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        key = (path, kind)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if len(self.cache) >= int(os.environ.get("POLAR2_MAX_OPEN_FILES", "8")):
            _, old = self.cache.popitem(last=False)
            old.close()
        if kind == "h5":
            h = h5py.File(path, "r")
        elif kind == "lmdb":
            h = lmdb.open(
                path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=512,
            )
        else:
            h = open(path, "rb")
        self.cache[key] = h
        return h

    def _npy(self, path):
        path = self._cached_path(path)
        if path not in self.index_cache:
            self.index_cache[path] = np.load(path, mmap_mode="r")
        return self.index_cache[path]

    def record(self, index):
        if index < 0:
            index += int(self.ends[-1])
        if index < 0 or index >= int(self.ends[-1]):
            raise IndexError(index)
        shard = int(np.searchsorted(self.ends, index, side="right"))
        local = int(index - (self.ends[shard - 1] if shard else 0))
        s = self.shards[shard]
        if self.split in s.get("selectors", {}):
            local = int(self._npy(s["selectors"][self.split])[local])
        kind = self.spec["kind"]
        masks = {k: 0.0 for k in [*PROPERTY_SHAPES, "forces", "MM_forces"]}
        props = {}
        meta = {}
        mm_pos = np.empty((0, 3))
        mm_q = np.empty(0)
        mm_f = np.empty((0, 3))
        origin = np.zeros(3)
        if kind == "grouped_hdf5":
            group, row = self._npy(s["index"])[local]
            h = self._handle(s["source"], "h5")[str(int(group))]
            row = int(row)
            z = h["atomic_numbers"][...]
            pos = h["positions"][row]
            q = int(h.attrs["total_charge"])
            mult = int(h.attrs["total_spin"])
            for key in ["energy", "forces", "dipole", "polarizability"]:
                props[key] = np.asarray(h[key][row])
                masks[key] = 1.0
            stored_unit = h.file.attrs.get("polarizability_unit")
            if stored_unit != "e*Angstrom^2/volt":
                raise ValueError(f"Unexpected SPICE-alpha unit: {stored_unit}")
            props["polarizability"] = props["polarizability"] * 14.3996454784255
            identifier = f'{Path(s["source"]).name}:{h.attrs["source_group"]}:{int(h["source_row"][row])}'
            meta = {
                "source_group": str(h.attrs["source_group"]),
                "original_spice_group": str(h.attrs["original_spice_group"]),
                "original_conformer": int(h["original_conformer"][row]),
                "multiplicity_provenance": "SPICE input convention verified against original mapped SMILES; not independently re-read from ORCA wavefunction",
            }
        elif kind == "indexed_xyz":
            from ase.io.extxyz import key_val_str_to_dict
            from ase.data import atomic_numbers as symbol_to_z

            f = self._handle(s["source"], "xyz")
            f.seek(int(self._npy(s["index"])[local]))
            n = int(f.readline())
            info = key_val_str_to_dict(f.readline().decode())
            lines = [f.readline().split() for _ in range(n)]
            z = np.array([symbol_to_z[x[0].decode()] for x in lines])
            pos = np.array([[float(v) for v in x[1:4]] for x in lines])
            q = int(info["total_charge_e"])
            mult = int(info["multiplicity"])
            props["energy"] = np.asarray(info["energy"])
            props["forces"] = np.array([[float(v) for v in x[4:7]] for x in lines])
            masks["energy"] = masks["forces"] = 1.0
            if info.get("status_dipole") == "OK":
                props["dipole"] = np.array([info["dipole_" + a + "_eA"] for a in "xyz"])
                masks["dipole"] = 1.0
            if info.get("status_quadrupole") == "OK":
                props["quadrupole"] = np.asarray(info["quadrupole_eA_traceless"])
                masks["quadrupole"] = 1.0
            if info.get("status_spin_multipole") == "OK":
                for k, field in [
                    ("spin_dipole", "spin_dipole_eA"),
                    ("spin_quadrupole", "spin_quadrupole_traceless"),
                ]:
                    props[k] = np.asarray(info[field])
                    masks[k] = 1.0
            identifier = str(info["argonne_rel"])
            meta = {
                "orca_origin_A": [float(info["R_orca_origin_" + a]) for a in "xyz"],
                "spin_case": str(info.get("spin_case", "unknown")),
                "spin_electrons": float(info.get("spin_electrons", 0.0)),
                "source_row": local,
            }
        elif kind == "native_lmdb":
            row = local + 1
            e = self._handle(s["source"], "lmdb")
            with e.begin() as t:
                d = json.loads(zlib.decompress(t.get(str(row).encode())))
            a = d["data"]
            z = unpack(d["numbers"])
            pos = unpack(d["positions"])
            q = int(a["charge"])
            mult = int(a["spin"])
            props["energy"] = np.asarray(d["energy"])
            props["forces"] = unpack(d["forces"])
            masks["energy"] = masks["forces"] = 1.0
            raw = a.get("multipoles")
            if raw is not None and len(raw) >= 3:
                mu = np.asarray(raw[1], dtype=float) * BOHR
                quad = six_to_matrix(raw[2]) * BOHR**2
                if np.isfinite(mu).all() and np.isfinite(quad).all():
                    props["dipole"], props["quadrupole"] = mu, traceless(quad)
                    masks["dipole"] = masks["quadrupole"] = 1.0
            identifier = f'{Path(s["source"]).name}:{row}:{a["source"]}'
            meta = {
                "source": a["source"],
                "data_id": a["data_id"],
                "unrestricted": bool(a["unrestricted"]),
                "native_origin_convention": "ORCA COM; raw moments[0] is not a physical origin",
                "native_multipoles_au": raw,
            }
        elif kind == "indexed_bms":
            idx = self._npy(s["index"])[local]
            src = self.spec["source_files"][int(idx["file"])]
            name = idx["path"].decode()
            g = self._handle(src, "h5")[name]
            z = g["Z"][...].astype(int)
            pos = g["coords_qm"][...]
            q = int(g["charge"][()])
            mult = 1
            props["energy"] = np.asarray(g["e_qmmm"][()] / EV_TO_KJMOL)
            props["forces"] = -g["grad_qmmm"][...] / EV_TO_KJMOL
            masks["energy"] = masks["forces"] = 1.0
            mu = g["dipo_qmmm"][...]
            props["dipole"] = mu
            masks["dipole"] = 1.0
            # Raw QM quadrupole retained as provenance; no enabled quadrupole target
            # until its traceless/primitive convention is independently verified.
            mm_pos = g["coords_mm"][...]
            mm_q = g["charges_mm"][...]
            mm_f = -g["grad_mm"][...] / EV_TO_KJMOL
            masks["MM_forces"] = 1.0
            identifier = f"{Path(src).name}:{name}"
            meta = {
                "native_QM_dipole_eA": mu.tolist(),
                "native_origin_convention": "ORCA default QM COM, fixed MM environment",
                "multiplicity_provenance": "BMS released calculation setup: singlet",
            }
        else:
            raise ValueError(kind)
        for key, shape in PROPERTY_SHAPES.items():
            value = np.asarray(props.get(key, np.zeros(shape)), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                if key == "energy":
                    raise ValueError((identifier, key, "invalid required label"))
                value = np.zeros(shape)
                masks[key] = 0.0
            props[key] = value
        assert np.isfinite(pos).all() and np.isfinite(props["forces"]).all()
        # Translate the complete system to the ORCA COM. Electric and spin
        # quadrupoles use their corresponding dipole cross terms before detracing.
        from ase.data import atomic_masses

        if kind == "indexed_xyz":
            origin = np.asarray(meta["orca_origin_A"])
        else:
            masses = self.orca_masses
            weights = np.array(
                [masses.get(str(int(zz)), float(atomic_masses[zz])) for zz in z]
            )
            origin = np.average(pos, axis=0, weights=weights)
        if self.name == "omol140m" and self.split == "train":
            overlay = self.root / "spin_overlay" / (Path(s["source"]).stem + ".npy")
            if overlay.exists():
                spin_index = int(self._npy(str(overlay))[local])
                if spin_index >= 0:
                    if self.spin_dataset is None:
                        self.spin_dataset = UnifiedDataset(
                            "omol4m", split="all", graph=False, root=self.root
                        )
                    spin_record = self.spin_dataset.record(spin_index)
                    geom = spin_record["positions"] + np.asarray(
                        spin_record["metadata"]["coordinate_translation_A"]
                    )
                    state_ok = (
                        np.array_equal(z, spin_record["atomic_numbers"])
                        and geom.shape == pos.shape
                        and np.max(abs(geom - pos)) < 2e-6
                        and abs(
                            float(props["energy"])
                            - float(spin_record["properties"]["energy"])
                        )
                        < 1e-5
                        and np.max(
                            abs(props["forces"] - spin_record["properties"]["forces"])
                        )
                        < 2e-6
                        and bool(a["unrestricted"])
                        == (spin_record["metadata"]["spin_case"] == "UKS")
                    )
                    meta["spin_overlay_state_verified"] = bool(state_ok)
                    if state_ok:
                        origin = np.asarray(
                            spin_record["metadata"]["coordinate_translation_A"]
                        )
                        for key in ["spin_dipole", "spin_quadrupole"]:
                            props[key] = spin_record["properties"][key]
                            masks[key] = spin_record["property_weights"][key]
        if kind == "indexed_xyz":
            for dk, qk, total in [
                ("dipole", "quadrupole", q),
                ("spin_dipole", "spin_quadrupole", meta["spin_electrons"]),
            ]:
                if not masks[dk] or not np.isfinite(total):
                    masks[dk] = masks[qk] = 0.0
                    props[dk] = np.zeros(3)
                    props[qk] = np.zeros((3, 3))
                    continue
                mu = props[dk].copy()
                props[qk] = traceless(
                    props[qk]
                    - np.outer(mu, origin)
                    - np.outer(origin, mu)
                    + total * np.outer(origin, origin)
                )
                props[dk] = mu - total * origin
        # SPICE-alpha dipoles are not part of this dataset's supervision: only the
        # full polarizability tensor is used, independently of its origin.
        if self.name == "spice_alpha":
            for k in masks:
                masks[k] = float(k == "polarizability")
        pos = pos - origin
        mm_pos = mm_pos - origin
        meta["coordinate_translation_A"] = origin.tolist()
        if kind in ("native_lmdb", "indexed_bms"):
            meta["native_moment_origin_reconstructed_A"] = origin.tolist()
        if any(not np.isfinite(v).all() for v in props.values()):
            raise ValueError(f"Non-finite normalized label: {identifier}")
        origin = np.zeros(3)
        return {
            "atomic_numbers": z.astype(np.int64),
            "positions": pos.astype(float),
            "properties": props,
            "property_weights": masks,
            "total_charge": q,
            "total_spin": mult,
            "head": self.head,
            "sample_id": identifier,
            "multipole_origin": origin,
            "MM_coordinate": mm_pos,
            "MM_charges": mm_q,
            "MM_forces": mm_f,
            "metadata": meta,
        }

    def __getitem__(self, index):
        r = self.record(index)
        if not self.graph:
            return r
        import torch
        from mace.data.atomic_data import AtomicData
        from mace.data.utils import Configuration
        from mace.tools import AtomicNumberTable

        properties = dict(
            r["properties"],
            total_charge=float(r["total_charge"]),
            total_spin=float(r["total_spin"]),
        )
        weights = {
            k: 0.0 for k in ["stress", "virials", "charges", "magmom", "magforces"]
        }
        weights.update(r["property_weights"])
        config = Configuration(
            atomic_numbers=r["atomic_numbers"],
            positions=r["positions"],
            properties=properties,
            property_weights=weights,
            head=self.head,
            pbc=np.zeros(3, dtype=bool),
            cell=np.zeros((3, 3)),
        )
        graph = AtomicData.from_config(
            config,
            AtomicNumberTable(self.atomic_numbers),
            self.cutoff,
            heads=self.heads,
        )
        dtype = graph.positions.dtype
        for key in ["quadrupole", "spin_dipole", "spin_quadrupole", "MM_forces"]:
            setattr(graph, key + "_weight", torch.tensor(weights[key], dtype=dtype))
        for key in ["MM_coordinate", "MM_charges", "MM_forces"]:
            setattr(graph, key, torch.as_tensor(r[key], dtype=dtype))
        graph.multipole_origin = torch.as_tensor(
            r["multipole_origin"], dtype=dtype
        ).reshape(1, 3)
        graph.source_name = self.name
        graph.sample_id = r["sample_id"]
        return graph


def collate_graphs(graphs):
    from mace.tools.torch_geometric.batch import Batch

    batch = Batch.from_data_list(graphs, follow_batch=["MM_coordinate"])
    batch.MM_batch = batch.MM_coordinate_batch
    return batch


class Polar2Dataset(UnifiedDataset):
    """Source-native reader selected by a small .polar2 JSON descriptor."""

    def __init__(self, descriptor, cutoff, atomic_numbers, heads):
        spec = json.loads(Path(descriptor).read_text())
        super().__init__(
            spec["dataset"],
            spec.get("split", "train"),
            cutoff=cutoff,
            atomic_numbers=atomic_numbers,
            root=spec["root"],
            heads=heads,
        )
        self.sample_probability = spec.get("probability", 1.0)
        self.batch_size = spec.get("batch_size", 1)
        self.limit = spec.get("limit")

    def __len__(self):
        full = super().__len__()
        return min(full, self.limit) if self.limit else full

    def record(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Spread a diagnostic validation cap over the entire source/split.
        # Taking its first rows would overrepresent the first chemistry shard.
        full = int(self.ends[-1])
        if len(self) < full:
            index = index * (full - 1) // max(1, len(self) - 1)
        return super().record(index)


def dataset_leaves(dataset, offset=0):
    from torch.utils.data import ConcatDataset

    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from dataset_leaves(child, offset)
            offset += len(child)
    else:
        yield dataset, offset


class Polar2BatchSampler:
    """All ranks draw the same source; rank examples are disjoint per step.

    steps=-1 makes a complete shuffled pass through each active source.
    Positive steps retain the legacy virtual-epoch sampler for pilot experiments.
    """

    def __init__(
        self, dataset, steps=1000, rank=0, world_size=1, seed=1, validation=False
    ):
        self.leaves = list(dataset_leaves(dataset))
        self.steps = steps
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.validation = validation
        if any(not isinstance(d, Polar2Dataset) for d, _ in self.leaves):
            raise TypeError("POLAR-2 sampling requires .polar2 descriptors")

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.full_plan = None

    def __len__(self):
        if not self.validation:
            if self.steps == -1:
                from mace.data.polar2_epoch import epoch_plan
                return sum(x["steps"] for x in epoch_plan(self) if x["name"] not in os.environ.get("POLAR2_DISABLED_SOURCES", "").split(","))
            return self.steps
        return sum(
            max(0, (len(d) - self.rank + self.world_size - 1) // self.world_size + d.batch_size - 1) // d.batch_size
            for d, _ in self.leaves
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.validation:
            for d, offset in self.leaves:
                # Each real validation record appears once, including the tail.
                # Evaluation uses the unwrapped model and reduces only at the end,
                # so ranks may have different numbers of validation batches.
                local_ids = range(self.rank, len(d), self.world_size)
                for start in range(0, len(local_ids), d.batch_size):
                    yield [offset + i for i in local_ids[start:start + d.batch_size]]
            return
        if self.steps == -1:
            from mace.data.polar2_epoch import full_epoch_batches
            yield from full_epoch_batches(self)
            return
        probs = np.array([d.sample_probability for d, _ in self.leaves])
        control = os.environ.get("POLAR2_SAMPLING_CONTROL")
        if control:
            weights = json.loads(Path(control).read_text())["weights"]
            probs = np.array([weights.get(d.name, d.sample_probability) for d, _ in self.leaves])
        if not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError("Invalid POLAR-2 source sampling probabilities")
        if os.environ.get("POLAR2_SAMPLE_BY_RECORD_COUNT") == "1":
            # Source weights describe expected numbers of records, not batches.
            # Smaller OPoly batches must be selected more often to keep this ratio.
            probs /= np.array([d.batch_size for d, _ in self.leaves], dtype=float)
        probs /= probs.sum()
        for _ in range(self.steps):
            d, offset = self.leaves[int(rng.choice(len(self.leaves), p=probs))]
            ids = rng.choice(len(d), size=d.batch_size * self.world_size, replace=False)
            yield (
                ids[self.rank * d.batch_size : (self.rank + 1) * d.batch_size] + offset
            ).tolist()


class Polar2DataLoader:
    def __new__(cls, dataset, **kwargs):
        from torch.utils.data import DataLoader

        import torch

        kwargs.setdefault("generator", torch.Generator().manual_seed(1))
        if kwargs.get("num_workers", 0) > 0:
            kwargs.setdefault("persistent_workers", False)
            kwargs.setdefault("prefetch_factor", int(os.environ.get("POLAR2_PREFETCH_FACTOR", "1")))
        return DataLoader(dataset, collate_fn=collate_graphs, **kwargs)
