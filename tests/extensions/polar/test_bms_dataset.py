"""Check the paper split boundary, inclusion filter, and worker coverage."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from mace.cli.prepare_bms import scan_file
from mace.data.bms_dataset import BMSDataset


def write_frame(handle, name, gradient=1.0, missing=False):
    group = handle.create_group(name)
    for key, value in {
        "Z": [1, 8],
        "coords_qm": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "coords_mm": [[3.0, 0.0, 0.0]],
        "charge": 0,
        "e_qmmm": -100.0,
        "grad_qmmm": np.full((2, 3), gradient),
        "charges_mm": [1.0],
        "grad_mm": np.empty(0) if missing else np.ones((1, 3)),
    }.items():
        group[key] = value


def test_paper_gradient_filter_and_missing_labels(tmp_path):
    source = tmp_path / "MONOPEPTIDES_0.h5"
    with h5py.File(source, "w") as handle:
        write_frame(handle, "keep/0", gradient=-1999.9)
        write_frame(handle, "boundary/0", gradient=2000.0)
        write_frame(handle, "large_negative/0", gradient=-2001.0)
        write_frame(handle, "missing/0", missing=True)
        write_frame(handle, "nan/0", gradient=np.nan)
    report = scan_file((str(source), str(tmp_path)))
    assert report["raw"] == 5
    assert report["accepted"] == 1
    assert report["excluded"] == {
        "gradient_threshold": 2,
        "inconsistent_labels": 1,
        "nonfinite_labels": 1,
    }
    assert np.load(tmp_path / "MONOPEPTIDES_0.npy").tolist() == [b"keep/0"]


def test_paper_split_isolation_and_global_worker_shuffle(tmp_path, monkeypatch):
    files = ["MONOPEPTIDES_0.h5", "SMALL_MOLECULES_0.h5", "MINI_PROTEINS_0.h5"]
    records = []
    for file_id, filename in enumerate(files):
        with h5py.File(tmp_path / filename, "w") as handle:
            for i in range(8):
                name = f"molecule{i}/0"
                write_frame(handle, name)
                records.append((file_id, name.encode()))
    np.save(
        tmp_path / "records.npy",
        np.asarray(records, dtype=[("file", "u1"), ("path", "S20")]),
    )
    indices = {
        "train": np.arange(15),
        "valid": np.array([15]),
        "test": np.arange(16, 24),
    }
    for split, values in indices.items():
        np.save(tmp_path / f"{split}.npy", values)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "policy": "amp-bms-jacs-2026-random-1pct-v1",
                "files": [
                    {"file": name, "bytes": (tmp_path / name).stat().st_size}
                    for name in files
                ],
                "counts": {key: len(value) for key, value in indices.items()},
            }
        )
    )

    def dataset(split="train"):
        return BMSDataset(tmp_path, [1, 8], 4.0, split=split, split_manifest=manifest)

    def names(data):
        return [f"{file}:{name}" for file, name, _ in data._indexed_records()]

    training = dataset()
    first, second = names(training), names(training)
    assert len(first) == len(set(first)) == 15
    assert first != second
    assert first == names(dataset())
    assert {name.split(":")[0] for name in first} == set(files[:2])
    assert not set(first) & set(names(dataset("valid")))
    assert {name.split(":")[0] for name in names(dataset("test"))} == {files[2]}
    worker_parts = []
    for worker_id in range(4):
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda i=worker_id: SimpleNamespace(id=i, num_workers=4),
        )
        worker_parts.append(names(dataset()))
    assert set.union(*(set(part) for part in worker_parts)) == set(first)
    assert sum(map(len, worker_parts)) == len(first)
    for worker_id, part in enumerate(worker_parts):
        assert part == first[worker_id::4]
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    graphs = list(dataset("valid"))
    assert len(graphs) == 1
    assert (
        graphs[0].positions.dtype
        == graphs[0].MM_coordinate.dtype
        == graphs[0].MM_charges.dtype
    )
    assert graphs[0].MM_forces.shape == (1, 3)


def test_distributed_batch_sampler_complete_disjoint_equal_steps():
    from mace.data.bms_dataset import BMSDistributedBatchSampler

    for size in [8, 31, 32, 33, 34, 37, 65, 1473, 1337793]:
        samplers = [
            BMSDistributedBatchSampler(size, 4 if size < 100 else 576, rank, 8)
            for rank in range(8)
        ]
        batches = [list(sampler) for sampler in samplers]
        assert len({len(items) for items in batches}) == 1
        assert all(
            len(items) == len(sampler) for items, sampler in zip(batches, samplers)
        )
        flattened = [index for items in batches for batch in items for index in batch]
        assert len(flattened) == len(set(flattened)) == size
        assert set(flattened) == set(range(size))
        assert all(batch for items in batches for batch in items)
        samplers[0].skip_batches = 1
        assert list(samplers[0]) == batches[0][1:]


def test_distributed_component_loss_unequal_microbatches():
    from mace.cli.run_train_mm_ddp import global_component_loss

    # The ranks have unequal force counts and unequal microbatch sizes.
    weights = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    counts = torch.tensor([7.0, 42.0, 90.0], dtype=torch.float64)
    values = [
        torch.tensor([1.0, 4.0, 6.0], dtype=torch.float64),
        torch.tensor([2.0, 3.0, 8.0], dtype=torch.float64),
        torch.tensor([5.0, 2.0, 9.0], dtype=torch.float64),
    ]
    theta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    first = sum(
        global_component_loss(theta.square() * part, counts, weights, 2)
        for part in values[:2]
    )
    second = global_component_loss(theta.square() * values[2], counts, weights, 2)
    ddp_gradient = torch.autograd.grad((first + second) / 2, theta)[0]
    reference = (theta.square() * sum(values) * weights / counts).sum()
    torch.testing.assert_close(ddp_gradient, torch.autograd.grad(reference, theta)[0])
