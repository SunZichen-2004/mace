"""Persist the documented AMP-BMS split; author frame indices are not published."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

TRAIN = {
    "MONOPEPTIDES",
    "DIPEPTIDES",
    "TRIPEPTIDES",
    "REACTIONS",
    "SMALL_MOLECULES",
    "DIMERS",
}
TEST = {"PEPTIDES", "MINI_PROTEINS"}


def scan_file(argument):
    path, output = map(Path, argument)
    category = path.stem.rsplit("_", 1)[0]
    assert category in TRAIN | TEST, path
    accepted, excluded = [], {
        "gradient_threshold": 0,
        "inconsistent_labels": 0,
        "nonfinite_labels": 0,
    }
    total = 0
    with h5py.File(path, "r") as handle:
        for system in sorted(handle):
            for conformer in sorted(handle[system]):
                total += 1
                group = handle[system][conformer]
                qm_shape, mm_shape = group["coords_qm"].shape, group["coords_mm"].shape
                if (
                    group["grad_qmmm"].shape != qm_shape
                    or group["grad_mm"].shape != mm_shape
                    or group["charges_mm"].shape != (mm_shape[0],)
                ):
                    excluded["inconsistent_labels"] += 1
                    continue
                gradients = group["grad_qmmm"][()]
                # Apply the published inclusion criterion to all three splits.
                if not np.isfinite(gradients).all():
                    excluded["nonfinite_labels"] += 1
                    continue
                if np.max(np.abs(gradients), initial=0) >= 2000:
                    excluded["gradient_threshold"] += 1
                    continue
                if not (
                    np.isfinite(group["e_qmmm"][()]).all()
                    and np.isfinite(group["grad_mm"][()]).all()
                ):
                    excluded["nonfinite_labels"] += 1
                    continue
                accepted.append(f"{system}/{conformer}".encode())
    np.save(output / (path.stem + ".npy"), np.asarray(accepted, dtype="S"))
    result = {
        "file": path.name,
        "category": category,
        "raw": total,
        "accepted": len(accepted),
        "excluded": excluded,
        "bytes": path.stat().st_size,
    }
    (output / (path.stem + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.directory).glob("*.h5"))
    assert len(files) == 22
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_file, (str(path), str(output))) for path in files]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    reports = [
        json.loads((output / (path.stem + ".json")).read_text()) for path in files
    ]
    paths = [np.load(output / (path.stem + ".npy"), mmap_mode="r") for path in files]
    dtype = np.dtype(
        [("file", "u1"), ("path", f"S{max(p.dtype.itemsize for p in paths)}")]
    )
    records = np.empty(sum(map(len, paths)), dtype=dtype)
    offset = 0
    test_mask = np.zeros(len(records), dtype=bool)
    for i, (path, report) in enumerate(zip(paths, reports)):
        records["file"][offset : offset + len(path)] = i
        records["path"][offset : offset + len(path)] = path
        test_mask[offset : offset + len(path)] = report["category"] in TEST
        offset += len(path)
    candidates = np.flatnonzero(~test_mask)
    rng = np.random.default_rng(args.seed)
    # Exact random 1% of the eligible training-category conformations, without replacement.
    valid = np.sort(
        rng.choice(candidates, size=int(np.ceil(0.01 * len(candidates))), replace=False)
    )
    valid_mask = np.zeros(len(records), dtype=bool)
    valid_mask[valid] = True
    splits = {
        "train": np.flatnonzero(~test_mask & ~valid_mask),
        "valid": valid,
        "test": np.flatnonzero(test_mask),
    }
    assert sum(map(len, splits.values())) == len(records)
    np.save(output / "records.npy", records)
    for name, indices in splits.items():
        np.save(output / (name + ".npy"), indices)
    manifest = {
        "policy": "amp-bms-jacs-2026-random-1pct-v1",
        "paper": "https://doi.org/10.1021/jacs.6c00217",
        "author_indices_available": False,
        "seed": args.seed,
        "validation_fraction": 0.01,
        "split_unit": "conformation",
        "gradient_limit_kJ_mol_A": 2000,
        "test_categories": sorted(TEST),
        "train_valid_categories": sorted(TRAIN),
        "files": reports,
        "counts": {name: len(indices) for name, indices in splits.items()},
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [output / "records.npy", *(output / (n + ".npy") for n in splits)]
        },
        "preparation_seconds": time.time() - start,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
