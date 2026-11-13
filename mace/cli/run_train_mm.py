"""Fine-tune MACE-POLAR on BMS25 energy, QM forces, and MM forces."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from schedulefree import AdamWScheduleFree

from mace.calculators.foundations_models import mace_polar
from mace.data.bms_dataset import BMSBatches, BMSDataset, collate_bms
from mace.modules.loss import WeightedEnergyForcesL1L2Loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--foundation_model", default="polar-1-m")
    parser.add_argument("--fukui_regularisation", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_num_epochs", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--energy_weight", type=float, default=10.0)
    parser.add_argument("--forces_weight", type=float, default=10.0)
    parser.add_argument("--MM_forces_weight", type=float, default=10.0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--subsets", nargs="*", help="Optional shard prefixes for benchmarks"
    )
    parser.add_argument("--max_edges", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument(
        "--enable_cueq", type=lambda s: s.lower() == "true", default=True
    )
    parser.add_argument(
        "--cueq_conv_fusion", type=lambda s: s.lower() == "true", default=False
    )
    args = parser.parse_args()
    args.split_manifest = str(Path(args.split_manifest).resolve())
    split_manifest = json.loads(Path(args.split_manifest).read_text())
    for filename, expected in split_manifest["sha256"].items():
        actual = hashlib.sha256(
            (Path(args.split_manifest).parent / filename).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"BMS25 split index checksum mismatch: {filename}")
    args.split_manifest_sha256 = hashlib.sha256(
        Path(args.split_manifest).read_bytes()
    ).hexdigest()
    if args.fukui_regularisation is not None and not 0 <= args.fukui_regularisation < 1:
        parser.error("--fukui_regularisation requires 0 <= value < 1")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir(exist_ok=True)
    (work / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    logging.basicConfig(level=logging.INFO)
    model = mace_polar(args.foundation_model, device="cpu", return_raw_model=True)
    model = model.float()
    if args.fukui_regularisation is not None:
        model.fukui_regularisation = args.fukui_regularisation
    if args.enable_cueq:
        from mace.cli.convert_e3nn_cueq import run

        model = run(model, device=device, conv_fusion=args.cueq_conv_fusion)
        assert model.cueq_config.enabled
        assert model.cueq_config.conv_fusion == args.cueq_conv_fusion
        print(f"cuEq enabled; conv_fusion={model.cueq_config.conv_fusion}", flush=True)
    model = model.to(device=device, dtype=torch.float32)
    assert all(
        p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point()
    )
    print("Precision: float32 parameters and inputs; autocast disabled", flush=True)
    model.set_realspace_method("analytical")
    if args.fukui_regularisation is not None:
        assert model.fukui_regularisation == args.fukui_regularisation
    print(f"Model fukui_regularisation={model.fukui_regularisation}", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    assert not getattr(model, "recompute_field_after_scf", False)
    print(
        f"Loaded {args.foundation_model}; {sum(p.numel() for p in model.parameters())} parameters; {torch.cuda.get_device_name()}",
        flush=True,
    )
    print(
        "Loss: per-QM-atom energy L1 + QM force component MSE + MM force component MSE",
        flush=True,
    )
    print(
        "Original model modules and field-refresh setting preserved; MM sources are point charges.",
        flush=True,
    )
    datasets = [
        BMSDataset(
            args.train_file,
            model.atomic_numbers.tolist(),
            float(model.r_max),
            split,
            args.seed,
            split_manifest=args.split_manifest,
        )
        for split in ("train", "valid")
    ]
    if args.subsets:
        for dataset in datasets:
            dataset.files = [
                p
                for p in dataset.files
                if any(p.name.startswith(s) for s in args.subsets)
            ]
    print(
        f"BMS25 split: {split_manifest['policy']}; counts={split_manifest['counts']}; "
        "training conformations globally shuffled without replacement; test excluded",
        flush=True,
    )
    loaders = [
        torch.utils.data.DataLoader(
            BMSBatches(dataset, args.batch_size, args.max_edges, args.max_pairs),
            batch_size=None,
            num_workers=args.num_workers,
            collate_fn=collate_bms,
            pin_memory=True,
            **(
                {"multiprocessing_context": "spawn", "persistent_workers": True}
                if args.num_workers
                else {}
            ),
        )
        for dataset in datasets
    ]
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        warmup_steps=2000,
        weight_decay=1e-8,
    )
    loss_fn = WeightedEnergyForcesL1L2Loss(
        args.energy_weight, args.forces_weight, args.MM_forces_weight
    ).to(device)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    deadline = int(os.environ.get("MACE_TRAIN_STOP_AT", "0"))
    step = 0
    epoch = 0
    seen = 0
    resume = work / "checkpoints" / "latest.pt"
    if resume.exists():
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint["args"].get("fukui_regularisation") != args.fukui_regularisation:
            raise ValueError(
                "Checkpoint Fukui regularisation differs; use a new work_dir"
            )
        if (
            checkpoint["args"].get("split_manifest_sha256")
            != args.split_manifest_sha256
        ):
            raise ValueError("Checkpoint data split differs; use a new work_dir")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, epoch, seen = checkpoint["step"], checkpoint["epoch"], checkpoint["seen"]
        print(f"Resuming step {step}, epoch {epoch}", flush=True)
    else:
        print("Fresh run from foundation weights; optimizer step=0, seen=0", flush=True)

    def save():
        temporary = resume.with_suffix(".pt.tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "seen": seen,
                "args": vars(args),
            },
            temporary,
        )
        temporary.replace(resume)

    def metrics(ref, pred):
        sizes = ref.ptr[1:] - ref.ptr[:-1]
        return {
            "energy_mae_per_atom_eV": float(
                ((pred["energy"] - ref.energy) / sizes).abs().mean().detach()
            ),
            "forces_rmse_eV_A": float(
                (pred["forces"] - ref.forces).square().mean().sqrt().detach()
            ),
            "MM_forces_rmse_eV_A": float(
                (pred["MM_forces"] - ref.MM_forces).square().mean().sqrt().detach()
            ),
        }

    try:
        model.train()
        optimizer.train()
        for epoch in range(epoch, args.max_num_epochs):
            last_step = time.perf_counter()
            for ref in loaders[0]:
                if (
                    stop[0]
                    or (deadline and time.time() >= deadline)
                    or (args.max_steps and step >= args.max_steps)
                ):
                    return
                loaded = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                ref = ref.to(device, non_blocking=True)
                if step == 0:
                    assert (
                        ref.positions.dtype
                        == ref.MM_coordinate.dtype
                        == ref.MM_charges.dtype
                        == torch.float32
                    )
                    print(f"First training batch IDs: {ref.sample_id[:12]}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(ref.to_dict(), training=True)
                loss = loss_fn(ref, pred)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {ref.sample_id}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        f"Nonfinite parameter gradients: {ref.sample_id}"
                    )
                optimizer.step()
                torch.cuda.synchronize()
                finished = time.perf_counter()
                step += 1
                seen += len(ref.ptr) - 1
                if step == 1 or step % args.log_interval == 0:
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "seen": seen,
                        "loss": float(loss.detach()),
                        "step_seconds": finished - loaded,
                        "batch_seconds": finished - last_step,
                        "peak_memory_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                        "QM_atoms": len(ref.positions),
                        "QM_edges": ref.edge_index.shape[1],
                        "batch_size": len(ref.ptr) - 1,
                        "MM_atoms": len(ref.MM_coordinate),
                        **metrics(ref, pred),
                    }
                    print(json.dumps(entry), flush=True)
                    with (work / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(entry) + "\n")
                if step % args.save_interval == 0:
                    save()
                    optimizer.eval()
                    model.eval()
                    valid = next(iter(loaders[1])).to(device)
                    result = model(valid.to_dict())
                    print(
                        json.dumps({"validation_step": step, **metrics(valid, result)}),
                        flush=True,
                    )
                    optimizer.train()
                    model.train()
                last_step = time.perf_counter()
    finally:
        save()
        optimizer.eval()
        model.eval()
        torch.save(model.to("cpu"), work / "mace_polar_bms25.model")
        print(
            f"Saved checkpoint and evaluation model at step {step}, {seen} training samples.",
            flush=True,
        )


if __name__ == "__main__":
    main()
