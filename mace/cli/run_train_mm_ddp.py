"""Distributed BMS25 training with energy L1 and separate QM/MM force MSEs."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from schedulefree import AdamWScheduleFree

from mace.data.bms_dataset import (
    BMSBatches,
    BMSDistributedBatchSampler,
    BMSIndexedDataset,
    collate_bms,
    collate_bms_microbatches,
)


def component_sums(ref, pred):
    sizes = ref.ptr[1:] - ref.ptr[:-1]
    return torch.stack(
        [
            ((pred["energy"] - ref.energy) / sizes).abs().sum(),
            (pred["forces"] - ref.forces).square().sum(),
            (pred["MM_forces"] - ref.MM_forces).square().sum(),
        ]
    )


def global_component_loss(sums, counts, weights, world_size):
    """DDP averages gradients, so compensate by world_size after global normalization."""
    return world_size * (sums * weights / counts.clamp_min(1)).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--initial_model", required=True)
    parser.add_argument("--batch_size", type=int, default=576)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_num_epochs", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--energy_weight", type=float, default=10.0)
    parser.add_argument("--forces_weight", type=float, default=10.0)
    parser.add_argument("--MM_forces_weight", type=float, default=10.0)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--max_edges", type=int, default=450000)
    parser.add_argument("--max_pairs", type=int, default=42000000)
    parser.add_argument(
        "--enable_cueq", type=lambda s: s.lower() == "true", default=True
    )
    parser.add_argument(
        "--cueq_conv_fusion", type=lambda s: s.lower() == "true", default=False
    )
    args = parser.parse_args()
    rank, world_size, local_rank = (
        int(os.environ[k]) for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    work = Path(args.work_dir)
    args.world_size = world_size
    args.split_manifest_sha256 = hashlib.sha256(
        Path(args.split_manifest).read_bytes()
    ).hexdigest()
    args.initial_model_sha256 = hashlib.sha256(
        Path(args.initial_model).read_bytes()
    ).hexdigest()
    manifest = json.loads(Path(args.split_manifest).read_text())
    for name, expected in manifest["sha256"].items():
        assert (
            hashlib.sha256(
                (Path(args.split_manifest).parent / name).read_bytes()
            ).hexdigest()
            == expected
        )
    if rank == 0:
        (work / "checkpoints").mkdir(parents=True, exist_ok=True)
        (work / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    dist.barrier()
    model = torch.load(
        args.initial_model, map_location="cpu", weights_only=False
    ).float()
    assert model.recompute_field_after_scf and model.fukui_regularisation == 0.1
    if args.enable_cueq:
        from mace.cli.convert_e3nn_cueq import run

        model = run(model, device=device, conv_fusion=args.cueq_conv_fusion)
        assert (
            model.cueq_config.enabled
            and model.cueq_config.conv_fusion == args.cueq_conv_fusion
        )
    model = model.to(device=device, dtype=torch.float32)
    model.set_realspace_method("analytical")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    distributed_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        warmup_steps=2000,
        weight_decay=1e-8,
    )
    weights = torch.tensor(
        [args.energy_weight, args.forces_weight, args.MM_forces_weight], device=device
    )
    datasets = [
        BMSIndexedDataset(
            args.train_file,
            model.atomic_numbers.tolist(),
            float(model.r_max),
            args.split_manifest,
            split,
        )
        for split in ["train", "valid"]
    ]
    samplers = [
        BMSDistributedBatchSampler(
            len(dataset), args.batch_size, rank, world_size, args.seed, shuffle=(i == 0)
        )
        for i, dataset in enumerate(datasets)
    ]
    # Share the allocation's RAM with the 110 GB node-local HDF5 cache. Keep
    # only training workers persistent; validation reads in the rank process.
    loaders = [
        torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers if index == 0 else 0,
            collate_fn=partial(
                collate_bms_microbatches,
                batch_size=args.batch_size,
                max_edges=args.max_edges,
                max_pairs=args.max_pairs,
            ),
            **(
                {
                    "multiprocessing_context": "spawn",
                    "persistent_workers": True,
                    "prefetch_factor": 1,
                }
                if args.num_workers and index == 0
                else {}
            ),
        )
        for index, (dataset, sampler) in enumerate(zip(datasets, samplers))
    ]
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    deadline = int(os.environ.get("MACE_TRAIN_STOP_AT", "0"))
    step, epoch, seen, batches_completed = 0, 0, 0, 0
    best_validation = float("inf")
    resume = work / "checkpoints/latest.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        for key in [
            "split_manifest_sha256",
            "initial_model_sha256",
            "world_size",
            "energy_weight",
            "forces_weight",
            "MM_forces_weight",
        ]:
            if state["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, seen, batches_completed = (
            state[k] for k in ["step", "epoch", "seen", "batches_completed"]
        )
        best_validation = state.get("best_validation", best_validation)
        rng = state["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state(rng["cuda"].cpu())
    elif rank == 0:
        print(
            f"Loaded initial model {args.initial_model}; fresh optimizer step=0",
            flush=True,
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "dtype": "float32",
                    "recompute_field_after_scf": model.recompute_field_after_scf,
                    "fukui_regularisation": model.fukui_regularisation,
                    "cueq": args.enable_cueq,
                    "conv_fusion": args.cueq_conv_fusion,
                    "per_rank_batch": args.batch_size,
                    "global_batch": args.batch_size * world_size,
                    "split_counts": manifest["counts"],
                    "initial_step": step,
                }
            ),
            flush=True,
        )

    def save():
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        }
        states = [None] * world_size
        dist.all_gather_object(states, rng)
        if rank == 0:
            temporary = resume.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "seen": seen,
                    "batches_completed": batches_completed,
                    "best_validation": best_validation,
                    "rng": states,
                    "args": vars(args),
                },
                temporary,
            )
            temporary.replace(resume)
        dist.barrier()

    def evaluate(full=False):
        optimizer.eval()
        model.eval()
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        for microbatches in loaders[1]:
            for batch in microbatches:
                ref = batch.to(device)
                pred = model(ref.to_dict())
                totals[:3] += component_sums(ref, pred).detach().double()
                totals[3:] += totals.new_tensor(
                    [len(ref.ptr) - 1, ref.forces.numel(), ref.MM_forces.numel()]
                )
                del pred, ref
            if not full:
                break
        dist.all_reduce(totals)
        means = totals[:3] / totals[3:].clamp_min(1)
        result = {
            "validation_step": step,
            "full_validation": full,
            "validation_frames": int(totals[3]),
            "energy_mae_per_atom_eV": float(means[0]),
            "forces_rmse_eV_A": float(means[1].sqrt()),
            "MM_forces_rmse_eV_A": float(means[2].sqrt()),
            "loss": float((means * weights.double()).sum()),
        }
        if rank == 0:
            print(json.dumps(result), flush=True)
            with (work / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        optimizer.train()
        model.train()
        return result["loss"]

    model.train()
    optimizer.train()
    completed = False
    for epoch in range(epoch, args.max_num_epochs):
        samplers[0].epoch = epoch
        samplers[0].skip_batches = batches_completed
        for microbatches in loaders[0]:
            should_stop = (
                stop[0]
                or (deadline and time.time() >= deadline)
                or (args.max_steps and step >= args.max_steps)
            )
            flag = torch.tensor(int(bool(should_stop)), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                completed = True
                break
            started = time.perf_counter()
            counts = torch.tensor(
                [
                    sum(len(batch.ptr) - 1 for batch in microbatches),
                    sum(batch.forces.numel() for batch in microbatches),
                    sum(batch.MM_forces.numel() for batch in microbatches),
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(counts)
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, dtype=torch.float64, device=device)
            torch.cuda.reset_peak_memory_stats(device)
            for index, batch in enumerate(microbatches):
                context = (
                    distributed_model.no_sync()
                    if index + 1 < len(microbatches)
                    else nullcontext()
                )
                with context:
                    ref = batch.to(device)
                    pred = distributed_model(ref.to_dict(), training=True)
                    sums = component_sums(ref, pred)
                    loss = global_component_loss(
                        sums, counts.to(sums.dtype), weights, world_size
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Nonfinite loss rank={rank}: {ref.sample_id}"
                        )
                    loss.backward()
                    totals += sums.detach().double()
                    del ref, pred, sums, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradients rank={rank}")
            optimizer.step()
            dist.all_reduce(totals)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            step += 1
            seen += int(counts[0])
            batches_completed += 1
            local = torch.tensor(
                [
                    elapsed,
                    torch.cuda.max_memory_allocated(device) / 1024**3,
                    sum(len(batch.ptr) - 1 for batch in microbatches),
                    len(microbatches),
                ],
                device=device,
                dtype=torch.float64,
            )
            details = [torch.zeros_like(local) for _ in range(world_size)]
            dist.all_gather(details, local)
            if step == 1:
                ids = [None] * world_size
                dist.all_gather_object(
                    ids, [name for batch in microbatches for name in batch.sample_id]
                )
                flat = [item for group in ids for item in group]
                assert len(flat) == len(
                    set(flat)
                ), "DDP ranks duplicated training records"
                digest = hashlib.sha256()
                for parameter in model.parameters():
                    digest.update(parameter.detach().cpu().numpy().tobytes())
                hashes = [None] * world_size
                dist.all_gather_object(hashes, digest.hexdigest())
                assert len(set(hashes)) == 1, "DDP parameters differ across ranks"
                if rank == 0:
                    (work / "ddp_verified.json").write_text(
                        json.dumps(
                            {
                                "world_size": world_size,
                                "first_step_ids": ids,
                                "unique_frames": len(flat),
                                "parameter_sha256_per_rank": hashes,
                                "synchronized": True,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
            if rank == 0:
                means = totals / counts
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "seen": seen,
                    "loss": float((means * weights.double()).sum()),
                    "energy_mae_per_atom_eV": float(means[0]),
                    "forces_rmse_eV_A": float(means[1].sqrt()),
                    "MM_forces_rmse_eV_A": float(means[2].sqrt()),
                    "batch_size": int(counts[0]),
                    "step_seconds": max(float(x[0]) for x in details),
                    "peak_memory_GiB": max(float(x[1]) for x in details),
                    "rank_batch_sizes": [int(x[2]) for x in details],
                    "rank_microbatches": [int(x[3]) for x in details],
                }
                print(json.dumps(entry), flush=True)
                with (work / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
            del microbatches
            if step % args.save_interval == 0:
                save()
                evaluate(full=False)
        if completed:
            break
        validation_loss = evaluate(full=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            optimizer.eval()
            model.eval()
            if rank == 0:
                torch.save(deepcopy(model).cpu(), work / "best.model")
            optimizer.train()
            model.train()
        batches_completed = 0
        epoch += 1
        save()
    save()
    optimizer.eval()
    model.eval()
    if rank == 0:
        torch.save(deepcopy(model).cpu(), work / "mace_polar_bms25.model")
        print(f"Saved step={step}, epoch={epoch}, seen={seen}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
