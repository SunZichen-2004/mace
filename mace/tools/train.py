###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if os.environ.get("POLAR2_SAMPLING_CONTROL"):
        from .polar2_sampling import update_sampling
        update_sampling(eval_metrics, epoch, valid_loader_name)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = "    None" if eval_metrics.get("rmse_e_per_atom") is None else format(eval_metrics["rmse_e_per_atom"] * 1e3, "8.2f")
        error_f = "    None" if eval_metrics.get("rmse_f") is None else format(eval_metrics["rmse_f"] * 1e3, "8.2f")
        msg = (
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}"
        )
        for key, label in (
            ("loss_energy", "L_E"),
            ("loss_forces", "L_F"),
            ("loss_MM_forces", "L_F_MM"),
            ("loss_dipole", "L_mu"),
            ("loss_quadrupole", "L_Q"),
            ("loss_spin_dipole", "L_spin_mu"),
            ("loss_spin_quadrupole", "L_spin_Q"),
            ("loss_polarizability", "L_alpha"),
            ("loss_interaction_energy", "L_E_int"),
            ("loss_interaction_forces", "L_F_int"),
            ("loss_fragment_charge", "L_frag_Q"),
            ("loss_fragment_spin", "L_frag_S"),
            ("loss_gaussian_charge", "L_Gq"),
            ("loss_gaussian_dipole", "L_Gmu"),
            ("loss_gaussian_spin", "L_Gspin"),
            ("loss_gaussian_spin_dipole", "L_Gspin_mu"),
        ):
            if key in eval_metrics:
                msg += f", {label}={eval_metrics[key]:.6f}"
        msg += (
            f", RMSE_E_per_atom={error_e} meV, RMSE_F={error_f} meV / A"
        )
        if "mae_e_per_atom" in eval_metrics:
            msg += f", MAE_E_per_atom={eval_metrics['mae_e_per_atom'] * 1e3:8.2f} meV"
        if "mae_f" in eval_metrics:
            msg += f", MAE_F={eval_metrics['mae_f'] * 1e3:8.2f} meV / A"
        for component, label in (("charge", "Gq"), ("dipole", "Gmu"),
                                 ("spin", "Gspin"), ("spin_dipole", "Gspin_mu")):
            for statistic in ("mae", "rmse"):
                key = statistic + "_gaussian_" + component
                if key in eval_metrics:
                    msg += f", {statistic.upper()}_{label}={eval_metrics[key]:.6f}"
        if "rmse_mu" in eval_metrics:
            msg += f", RMSE_mu={eval_metrics['rmse_mu']:.6f} e·Å"
        if "mae_mu" in eval_metrics:
            msg += f", MAE_mu={eval_metrics['mae_mu']:.6f} e·Å"
        if "rmse_mu_field" in eval_metrics:
            msg += f", RMSE_mu_field={eval_metrics['rmse_mu_field']:.6f} e·Å"
        if "mae_mu_field" in eval_metrics:
            msg += f", MAE_mu_field={eval_metrics['mae_mu_field']:.6f} e·Å"
        if "rmse_polarizability" in eval_metrics:
            msg += (
                f", RMSE_alpha={eval_metrics['rmse_polarizability']:.6f} e·Å²/V"
            )
        if "mae_polarizability" in eval_metrics:
            msg += (
                f", MAE_alpha={eval_metrics['mae_polarizability']:.6f} e·Å²/V"
            )
        # POLAR-2 reports both physical alpha responses in A^3. SPICE-alpha
        # shares the omol head; these metrics include only alpha-labelled rows.
        for quantity, label, unit in (
            ("alpha_field_A3", "alpha_field[SPICE-alpha]", "Å³"),
            ("alpha_multipole_A3", "alpha_multipole[SPICE-alpha]", "Å³"),
            ("MM_forces", "F_MM", "eV/Å"),
        ):
            for statistic in ("mae", "rmse"):
                key = f"{statistic}_{quantity}"
                if key in eval_metrics:
                    msg += f", {statistic.upper()}_{label}={eval_metrics[key]:.6f} {unit}"
        if "rmse_quadrupole" in eval_metrics:
            msg += f", RMSE_Q={eval_metrics['rmse_quadrupole']:.6f} e·Å² traceless"
        if "mae_quadrupole" in eval_metrics:
            msg += f", MAE_Q={eval_metrics['mae_quadrupole']:.6f} e·Å² traceless"
        if "rmse_spin_dipole" in eval_metrics:
            msg += f", RMSE_spin_mu={eval_metrics['rmse_spin_dipole']:.6f} e·Å"
        if "mae_spin_dipole" in eval_metrics:
            msg += f", MAE_spin_mu={eval_metrics['mae_spin_dipole']:.6f} e·Å"
        if "rmse_spin_quadrupole" in eval_metrics:
            msg += (
                f", RMSE_spin_Q={eval_metrics['rmse_spin_quadrupole']:.6f}"
                " e·Å² traceless"
            )
        if "mae_spin_quadrupole" in eval_metrics:
            msg += (
                f", MAE_spin_Q={eval_metrics['mae_spin_quadrupole']:.6f}"
                " e·Å² traceless"
            )
        if "rmse_interaction_energy" in eval_metrics:
            msg += (
                f", RMSE_E_int={eval_metrics['rmse_interaction_energy'] * 1e3:.6f} meV"
                f", RMSE_E_int_per_atom={eval_metrics['rmse_interaction_energy_per_atom'] * 1e3:.6f} meV"
            )
        if "rmse_fragment_charge" in eval_metrics:
            msg += (
                f", RMSE_frag_Q={eval_metrics['rmse_fragment_charge']:.6f} e"
                f", RMSE_frag_S={eval_metrics['rmse_fragment_spin']:.6f}"
            )
        if "rmse_interaction_forces" in eval_metrics:
            msg += f", RMSE_F_int={eval_metrics['rmse_interaction_forces'] * 1e3:.6f} meV / A"
        logging.info(msg)
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = (eval_metrics["rmse_e_per_atom"] if eval_metrics.get("rmse_e_per_atom") is not None else float("nan")) * 1e3
        error_f = (eval_metrics["rmse_f"] if eval_metrics.get("rmse_f") is not None else float("nan")) * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        errors_magforces = None
        if "rmse_magf" in eval_metrics.keys():
            errors_magforces = eval_metrics["rmse_magf"] * 1e3
        msg = f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3"
        msg += (
            f", RMSE_magforces={errors_magforces:8.2f} meV / μB"
            if errors_magforces is not None
            else ""
        )
        logging.info(msg)
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = (eval_metrics["rmse_e_per_atom"] if eval_metrics.get("rmse_e_per_atom") is not None else float("nan")) * 1e3
        error_f = (eval_metrics["rmse_f"] if eval_metrics.get("rmse_f") is not None else float("nan")) * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = (eval_metrics["rmse_f"] if eval_metrics.get("rmse_f") is not None else float("nan")) * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = (eval_metrics["rmse_e_per_atom"] if eval_metrics.get("rmse_e_per_atom") is not None else float("nan")) * 1e3
        error_f = (eval_metrics["rmse_f"] if eval_metrics.get("rmse_f") is not None else float("nan")) * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    data_aug_magmom: Optional[bool] = False,
    data_aug_magmom_mode: str = "non-soc",
):
    lowest_loss = np.inf
    full_best_path = os.environ.get("POLAR2_FULL_BEST_STATE")
    if full_best_path and os.path.exists(full_best_path):
        import json
        with open(full_best_path) as stream:
            lowest_loss = json.load(stream)["loss"]
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # The stock loop evaluates the whole validation set before epoch 0.
    # MACE_SKIP_INITIAL_EVAL=1 starts the epoch immediately; end-of-epoch
    # evaluation is unchanged.
    if os.environ.get("MACE_SKIP_INITIAL_EVAL", "") in ("1", "true", "True"):
        logging.info("Skipping the validation pass that runs before epoch 0")
    else:
        comparable_monitor = bool(os.environ.get("POLAR2_SAMPLING_CONTROL"))
        if comparable_monitor and "ScheduleFree" in type(optimizer).__name__:
            optimizer.eval()
        initial_context = ema.average_parameters() if comparable_monitor and ema is not None else nullcontext()
        with initial_context:
            polar2_valid_losses = []
            for valid_loader_name, valid_loader in valid_loaders.items():
                if os.environ.get('POLAR2_INITIAL_MONITOR_ONLY') == '1' and not valid_loader_name.endswith(('/spice_alpha', '/omol4m')):
                    continue
                valid_loss_head, eval_metrics = evaluate(
                    model=model,
                    loss_fn=loss_fn,
                    data_loader=valid_loader,
                    output_args=output_args,
                    device=device,
                )
                polar2_valid_losses.append(valid_loss_head)
                if rank == 0:
                    valid_err_log(
                        valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
                    )
            valid_loss = (sum(polar2_valid_losses) / len(polar2_valid_losses)
                          if getattr(loss_fn, "is_polar2", False) else valid_loss_head)
        if comparable_monitor and "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None

    if data_aug_magmom:
        # pylint: disable=cyclic-import
        from mace.data.augmentation import create_random_rotation_loader

        train_loader = create_random_rotation_loader(
            train_loader, mode=data_aug_magmom_mode
        )

    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()

        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {}
                polar2_valid_losses = []
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if not (full_best_path and valid_loader_name.endswith("/omol4m")):
                        polar2_valid_losses.append(valid_loss_head)
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_log_dict[valid_loader_name] = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    sum(polar2_valid_losses) / len(polar2_valid_losses)
                    if getattr(loss_fn, "is_polar2", False) else valid_loss_head
                )
            if log_wandb:
                wandb.log(wandb_log_dict)
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
                        if full_best_path:
                            import json
                            from pathlib import Path
                            best_file = Path(full_best_path)
                            temporary = best_file.with_suffix(".tmp")
                            temporary.write_text(json.dumps(dict(epoch=epoch, loss=lowest_loss,
                                criterion="mean full-source POLAR2 loss; omol4m diagnostic excluded",
                                ema=ema is not None), indent=2))
                            temporary.replace(best_file)
        callback = globals().get("polar2_after_validation")
        if callback is not None:
            callback(epoch)
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        previous_step_end = time.perf_counter()
        for batch in data_loader:
            batch_ready = time.perf_counter()
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            if getattr(loss_fn, "is_polar2", False):
                opt_metrics["data_wait_seconds"] = batch_ready - previous_step_end
                opt_metrics["iteration_seconds"] = time.perf_counter() - previous_step_end
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)
            previous_step_end = time.perf_counter()


def _mae_rmse(delta: torch.Tensor) -> Tuple[float, float]:
    values = to_numpy(delta.detach().double().reshape(-1))
    return compute_mae(values), compute_rmse(values)


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    captured = {}
    polar2_timing = getattr(loss_fn, "is_polar2", False) and device.type == "cuda"
    timing_events = {}
    def mark_phase(name):
        if polar2_timing:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            timing_events[name] = event
    mark_phase("start")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        kwargs = dict(
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        if output_args.get("magforces", False):
            kwargs["compute_magforces"] = True
        if hasattr(loss_fn, "forward_kwargs"):
            kwargs = loss_fn.forward_kwargs(batch, kwargs)
        output = model(batch_dict, **kwargs)
        captured["output"] = output
        mark_phase("forward")
        if getattr(loss_fn, "is_polar2", False):
            captured["parts"] = loss_fn.weighted_components(ref=batch, pred=output)
            loss = sum(captured["parts"].values())
        else:
            loss = loss_fn(pred=output, ref=batch)
        is_polar2 = getattr(loss_fn, "is_polar2", False)
        scale = 1.
        if is_polar2:
            from .polar2_gradients import backward_scale, clip_scaled_gradients
            scale = backward_scale(loss)
        mark_phase("loss_reduce")
        (loss * scale).backward()
        mark_phase("backward")
        if is_polar2:
            captured["backward_scale"] = scale
            captured["gradient_norm_before_clip"] = clip_scaled_gradients(
                model.parameters(), scale, max_grad_norm)
        elif max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    try:
        loss = closure()
    except FloatingPointError as error:
        if getattr(loss_fn, "is_polar2", False):
            from .polar2_diagnostics import save_failure
            save_failure(model, batch, captured, error)
        raise
    optimizer.step()

    if ema is not None:
        ema.update()
    mark_phase("update")

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }
    for key in ("backward_scale", "gradient_norm_before_clip"):
        if key in captured:
            loss_dict[key] = captured[key]
    output = captured.get("output")
    if output is not None:
        num_atoms = batch.ptr[1:] - batch.ptr[:-1]
        if output.get("energy") is not None and batch.energy is not None:
            delta_e = (batch.energy - output["energy"]).detach()
            loss_dict["mae_e"], loss_dict["rmse_e"] = _mae_rmse(delta_e)
            loss_dict["mae_e_per_atom"], loss_dict["rmse_e_per_atom"] = _mae_rmse(
                delta_e / num_atoms
            )
        if output.get("forces") is not None and batch.forces is not None:
            loss_dict["mae_f"], loss_dict["rmse_f"] = _mae_rmse(
                batch.forces - output["forces"]
            )
        if output.get("dipole") is not None and getattr(batch, "dipole", None) is not None:
            loss_dict["mae_mu"], loss_dict["rmse_mu"] = _mae_rmse(
                batch.dipole - output["dipole"]
            )
        field_mu = output.get("total_dipole_from_electric_field") if output else None
        if field_mu is not None and getattr(batch, "dipole", None) is not None:
            loss_dict["mae_mu_field"], loss_dict["rmse_mu_field"] = _mae_rmse(
                batch.dipole - field_mu
            )
        if (
            output.get("quadrupole") is not None
            and getattr(batch, "quadrupole", None) is not None
        ):
            from mace.modules.loss import _QUADRUPOLE_COMPONENTS
            from mace.modules.utils import traceless_quadrupole

            n_graphs = int(batch.ptr.numel() - 1)
            pred_q = output["quadrupole"]
            ref_q = traceless_quadrupole(batch.quadrupole.view(n_graphs, 3, 3))
            delta_q = torch.stack(
                [ref_q[:, i, j] - pred_q[:, i, j] for i, j in _QUADRUPOLE_COMPONENTS],
                dim=-1,
            )
            loss_dict["mae_quadrupole"], loss_dict["rmse_quadrupole"] = _mae_rmse(
                delta_q
            )

    if output is not None and getattr(loss_fn, "is_polar2", False):
        with torch.no_grad():
            parts = captured["parts"]
            values = torch.stack([v.detach() for v in parts.values()])
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(values)
                values /= torch.distributed.get_world_size()
            loss_dict.update({"loss_" + k: float(v) for k, v in zip(parts, values)})
            loss_dict["loss"] = float(values.sum())
            loss_dict["error_metrics_scope"] = "rank_local; loss components globally reduced"

            loss_dict["batch_size"] = batch.num_graphs
            loss_dict["source"] = ("spice_alpha" if (batch.polarizability_weight > 0).any()
                                   else "bms25" if (batch.MM_forces_weight > 0).any() else "omol4m")
            if getattr(batch, "source_name", None):
                loss_dict["source"] = batch.source_name[0]
            for target, prediction, metric in [
                ("MM_forces", "MM_forces", "MM_forces"),
                ("polarizability", "polar2_alpha_field", "alpha_field_A3"),
                ("polarizability", "polar2_alpha_multipole", "alpha_multipole_A3"),
                ("spin_dipole", "spin_dipole", "spin_dipole"),
                ("spin_quadrupole", "spin_quadrupole", "spin_quadrupole")]:
                estimate = output.get(prediction)
                mask = getattr(batch, target + "_weight") > 0
                if estimate is None or not mask.any():
                    continue
                if target == "MM_forces":mask = mask[batch.MM_batch]
                while mask.ndim < estimate.ndim:mask = mask.unsqueeze(-1)
                mask = mask.expand_as(estimate)
                if target == "polarizability":estimate = estimate * 14.3996454784255
                residual = (getattr(batch, target).reshape_as(estimate) - estimate)[mask]
                loss_dict["mae_" + metric], loss_dict["rmse_" + metric] = _mae_rmse(residual)
            # Do not report zero-filled, unsupervised labels as measured errors.
            if loss_dict["source"] == "spice_alpha":
                for key in list(loss_dict):
                    if (key.startswith(("mae_", "rmse_")) and "alpha_" not in key):del loss_dict[key]
            if not (batch.quadrupole_weight > 0).any():
                loss_dict.pop("mae_quadrupole", None)
                loss_dict.pop("rmse_quadrupole", None)
            if device.type == "cuda":
                loss_dict["peak_memory_GiB"] = torch.cuda.max_memory_allocated(device) / 2**30

    if output is not None and hasattr(loss_fn, "gaussian_residuals"):
        from mace.modules.utils import traceless_quadrupole
        from mace.modules.loss import _QUADRUPOLE_COMPONENTS
        with torch.no_grad():
            for component, delta in loss_fn.gaussian_residuals(batch, output).items():
                loss_dict["mae_gaussian_" + component], loss_dict["rmse_gaussian_" + component] = _mae_rmse(delta)
            if output.get("spin_dipole") is not None and getattr(batch, "spin_dipole", None) is not None:
                loss_dict["mae_spin_dipole"], loss_dict["rmse_spin_dipole"] = _mae_rmse(batch.spin_dipole - output["spin_dipole"])
            if output.get("spin_quadrupole") is not None and getattr(batch, "spin_quadrupole", None) is not None:
                n_graphs = int(batch.ptr.numel() - 1)
                ref_sq = traceless_quadrupole(batch.spin_quadrupole.view(n_graphs, 3, 3))
                delta_sq = torch.stack([ref_sq[:, i, j] - output["spin_quadrupole"][:, i, j]
                                        for i, j in _QUADRUPOLE_COMPONENTS], dim=-1)
                loss_dict["mae_spin_quadrupole"], loss_dict["rmse_spin_quadrupole"] = _mae_rmse(delta_sq)

    if getattr(batch, "interaction_targets", None) is not None:
        from mace.modules.loss import interaction_energy_error

        delta = interaction_energy_error(batch, output).detach()
        loss_dict["n_interactions"] = delta.numel()
        if delta.numel():
            loss_dict["mae_interaction_energy"], loss_dict["rmse_interaction_energy"] = _mae_rmse(delta)
            loss_dict["mae_interaction_energy_per_atom"], loss_dict["rmse_interaction_energy_per_atom"] = _mae_rmse(
                delta / batch.interaction_num_atoms)
            loss_dict["loss_interaction_energy"] = float(
                loss_fn.interaction_energy_weight * (
                    batch.interaction_weights * (delta / batch.interaction_num_atoms).square()).mean())
    if getattr(batch, "fragment_charge_targets", None) is not None:
        from mace.modules.loss import fragment_charge_spin_error

        charge_error, spin_error = (error.detach() for error in fragment_charge_spin_error(batch, output))
        loss_dict["n_fragments"] = charge_error.numel()
        if charge_error.numel():
            for name, error, weight in (
                ("fragment_charge", charge_error, loss_fn.fragment_charge_weight),
                ("fragment_spin", spin_error, loss_fn.fragment_spin_weight)):
                loss_dict["mae_" + name], loss_dict["rmse_" + name] = _mae_rmse(error)
                loss_dict["loss_" + name] = float(weight * (batch.fragment_weights * error.square()).mean())
    if getattr(loss_fn, "interaction_force_weight", 0.0) != 0:
        from mace.modules.loss import interaction_force_error

        delta_force = interaction_force_error(batch, output).detach()
        if delta_force.numel():
            loss_dict["mae_interaction_forces"], loss_dict["rmse_interaction_forces"] = _mae_rmse(delta_force)
            loss_dict["loss_interaction_forces"] = float(loss_fn.interaction_force_weight * (
                batch.interaction_force_weights[:, None] * delta_force.square()).mean())
    if polar2_timing:
        mark_phase("metrics")
        timing_events["metrics"].synchronize()
        previous = "start"
        for phase in ("forward", "loss_reduce", "backward", "update", "metrics"):
            loss_dict[phase + "_gpu_seconds"] = timing_events[previous].elapsed_time(timing_events[phase]) / 1000.
            previous = phase
        loss_dict["time"] = time.time() - start_time
        loss_dict["atoms"] = batch.positions.shape[0]
        loss_dict["edges"] = batch.edge_index.shape[1]
        loss_dict["MM_atoms"] = batch.MM_coordinate.shape[0]
    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        kwargs = dict(
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        if output_args.get("magforces", False):
            kwargs["compute_magforces"] = True
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            if hasattr(loss_fn, "forward_kwargs"):
                kwargs = loss_fn.forward_kwargs(batch, kwargs)
            output = model(batch_dict, **kwargs)
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


# Keep parameters frozen/active after evaluation
@contextmanager
def preserve_grad_state(model):
    # save the original requires_grad state for all parameters
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:

    polar2 = getattr(loss_fn, "is_polar2", False)
    if polar2:
        from mace.tools.polar2_metrics import Polar2Metrics
        metrics = Polar2Metrics(loss_fn, device)
        # Uneven rank tails are valid: no forward-time DDP collectives.
        if isinstance(model, DistributedDataParallel):
            model = model.module
    else:
        metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()

    aborted = False
    with preserve_grad_state(model):
        for batch in data_loader:
            stop_check = globals().get('polar2_eval_should_stop')
            if polar2 and stop_check is not None and stop_check():
                aborted = True
                break
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            kwargs = dict(
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            if output_args.get("magforces", False):
                kwargs["compute_magforces"] = True
            if hasattr(loss_fn, "forward_kwargs"):
                kwargs = loss_fn.forward_kwargs(batch, kwargs)
            output = model(batch_dict, **kwargs)
            metrics.update(batch, output)
    if polar2:
        flag = torch.tensor(int(aborted), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        if flag.item():
            globals()['polar2_eval_abort']()
    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    if not polar2:
        metrics.reset()

    return avg_loss, aux


class MACELoss(Metric):
    _LOSS_PART_NAMES = (
        "energy",
        "forces",
        "dipole",
        "quadrupole",
        "spin_dipole",
        "spin_quadrupole",
        "polarizability",
        "MM_forces",
        "interaction_energy",
        "interaction_forces",
        "fragment_charge",
        "fragment_spin",
        "gaussian_charge",
        "gaussian_dipole",
        "gaussian_spin",
        "gaussian_spin_dipole",
    )

    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        for quantity in ("alpha_field_A3", "alpha_multipole_A3", "MM_forces"):
            for suffix in ("count", "abs", "sq"):
                self.add_state("polar2_" + quantity + "_" + suffix,
                               default=torch.tensor(0.0), dist_reduce_fx="sum")
        for component in ("charge", "dipole", "spin", "spin_dipole"):
            for suffix in ("count", "loss_count", "abs", "sq", "weighted_sq"):
                self.add_state("gaussian_" + component + "_" + suffix,
                               default=torch.tensor(0.0), dist_reduce_fx="sum")
        for name in ("interaction_force_count", "interaction_force_abs",
                     "interaction_force_sq", "interaction_force_weighted_sq"):
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        for name in ("fragment_count", "fragment_charge_abs", "fragment_charge_sq",
                     "fragment_spin_abs", "fragment_spin_sq",
                     "fragment_charge_weighted_sq", "fragment_spin_weighted_sq"):
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        for name in ("interaction_count", "interaction_abs", "interaction_sq",
                     "interaction_abs_per_atom", "interaction_sq_per_atom",
                     "interaction_weighted_sq_per_atom"):
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_loss", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "Mus_field_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_mus_field", default=[], dist_reduce_fx="cat")
        self.add_state("Qs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_qs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "SpinMus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_spin_mus", default=[], dist_reduce_fx="cat")
        self.add_state(
            "SpinQs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_spin_qs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

        self.add_state(
            "MagFs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("MagFs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_MagFs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "loss_part_sums",
            default=torch.zeros(len(self._LOSS_PART_NAMES), dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "loss_part_seen",
            default=torch.zeros(len(self._LOSS_PART_NAMES), dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        interaction_mode = getattr(batch, "interaction_targets", None) is not None
        fragment_mode = getattr(batch, "fragment_charge_targets", None) is not None
        sample_count = len(batch.interaction_base_graphs) if interaction_mode else (batch.num_graphs if fragment_mode else 1)
        if hasattr(self.loss_fn, "gaussian_residuals") or getattr(self.loss_fn, "is_polar2", False):
            sample_count = batch.num_graphs
        if hasattr(self.loss_fn, "weighted_components"):
            # Metric states perform their own distributed reduction.
            kwargs = {"ddp": False} if interaction_mode or fragment_mode or hasattr(self.loss_fn, "gaussian_residuals") or getattr(self.loss_fn, "is_polar2", False) else {}
            parts = self.loss_fn.weighted_components(ref=batch, pred=output, **kwargs)
            loss = torch.zeros(
                (), dtype=torch.get_default_dtype(), device=batch.energy.device
            )
            for name, value in parts.items():
                loss = loss + value
                index = self._LOSS_PART_NAMES.index(name)
                self.loss_part_sums[index] += value.detach() * sample_count
                self.loss_part_seen[index] += 1
        else:
            loss = self.loss_fn(pred=output, ref=batch)
        if hasattr(self.loss_fn, "gaussian_residuals"):
            with torch.no_grad():
                residuals = self.loss_fn.gaussian_residuals(batch, output)
                atom_weights = batch.weight[batch.batch]
                active = atom_weights != 0
                for component, residual in residuals.items():
                    delta = residual.detach()[active]
                    prefix = "gaussian_" + component + "_"
                    weights = atom_weights[active].view((-1,) + (1,) * (delta.ndim - 1))
                    getattr(self, prefix + "count").add_(delta.numel())
                    getattr(self, prefix + "loss_count").add_(residual.numel())
                    getattr(self, prefix + "abs").add_(delta.abs().sum())
                    getattr(self, prefix + "sq").add_(delta.square().sum())
                    getattr(self, prefix + "weighted_sq").add_((weights * delta.square()).sum())
        if getattr(self.loss_fn, "is_polar2", False):
            for target, prediction, quantity in [
                ("polarizability", "polar2_alpha_field", "alpha_field_A3"),
                ("polarizability", "polar2_alpha_multipole", "alpha_multipole_A3"),
                ("MM_forces", "MM_forces", "MM_forces")]:
                estimate = output.get(prediction)
                if estimate is None:continue
                weights = getattr(batch, target + "_weight")
                if target == "MM_forces":weights = weights[batch.MM_batch]
                while weights.ndim < estimate.ndim:weights = weights.unsqueeze(-1)
                active = weights.expand_as(estimate) != 0
                factor = 14.3996454784255 if target == "polarizability" else 1.
                delta = (estimate.detach() * factor - getattr(batch, target))[active]
                getattr(self, "polar2_" + quantity + "_count").add_(delta.numel())
                getattr(self, "polar2_" + quantity + "_abs").add_(delta.abs().sum())
                getattr(self, "polar2_" + quantity + "_sq").add_(delta.square().sum())
        self.total_loss += loss * sample_count
        self.num_data += sample_count if interaction_mode or fragment_mode else batch.num_graphs
        if interaction_mode:
            from mace.modules.loss import interaction_energy_error

            delta = interaction_energy_error(batch, output).detach()
            per_atom = delta / batch.interaction_num_atoms
            self.interaction_count += delta.numel()
            self.interaction_abs += delta.abs().sum()
            self.interaction_sq += delta.square().sum()
            self.interaction_abs_per_atom += per_atom.abs().sum()
            self.interaction_sq_per_atom += per_atom.square().sum()
            self.interaction_weighted_sq_per_atom += (batch.interaction_weights * per_atom.square()).sum()

        if getattr(self.loss_fn, "interaction_force_weight", 0.0) != 0:
            from mace.modules.loss import interaction_force_error

            delta_force = interaction_force_error(batch, output).detach()
            self.interaction_force_count += delta_force.numel()
            self.interaction_force_abs += delta_force.abs().sum()
            self.interaction_force_sq += delta_force.square().sum()
            self.interaction_force_weighted_sq += (batch.interaction_force_weights[:, None] * delta_force.square()).sum()

        if fragment_mode:
            from mace.modules.loss import fragment_charge_spin_error

            charge_error, spin_error = fragment_charge_spin_error(batch, output)
            self.fragment_count += charge_error.numel()
            for name, error in (("fragment_charge", charge_error), ("fragment_spin", spin_error)):
                error = error.detach()
                getattr(self, name + "_abs").add_(error.abs().sum())
                getattr(self, name + "_sq").add_(error.square().sum())
                getattr(self, name + "_weighted_sq").add_((batch.fragment_weights * error.square()).sum())

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
            filter_nonzero_weight(
                batch, self.delta_es_per_atom, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )

        if output.get("magforces") is not None and batch.magforces is not None:
            self.MagFs.append(batch.magforces)
            self.delta_MagFs.append(batch.magforces - output["magforces"])
            self.MagFs_computed += filter_nonzero_weight(
                batch,
                self.delta_MagFs,
                batch.weight,
                batch.magforces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
            filter_nonzero_weight(
                batch,
                self.delta_mus_per_atom,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        field_mu = output.get("total_dipole_from_electric_field")
        if field_mu is not None and batch.dipole is not None:
            self.delta_mus_field.append(batch.dipole - field_mu)
            self.Mus_field_computed += filter_nonzero_weight(
                batch,
                self.delta_mus_field,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("quadrupole") is not None
            and getattr(batch, "quadrupole", None) is not None
        ):
            from mace.modules.loss import _QUADRUPOLE_COMPONENTS
            from mace.modules.utils import traceless_quadrupole

            n_graphs = int(batch.ptr.numel() - 1)
            pred_q = output["quadrupole"]
            ref_q = traceless_quadrupole(batch.quadrupole.view(n_graphs, 3, 3))
            delta_q = torch.stack(
                [ref_q[:, i, j] - pred_q[:, i, j] for i, j in _QUADRUPOLE_COMPONENTS],
                dim=-1,
            )
            self.delta_qs.append(delta_q)
            self.Qs_computed += filter_nonzero_weight(
                batch,
                self.delta_qs,
                batch.weight,
                getattr(batch, "quadrupole_weight", torch.ones_like(batch.weight)),
            )
        if (
            output.get("spin_dipole") is not None
            and getattr(batch, "spin_dipole", None) is not None
        ):
            self.delta_spin_mus.append(batch.spin_dipole - output["spin_dipole"])
            self.SpinMus_computed += filter_nonzero_weight(
                batch,
                self.delta_spin_mus,
                batch.weight,
                getattr(batch, "spin_dipole_weight", torch.ones_like(batch.weight)),
            )
        if (
            output.get("spin_quadrupole") is not None
            and getattr(batch, "spin_quadrupole", None) is not None
        ):
            from mace.modules.loss import _QUADRUPOLE_COMPONENTS
            from mace.modules.utils import traceless_quadrupole

            n_graphs = int(batch.ptr.numel() - 1)
            pred_sq = output["spin_quadrupole"]
            ref_sq = traceless_quadrupole(batch.spin_quadrupole.view(n_graphs, 3, 3))
            delta_sq = torch.stack(
                [ref_sq[:, i, j] - pred_sq[:, i, j] for i, j in _QUADRUPOLE_COMPONENTS],
                dim=-1,
            )
            self.delta_spin_qs.append(delta_sq)
            self.SpinQs_computed += filter_nonzero_weight(
                batch,
                self.delta_spin_qs,
                batch.weight,
                getattr(batch, "spin_quadrupole_weight", torch.ones_like(batch.weight)),
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        for index, name in enumerate(self._LOSS_PART_NAMES):
            if float(self.loss_part_seen[index]) > 0:
                aux[f"loss_{name}"] = to_numpy(
                    self.loss_part_sums[index] / self.num_data
                ).item()
        if hasattr(self.loss_fn, "gaussian_residuals"):
            for component in ("charge", "dipole", "spin", "spin_dipole"):
                prefix = "gaussian_" + component + "_"
                count = getattr(self, prefix + "count")
                if count > 0:
                    aux["mae_gaussian_" + component] = to_numpy(getattr(self, prefix + "abs") / count).item()
                    aux["rmse_gaussian_" + component] = to_numpy((getattr(self, prefix + "sq") / count).sqrt()).item()
                    exact = to_numpy(self.loss_fn.gaussian_weights[component] * getattr(self, prefix + "weighted_sq") / getattr(self, prefix + "loss_count")).item()
                    old = aux["loss_gaussian_" + component]
                    aux["loss"] += exact - old
                    aux["loss_gaussian_" + component] = exact
        if self.interaction_count > 0:
            count = self.interaction_count
            aux["mae_interaction_energy"] = to_numpy(self.interaction_abs / count).item()
            aux["rmse_interaction_energy"] = to_numpy((self.interaction_sq / count).sqrt()).item()
            aux["mae_interaction_energy_per_atom"] = to_numpy(self.interaction_abs_per_atom / count).item()
            aux["rmse_interaction_energy_per_atom"] = to_numpy((self.interaction_sq_per_atom / count).sqrt()).item()
            exact_loss = to_numpy(self.loss_fn.interaction_energy_weight * self.interaction_weighted_sq_per_atom / count).item()
            aux["loss"] += exact_loss - aux["loss_interaction_energy"]
            aux["loss_interaction_energy"] = exact_loss
        if self.interaction_force_count > 0:
            count = self.interaction_force_count
            aux["mae_interaction_forces"] = to_numpy(self.interaction_force_abs / count).item()
            aux["rmse_interaction_forces"] = to_numpy((self.interaction_force_sq / count).sqrt()).item()
            exact_loss = to_numpy(self.loss_fn.interaction_force_weight * self.interaction_force_weighted_sq / count).item()
            aux["loss"] += exact_loss - aux["loss_interaction_forces"]
            aux["loss_interaction_forces"] = exact_loss
        if self.fragment_count > 0:
            for name in ("fragment_charge", "fragment_spin"):
                aux["mae_" + name] = to_numpy(getattr(self, name + "_abs") / self.fragment_count).item()
                aux["rmse_" + name] = to_numpy((getattr(self, name + "_sq") / self.fragment_count).sqrt()).item()
                weight = getattr(self.loss_fn, name + "_weight")
                exact_loss = to_numpy(weight * getattr(self, name + "_weighted_sq") / self.fragment_count).item()
                aux["loss"] += exact_loss - aux["loss_" + name]
                aux["loss_" + name] = exact_loss
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.MagFs_computed:
            MagFs = self.convert(self.MagFs)
            delta_MagFs = self.convert(self.delta_MagFs)
            aux["mae_magf"] = compute_mae(delta_MagFs)
            aux["rel_mae_magf"] = compute_rel_mae(delta_MagFs, MagFs)
            aux["rmse_magf"] = compute_rmse(delta_MagFs)
            aux["rel_rmse_magf"] = compute_rel_rmse(delta_MagFs, MagFs)
            aux["q95_magf"] = compute_q95(delta_MagFs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.Mus_field_computed:
            delta_mus_field = self.convert(self.delta_mus_field)
            aux["mae_mu_field"] = compute_mae(delta_mus_field)
            aux["rmse_mu_field"] = compute_rmse(delta_mus_field)
        if self.Qs_computed:
            delta_qs = self.convert(self.delta_qs)
            aux["mae_quadrupole"] = compute_mae(delta_qs)
            aux["rmse_quadrupole"] = compute_rmse(delta_qs)
            aux["q95_quadrupole"] = compute_q95(delta_qs)
        if self.SpinMus_computed:
            delta_spin_mus = self.convert(self.delta_spin_mus)
            aux["mae_spin_dipole"] = compute_mae(delta_spin_mus)
            aux["rmse_spin_dipole"] = compute_rmse(delta_spin_mus)
        if self.SpinQs_computed:
            delta_spin_qs = self.convert(self.delta_spin_qs)
            aux["mae_spin_quadrupole"] = compute_mae(delta_spin_qs)
            aux["rmse_spin_quadrupole"] = compute_rmse(delta_spin_qs)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        for quantity in ("alpha_field_A3", "alpha_multipole_A3", "MM_forces"):
            count = getattr(self, "polar2_" + quantity + "_count")
            if count > 0:
                aux["mae_" + quantity] = float(getattr(self, "polar2_" + quantity + "_abs") / count)
                aux["rmse_" + quantity] = float((getattr(self, "polar2_" + quantity + "_sq") / count).sqrt())
        return aux["loss"], aux
