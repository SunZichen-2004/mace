###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional

import torch
import torch.distributed as dist

from mace.modules.utils import traceless_quadrupole
from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


DIPOLE_PRED_KEYS = {
    "dipole": ("dipole",),
    "field": ("total_dipole_from_electric_field",),
    "both": ("dipole", "total_dipole_from_electric_field"),
}


def weighted_mean_squared_error_dipole(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    pred_key: str = "dipole",
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred[pred_key]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


def dipole_mse_terms(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    pred_dipole_key: str = "dipole",
) -> torch.Tensor:
    """Sum of per-atom dipole MSEs.

    ``dipole`` is the charge-density dipole. ``field`` is ∂E/∂F.
    ``both`` adds the two terms with equal weight.
    """
    try:
        keys = DIPOLE_PRED_KEYS[pred_dipole_key]
    except KeyError as exc:
        raise ValueError(
            "pred_dipole_key must be 'dipole', 'field', or 'both', "
            f"got {pred_dipole_key!r}"
        ) from exc
    loss = weighted_mean_squared_error_dipole(ref, pred, ddp, pred_key=keys[0])
    for key in keys[1:]:
        loss = loss + weighted_mean_squared_error_dipole(ref, pred, ddp, pred_key=key)
    return loss


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool
    ] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype
    )
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        magforces_weight=1.0,
        huber_delta=0.01,
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "magforces_weight",
            torch.tensor(magforces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        configs_magforces_weight = torch.repeat_interleave(
            ref.magforces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss_magforces = reduce_loss(loss_magforces, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_magforces = 0
            if "magforces" in pred.keys() and (
                pred["magforces"] is not None and ref["magforces"] is not None
            ):
                loss_magforces = torch.nn.functional.huber_loss(
                    configs_magforces_weight * ref["magforces"],
                    configs_magforces_weight * pred["magforces"],
                    reduction="mean",
                    delta=self.huber_delta,
                )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
            + self.magforces_weight * loss_magforces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, magforces_weight={self.magforces_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.virials_weight * loss_virials
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = (
            weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        )  # scale adjustment
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


class DipolePolarLoss(torch.nn.Module):
    def __init__(
        self, dipole_weight=1.0, polarizability_weight=1.0
    ) -> (
        None
    ):  # dipole_mean=None,dipole_std=None,polarizability_mean=None,polarizability_std=None
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_dipole = weighted_mean_squared_error_dipole(
            ref, pred, ddp
        )  # ,self.dipole_mean,self.dipole_std) #* 100.0  # scale adjustment

        loss_polarizability = weighted_mean_squared_error_polarizability(
            ref, pred, ddp
        )  # ,self.polarizability_mean,self.polarizability_std) #* 100.0  # scale adjustment
        return (
            self.dipole_weight * loss_dipole
            + self.polarizability_weight * loss_polarizability
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dipole_weight={self.dipole_weight:.3f}, polarizability_weight={self.polarizability_weight:.3f})"
        )


_QUADRUPOLE_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def weighted_mean_squared_error_quadrupole(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    key: str = "quadrupole",
) -> torch.Tensor:
    n_graphs = int(ref.ptr.numel() - 1)
    pred_q = pred[key]
    ref_q = traceless_quadrupole(ref[key].view(n_graphs, 3, 3))
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    rows = []
    for i, j in _QUADRUPOLE_COMPONENTS:
        rows.append((ref_q[:, i, j] - pred_q[:, i, j]).unsqueeze(-1))
    raw_loss = torch.square(torch.cat(rows, dim=-1) / num_atoms)
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_spin_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["spin_dipole"] - pred["spin_dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        pred_dipole_key: str = "dipole",
    ) -> None:
        super().__init__()
        if pred_dipole_key not in DIPOLE_PRED_KEYS:
            raise ValueError(
                "pred_dipole_key must be 'dipole', 'field', or 'both', "
                f"got {pred_dipole_key!r}"
            )
        self.pred_dipole_key = pred_dipole_key
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = (
            dipole_mse_terms(ref, pred, ddp, self.pred_dipole_key) * 100.0
        )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f}, "
            f"pred_dipole_key={self.pred_dipole_key})"
        )


class WeightedEnergyForcesMultipoleSpinLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        quadrupole_weight=1.0,
        spin_dipole_weight=0.0,
        spin_quadrupole_weight=0.0,
        polarizability_weight=0.0,
        pred_dipole_key: str = "dipole",
    ) -> None:
        super().__init__()
        if pred_dipole_key not in DIPOLE_PRED_KEYS:
            raise ValueError(
                "pred_dipole_key must be 'dipole', 'field', or 'both', "
                f"got {pred_dipole_key!r}"
            )
        self.pred_dipole_key = pred_dipole_key
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "quadrupole_weight",
            torch.tensor(quadrupole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "spin_dipole_weight",
            torch.tensor(spin_dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "spin_quadrupole_weight",
            torch.tensor(spin_quadrupole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def weighted_components(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """Weighted terms that sum to ``forward``. Missing predictions are omitted."""
        parts: Dict[str, torch.Tensor] = {
            "energy": self.energy_weight
            * weighted_mean_squared_error_energy(ref, pred, ddp),
            "forces": self.forces_weight * mean_squared_error_forces(ref, pred, ddp),
            "dipole": self.dipole_weight
            * dipole_mse_terms(ref, pred, ddp, self.pred_dipole_key),
            "quadrupole": self.quadrupole_weight
            * weighted_mean_squared_error_quadrupole(ref, pred, ddp),
        }
        if pred.get("spin_dipole") is not None and getattr(ref, "spin_dipole", None) is not None:
            parts["spin_dipole"] = (
                self.spin_dipole_weight
                * weighted_mean_squared_error_spin_dipole(ref, pred, ddp)
            )
        if (
            pred.get("spin_quadrupole") is not None
            and getattr(ref, "spin_quadrupole", None) is not None
        ):
            parts["spin_quadrupole"] = (
                self.spin_quadrupole_weight
                * weighted_mean_squared_error_quadrupole(
                    ref, pred, ddp, key="spin_quadrupole"
                )
            )
        if self.polarizability_weight != 0:
            parts["polarizability"] = (
                self.polarizability_weight
                * weighted_mean_squared_error_polarizability(ref, pred, ddp)
            )
        return parts

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        parts = self.weighted_components(ref, pred, ddp)
        loss = torch.zeros((), dtype=torch.get_default_dtype(), device=ref.energy.device)
        for value in parts.values():
            loss = loss + value
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f}, "
            f"quadrupole_weight={self.quadrupole_weight:.3f}, "
            f"spin_dipole_weight={self.spin_dipole_weight:.3f}, "
            f"spin_quadrupole_weight={self.spin_quadrupole_weight:.3f}, "
            f"polarizability_weight={self.polarizability_weight:.3f}, "
            f"pred_dipole_key={self.pred_dipole_key})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, MM_forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

        self.register_buffer(
            "MM_forces_weight",
            torch.tensor(MM_forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        loss_MM_forces = loss_energy.new_zeros(())
        if getattr(ref, "MM_forces", None) is not None:
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
            if pred.get("MM_forces") is None:
                raise ValueError("MM force targets require model MM_forces.")
            if ref["MM_forces"].numel():
                loss_MM_forces = reduce_loss(
                    (ref["MM_forces"] - pred["MM_forces"]).square(), ddp
                )
        return (self.energy_weight * loss_energy + self.forces_weight * loss_forces
                + self.MM_forces_weight * loss_MM_forces)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


def interaction_energy_error(ref: Batch, pred: TensorDict) -> torch.Tensor:
    """Reference minus E(complex)-sum E(fragment), all from the current model."""
    energies = pred["energy"]
    terms = energies[ref.interaction_indices] * ref.interaction_coefficients
    predicted = energies.new_zeros(ref.interaction_targets.shape).index_add(
        0, ref.interaction_sample_ids, terms
    )
    return ref.interaction_targets - predicted


def interaction_energy_l2(ref: Batch, pred: TensorDict, ddp=None) -> torch.Tensor:
    error = interaction_energy_error(ref, pred) / ref.interaction_num_atoms
    raw = ref.interaction_weights * error.square()
    # An all-isolated local batch still participates in DDP count reduction.
    total = raw.sum() + pred["energy"].sum() * 0.0
    count = raw.new_tensor(raw.numel())
    enabled = is_ddp_enabled() if ddp is None else ddp
    if enabled and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total = total * dist.get_world_size()
    return total / count.clamp_min(1)


def interaction_force_error(ref: Batch, pred: TensorDict):
    forces = pred.get("forces")
    if forces is None:
        raise ValueError("Interaction-force loss requires computed atomic forces")
    return ref.interaction_force_targets - (
        forces[ref.interaction_force_full] - forces[ref.interaction_force_isolated])


def interaction_force_l2(ref: Batch, pred: TensorDict, ddp=None):
    error = interaction_force_error(ref, pred)
    total = (ref.interaction_force_weights[:, None] * error.square()).sum() + pred["forces"].sum() * 0.0
    count = error.new_tensor(error.numel())
    enabled = is_ddp_enabled() if ddp is None else ddp
    if enabled and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total = total * dist.get_world_size()
    return total / count.clamp_min(1)


def interaction_base_view(ref: Batch, pred: TensorDict):
    """Exclude generated fragment graphs from ordinary E/F/multipole losses.

    This preserves the original batch's loss normalization and does not dilute
    isolated-source supervision by the number/size of auxiliary fragments.
    """
    graph_ids, node_ids = ref.interaction_base_graphs, ref.interaction_base_nodes
    sizes = (ref.ptr[1:] - ref.ptr[:-1])[graph_ids]
    base = Batch(ptr=torch.cat((sizes.new_zeros(1), sizes.cumsum(0))))
    node_fields = ("forces", "magforces", "charges", "magmom")
    graph_fields = (
        "energy",
        "weight",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "virials_weight",
        "dipole_weight",
        "charges_weight",
        "polarizability_weight",
        "magforces_weight",
        "spin_dipole_weight",
        "spin_quadrupole_weight",
        "quadrupole_weight",
        "stress",
        "virials",
        "dipole",
        "quadrupole",
        "spin_dipole",
        "spin_quadrupole",
        "polarizability",
        "total_charge",
        "total_spin",
    )
    for key in graph_fields + node_fields:
        value = getattr(ref, key, None)
        if value is not None:
            base[key] = value[node_ids if key in node_fields else graph_ids]
    result = {}
    for key in (
        "energy",
        "stress",
        "virials",
        "dipole",
        "quadrupole",
        "spin_dipole",
        "spin_quadrupole",
        "polarizability",
        "total_dipole_from_electric_field",
        "forces",
        "magforces",
        "charges",
        "magmom",
    ):
        value = pred.get(key)
        if value is not None:
            result[key] = value[node_ids if key in node_fields else graph_ids]
    return base, result


class InteractionEnergyLoss(torch.nn.Module):
    """Ordinary loss on source samples plus per-atom interaction-energy L2."""

    def __init__(self, base_loss, interaction_energy_weight=100.0, interaction_force_weight=0.0):
        super().__init__()
        if not isinstance(
            base_loss, (WeightedEnergyForcesLoss, WeightedEnergyForcesMultipoleSpinLoss)
        ):
            raise ValueError(
                "Interaction training supports weighted or energy_forces_multipole_spin/dipole_quadrupole loss"
            )
        self.base_loss = base_loss
        self.register_buffer(
            "interaction_energy_weight",
            torch.tensor(interaction_energy_weight, dtype=torch.get_default_dtype()),
        )

        if not 0 <= interaction_force_weight < float("inf"):
            raise ValueError("interaction_force_weight must be finite and nonnegative")
        self.register_buffer("interaction_force_weight", torch.tensor(
            interaction_force_weight, dtype=torch.get_default_dtype()))

    def weighted_components(self, ref, pred, ddp=None):
        base_ref, base_pred = interaction_base_view(ref, pred)
        if hasattr(self.base_loss, "weighted_components"):
            parts = self.base_loss.weighted_components(base_ref, base_pred, ddp)
        else:
            parts = {
                "energy": self.base_loss.energy_weight
                * weighted_mean_squared_error_energy(base_ref, base_pred, ddp),
                "forces": self.base_loss.forces_weight
                * mean_squared_error_forces(base_ref, base_pred, ddp),
            }
        parts["interaction_energy"] = (
            self.interaction_energy_weight * interaction_energy_l2(ref, pred, ddp)
        )
        if self.interaction_force_weight != 0:
            parts["interaction_forces"] = self.interaction_force_weight * interaction_force_l2(ref, pred, ddp)
        return parts

    def forward(self, ref, pred, ddp=None):
        return sum(self.weighted_components(ref, pred, ddp).values())


def fragment_charge_spin_error(ref: Batch, pred: TensorDict):
    """Reference minus sums of final atomic charge and signed spin monopoles."""
    errors = []
    for key, targets in (
        ("charges", ref.fragment_charge_targets),
        ("spins", ref.fragment_spin_targets),
    ):
        values = pred.get(key)
        if values is None or values.numel() != ref.positions.shape[0]:
            raise ValueError(
                "Fragment loss requires final atomic charges and spins from the model"
            )
        values = values.reshape(-1)
        totals = values.new_zeros(targets.shape).index_add(
            0, ref.fragment_ids, values[ref.fragment_node_indices]
        )
        errors.append(targets - totals)
    return tuple(errors)


def fragment_charge_spin_l2(ref: Batch, pred: TensorDict, ddp=None):
    errors = fragment_charge_spin_error(ref, pred)
    count = errors[0].new_tensor(errors[0].numel())
    enabled = is_ddp_enabled() if ddp is None else ddp
    scale = 1
    if enabled and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        scale = dist.get_world_size()
    # All ranks participate even when some/all have no labeled fragments.
    return tuple(
        ((ref.fragment_weights * error.square()).sum() + pred[key].sum() * 0.0)
        * scale
        / count.clamp_min(1)
        for error, key in zip(errors, ("charges", "spins"))
    )


class FragmentChargeSpinLoss(torch.nn.Module):
    """Add per-fragment total charge/spin MSE; no atom-count division."""

    def __init__(
        self, base_loss, fragment_charge_weight=1000.0, fragment_spin_weight=1000.0
    ):
        super().__init__()
        if not isinstance(
            base_loss, (WeightedEnergyForcesLoss, WeightedEnergyForcesMultipoleSpinLoss)
        ):
            raise ValueError(
                "Fragment training supports weighted and energy_forces_multipole_spin/dipole_quadrupole losses"
            )
        self.base_loss = base_loss
        for name, value in (
            ("fragment_charge_weight", fragment_charge_weight),
            ("fragment_spin_weight", fragment_spin_weight),
        ):
            if not 0 <= value < float("inf"):
                raise ValueError("Fragment loss weights must be finite and nonnegative")
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.get_default_dtype())
            )

    def weighted_components(self, ref, pred, ddp=None):
        if hasattr(self.base_loss, "weighted_components"):
            parts = self.base_loss.weighted_components(ref, pred, ddp)
        else:
            parts = {
                "energy": self.base_loss.energy_weight
                * weighted_mean_squared_error_energy(ref, pred, ddp),
                "forces": self.base_loss.forces_weight
                * mean_squared_error_forces(ref, pred, ddp),
            }
        charge, spin = fragment_charge_spin_l2(ref, pred, ddp)
        parts["fragment_charge"] = self.fragment_charge_weight * charge
        parts["fragment_spin"] = self.fragment_spin_weight * spin
        return parts

    def forward(self, ref, pred, ddp=None):
        return sum(self.weighted_components(ref, pred, ddp).values())
