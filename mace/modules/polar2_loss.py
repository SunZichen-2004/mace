"""Masked POLAR-2 objective, with independently supervised dipole responses.

Energy: L1 per atom; forces: mean per-atom L2 norm; moments and full Cartesian
polarizability: component MSE after division by atom count. Both electric
branches receive half of the corresponding property weight. No consistency
penalty. All labels and predictions refer to the same ORCA COM frame.
"""

import torch
from torch import distributed as dist


class POLAR2Loss(torch.nn.Module):
    is_polar2 = True

    def __init__(
        self,
        energy_weight=10.0,
        forces_weight=10.0,
        MM_forces_weight=10.0,
        dipole_weight=1.0,
        quadrupole_weight=1.0,
        spin_dipole_weight=1.0,
        spin_quadrupole_weight=1.0,
        polarizability_weight=1.0,
    ):
        super().__init__()
        for name, value in locals().copy().items():
            if name.endswith("_weight"):
                self.register_buffer(name, torch.tensor(float(value)))

    def forward_kwargs(self, batch, kwargs):
        kwargs = dict(kwargs)
        kwargs["compute_force"] = bool(
            (float(self.forces_weight) > 0 and (batch.forces_weight > 0).any())
            or (float(self.MM_forces_weight) > 0 and (batch.MM_forces_weight > 0).any())
        )
        kwargs["compute_polar2_response"] = bool(
            float(self.polarizability_weight) > 0
            and (batch.polarizability_weight > 0).any()
        )
        return kwargs

    def component_counts(self, ref, ddp=True):
        result = {}
        for name in ("energy", "forces", "MM_forces", "dipole", "quadrupole", "spin_dipole", "spin_quadrupole", "polarizability"):
            target = getattr(ref, name)
            if name in ("quadrupole", "spin_quadrupole"):
                target = target.reshape(-1, 3, 3)
            mask = getattr(ref, name + "_weight").to(target)
            if name in ("forces", "MM_forces"):
                ids = ref.batch if name == "forces" else ref.MM_batch
                mask = mask.reshape(-1)[ids] * ref.weight[ids]
            else:
                mask = mask * ref.weight.reshape((-1,) + (1,) * (mask.ndim - 1))
            while mask.ndim < target.ndim:
                mask = mask.unsqueeze(-1)
            count = (mask.expand_as(target) != 0).sum().to(target)
            if name == 'forces':
                count = count / 3  # Official L2 loss averages over atoms, not components.
            if ddp and dist.is_available() and dist.is_initialized():
                dist.all_reduce(count)
            result[name] = count
        return result

    def weighted_components(self, ref, pred, ddp=True, global_counts=None):
        n = (ref.ptr[1:] - ref.ptr[:-1]).to(ref.energy)
        zero = pred["energy"].sum() * 0.0
        parts = {}
        specifications = [
            ("energy", ["energy"], "energy", True),
            ("forces", ["forces"], "forces", False),
            ("MM_forces", ["MM_forces"], "MM_forces", False),
            ("dipole", ["dipole", "total_dipole_from_electric_field"], "dipole", False),
            ("quadrupole", ["quadrupole"], "quadrupole", False),
            ("spin_dipole", ["spin_dipole"], "spin_dipole", False),
            ("spin_quadrupole", ["spin_quadrupole"], "spin_quadrupole", False),
            (
                "polarizability",
                ["polar2_alpha_field", "polar2_alpha_multipole"],
                "polarizability",
                False,
            ),
        ]
        for name, branches, target_name, absolute in specifications:
            if float(getattr(self, name + "_weight")) == 0.0:
                parts[name] = zero
                continue
            target = getattr(ref, target_name)
            if name in ("quadrupole", "spin_quadrupole"):
                target = target.reshape(-1, 3, 3)
            mask = getattr(ref, name + "_weight").to(target)
            if name in ("forces", "MM_forces"):
                graph_ids = ref.batch if name == "forces" else ref.MM_batch
                mask = mask.reshape(-1)[graph_ids] * ref.weight[graph_ids]
            else:
                mask = mask * ref.weight.reshape((-1,) + (1,) * (mask.ndim - 1))
            while mask.ndim < target.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(target)
            count = (mask != 0).sum().to(target)
            if name == 'forces':
                count = count / 3
            global_count = count.detach().clone()
            distributed = ddp and dist.is_available() and dist.is_initialized()
            if global_counts is not None:
                global_count = global_counts[name]
            elif distributed:
                dist.all_reduce(global_count)
            total = zero
            for branch in branches:
                if count.item() == 0:
                    continue
                prediction = pred.get(branch)
                if prediction is None:
                    raise ValueError(f"Missing active POLAR-2 prediction: {branch}")
                if name == "polarizability":
                    # Model response e A^2/V -> conventional polarizability A^3.
                    prediction = prediction * 14.3996454784255
                delta = prediction - target
                if name not in ("forces", "MM_forces"):
                    delta = delta / n.reshape((-1,) + (1,) * (delta.ndim - 1))
                # Select BEFORE squaring; missing/NaN labels cannot contaminate loss.
                active = mask != 0
                residual = delta[active].double()
                if name == "forces":
                    # Every atom has one scalar mask shared by its xyz components.
                    atom_weights = mask[active].reshape(-1, 3)[:, 0]
                    errors = torch.linalg.vector_norm(residual.reshape(-1, 3), ord=2, dim=-1)
                    total = total + (atom_weights * errors).sum()
                else:
                    total = total + (
                        mask[active] * (residual.abs() if absolute else residual.square())
                    ).sum() / len(branches)
            scale = dist.get_world_size() if distributed else 1
            parts[name] = (
                getattr(self, name + "_weight")
                * total
                * scale
                / global_count.clamp_min(1)
            )
        return parts

    def forward(self, ref, pred):
        return sum(self.weighted_components(ref, pred).values())

    def __repr__(self):
        return "POLAR2Loss(E: L1/N; F: mean vector L2 norm; D,Q,spin,alpha: MSE/N^2; dual D/alpha; masks; no consistency)"
