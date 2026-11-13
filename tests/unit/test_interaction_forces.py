"""Interaction forces map complex atoms back to independently predicted fragments."""

from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mace.data.atomic_data import AtomicData
from mace.data.interaction_dataset import InteractionSample, collate_interactions
from mace.data.utils import Configuration
from mace.modules.loss import (
    InteractionEnergyLoss,
    WeightedEnergyForcesLoss,
    interaction_force_l2,
)
from mace.tools.train import MACELoss
from mace.tools.utils import AtomicNumberTable


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def graph(n):
    c = Configuration(
        np.ones(n, dtype=int),
        np.arange(n * 3).reshape(n, 3).astype(float),
        dict(energy=0.0, forces=np.zeros((n, 3)), head="Default"),
        dict(energy=1.0, forces=1.0),
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
        weight=0.0,
    )
    return AtomicData.from_config(
        c, z_table=AtomicNumberTable([1]), cutoff=5.0, heads=["Default"]
    )


def sample():
    return InteractionSample(
        [graph(4), graph(2), graph(2)],
        0.0,
        1.0,
        np.zeros((4, 3)),
        np.array([2, 0, 3, 1]),
    )


def test_mapping_component_mse_and_opposite_force_gradients():
    batch = collate_interactions([sample()])
    assert batch.interaction_force_isolated.tolist() == [6, 4, 7, 5]
    forces = torch.zeros(8, 3)
    forces[0, 0] = 1
    forces[1, 0] = -1
    forces.requires_grad_()
    pred = {"energy": torch.zeros(3, requires_grad=True), "forces": forces}
    loss = InteractionEnergyLoss(WeightedEnergyForcesLoss(), 1000, 1000)(batch, pred)
    assert loss.item() == pytest.approx(1000 / 6)
    loss.backward()
    assert forces.grad[0, 0].item() == pytest.approx(1000 / 6)
    assert forces.grad[6, 0].item() == pytest.approx(-1000 / 6)
    assert forces.grad[1, 0].item() == pytest.approx(-1000 / 6)
    assert forces.grad[4, 0].item() == pytest.approx(1000 / 6)


def test_validation_reports_force_loss_and_rmse():
    batch = collate_interactions([sample()])
    forces = torch.zeros(8, 3)
    forces[:4] = 1
    pred = {"energy": torch.zeros(3), "forces": forces}
    metric = MACELoss(InteractionEnergyLoss(WeightedEnergyForcesLoss(), 1000, 1000))
    metric.update(batch, pred)
    _, result = metric.compute()
    assert result["rmse_interaction_forces"] == pytest.approx(1)
    assert result["loss_interaction_forces"] == pytest.approx(1000)
    assert result["loss"] == pytest.approx(1000)


def test_local_energy_and_force_cancel_with_full_parameter_gradients():
    # E_AB=local*(xA^2+xB^2)+k*(xB-xA)^2/2; isolated energies omit the k term.
    x = torch.tensor([0.0, 2.0, 0.0, 2.0], requires_grad=True)
    local = torch.tensor(3.0, requires_grad=True)
    k = torch.tensor(1.0, requires_grad=True)
    energy = local * x.square().sum() + k * (x[1] - x[0]).square() / 2
    fx = -torch.autograd.grad(energy, x, create_graph=True)[0]
    forces = torch.stack([fx, fx * 0, fx * 0], dim=1)
    ref = SimpleNamespace(
        interaction_force_full=torch.tensor([0, 1]),
        interaction_force_isolated=torch.tensor([2, 3]),
        interaction_force_targets=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        interaction_force_weights=torch.ones(2),
    )
    loss = 1000 * interaction_force_l2(ref, {"forces": forces}, ddp=False)
    assert loss.item() == pytest.approx(1000 / 3)
    loss.backward()
    assert local.grad.item() == pytest.approx(0.0, abs=1e-10)
    assert k.grad.item() == pytest.approx(4000 / 3)


def test_no_force_targets_has_zero_finite_gradient():
    batch = collate_interactions([InteractionSample([graph(2)])])
    forces = torch.zeros(2, 3, requires_grad=True)
    loss = interaction_force_l2(batch, {"forces": forces}, ddp=False)
    loss.backward()
    assert loss.item() == 0 and torch.all(forces.grad == 0)
