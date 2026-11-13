"""Regression coverage for masked multi-source validation and dual-alpha logs."""
import logging
from types import SimpleNamespace

import pytest
import torch

from mace.modules.polar2_loss import POLAR2Loss
from mace.tools.train import MACELoss, valid_err_log


def test_unlabelled_alpha_rows_do_not_enter_energy_or_dipole_metrics():
    batch = SimpleNamespace(
        ptr=torch.tensor([0, 2, 4]), batch=torch.tensor([0, 0, 1, 1]),
        MM_batch=torch.empty(0, dtype=torch.long), weight=torch.ones(2), num_graphs=2,
    )
    shapes = dict(energy=(2,), forces=(4, 3), MM_forces=(0, 3), dipole=(2, 3),
                  quadrupole=(2, 3, 3), spin_dipole=(2, 3),
                  spin_quadrupole=(2, 3, 3), polarizability=(2, 3, 3))
    for name, shape in shapes.items():
        setattr(batch, name, torch.zeros(shape))
        setattr(batch, name + '_weight', torch.zeros(2))
    batch.energy = torch.tensor([2., 1.e6])
    batch.energy_weight = torch.tensor([1., 0.])
    batch.dipole_weight = torch.tensor([[1., 1., 1.], [0., 0., 0.]])
    batch.polarizability_weight = torch.tensor([0., 1.])
    mu = torch.tensor([[1., 1., 1.], [1.e6, 1.e6, 1.e6]])
    pred = dict(energy=torch.zeros(2), dipole=mu,
                total_dipole_from_electric_field=mu,
                polar2_alpha_field=torch.full((2, 3, 3), 2. / 14.3996454784255),
                polar2_alpha_multipole=torch.full((2, 3, 3), 3. / 14.3996454784255))
    metric = MACELoss(POLAR2Loss())
    metric.update(batch, pred)
    _, aux = metric.compute()
    assert aux['mae_e_per_atom'] == pytest.approx(1.)
    assert aux['rmse_e_per_atom'] == pytest.approx(1.)
    assert aux['mae_mu_per_atom'] == pytest.approx(.5)
    assert aux['mae_alpha_field_A3'] == pytest.approx(2.)
    assert aux['rmse_alpha_multipole_A3'] == pytest.approx(3.)


def test_text_validation_reports_both_alpha_branches_and_mm_forces(caplog):
    metrics = dict(rmse_e_per_atom=.01, rmse_f=.1,
                   mae_alpha_field_A3=2., rmse_alpha_field_A3=3.,
                   mae_alpha_multipole_A3=4., rmse_alpha_multipole_A3=5.,
                   loss_MM_forces=.2, mae_MM_forces=.01, rmse_MM_forces=.02)
    with caplog.at_level(logging.INFO):
        valid_err_log(1., metrics, SimpleNamespace(log=lambda _: None),
                      'PerAtomRMSE', 1, 'omol')
    for text in ['MAE_alpha_field[SPICE-alpha]=2.000000 Å³',
                 'RMSE_alpha_field[SPICE-alpha]=3.000000 Å³',
                 'MAE_alpha_multipole[SPICE-alpha]=4.000000 Å³',
                 'RMSE_alpha_multipole[SPICE-alpha]=5.000000 Å³',
                 'L_F_MM=0.200000', 'RMSE_F_MM=0.020000 eV/Å']:
        assert text in caplog.text
