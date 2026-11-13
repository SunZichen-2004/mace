"""Hybrid routing, native updates, scheduler changes and exact checkpoint resume."""

import copy

import numpy as np
import pytest
import torch

from mace import data, modules, tools
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.muon import MuonAdamW
from mace.tools.scripts_utils import LRScheduler, get_optimizer, get_params_options
from mace.tools.train import take_step
from tests.unit.test_optimizer_param_groups import build_scale_shift_mace


def _args(*extra):
    return build_default_arg_parser().parse_args(
        ["--name=test", "--optimizer=muon", "--lr=0.001", "--muon_lr=0.01", *extra]
    )


def _parameters():
    generator = torch.Generator().manual_seed(123)
    return [
        torch.nn.Parameter(torch.randn(shape, dtype=torch.float64, generator=generator))
        for shape in ((4, 3), (3,), (2, 3))
    ]


def _optimizer(params, **kwargs):
    return MuonAdamW(
        [
            {
                "name": "mixed",
                "params": params,
                "param_names": [
                    "interactions.0.conv_tp_weights.layer0.weight",
                    "interactions.0.linear.weight",
                    "field_dependent_charges_maps.0.tp_mix.weight",
                ],
                "lr": 0.002,
                "weight_decay": 0.03,
            }
        ],
        lr=0.001,
        muon_lr=0.01,
        **kwargs,
    )


def test_muon_reports_missing_native_backend(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    with pytest.raises(ImportError, match="PyTorch >= 2.9"):
        _optimizer(_parameters())


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="requires PyTorch >= 2.9")
class TestMuon:
    def test_energy_and_force_training_updates_both_parameter_sets(self):
        model = build_scale_shift_mace()
        args = _args()
        optimizer = get_optimizer(args, get_params_options(args, model))
        config = data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array([[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]]),
            properties={"energy": 0.0, "forces": np.zeros((3, 3))},
            property_weights={"energy": 1.0, "forces": 1.0},
            cell=np.eye(3) * 10,
            pbc=(False, False, False),
        )
        graph = data.AtomicData.from_config(
            config, z_table=tools.AtomicNumberTable([1, 8]), cutoff=4.0
        )
        batch = tools.torch_geometric.batch.Batch.from_data_list([graph])
        before = {id(p): p.detach().clone() for p in model.parameters()}
        loss, metrics = take_step(
            model=model,
            loss_fn=modules.WeightedEnergyForcesLoss(energy_weight=1, forces_weight=10),
            batch=batch,
            optimizer=optimizer,
            ema=None,
            output_args={"forces": True, "virials": False, "stress": False},
            max_grad_norm=1.0,
            device=torch.device("cpu"),
        )
        assert torch.isfinite(loss)
        assert np.isfinite(metrics["rmse_f"])
        for use_muon in (True, False):
            assert any(
                not torch.equal(p, before[id(p)])
                for group in optimizer.param_groups
                if group["use_muon"] == use_muon
                for p in group["params"]
            )

    def test_real_mace_routes_only_radial_matrices_and_preserves_factors(self):
        model = build_scale_shift_mace().double()
        args = _args('--lr_params_factors={"interactions_lr_factor":0.25}')
        options = get_params_options(args, model)
        optimizer = get_optimizer(args, options)
        assigned = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(assigned) == len({id(p) for p in assigned})
        assert {id(p) for p in assigned} == {id(p) for p in model.parameters()}
        muon_names = {
            name
            for group in optimizer.param_groups
            if group["use_muon"]
            for name in group["param_names"]
        }
        assert muon_names == {
            name
            for name, p in model.named_parameters()
            if ".conv_tp_weights." in name and p.ndim == 2
        }
        assert muon_names
        for group in optimizer.param_groups:
            if group["use_muon"]:
                assert group["lr"] == pytest.approx(0.0025)
            elif group["name"].startswith("interactions"):
                assert group["lr"] == pytest.approx(0.00025)

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
    )
    def test_updates_match_separate_native_optimizers(self, device):
        params = [torch.nn.Parameter(p.detach().to(device)) for p in _parameters()]
        expected = [torch.nn.Parameter(p.detach().clone()) for p in params]
        optimizer = _optimizer(params)
        muon = torch.optim.Muon([expected[0]], lr=0.02, weight_decay=0.03)
        adamw = torch.optim.AdamW(
            expected[1:], lr=0.002, weight_decay=0.03, amsgrad=True
        )
        for _ in range(3):
            for p, ref in zip(params, expected):
                p.grad = p.detach().sin()
                ref.grad = p.grad.clone()
            optimizer.step()
            muon.step()
            adamw.step()
        for p, ref in zip(params, expected):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
        assert "momentum_buffer" in optimizer.state[params[0]]
        assert "exp_avg" in optimizer.state[params[1]]
        optimizer.zero_grad()
        assert all(p.grad is None for p in params)

    def test_checkpoint_resume_and_scheduler_control_actual_updates(self, tmp_path):
        params = _parameters()
        optimizer = _optimizer(params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        for p in params:
            p.grad = torch.ones_like(p)
        optimizer.step()
        scheduler.step()
        path = tmp_path / "optimizer.pt"
        torch.save(
            {
                "params": [p.detach() for p in params],
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            path,
        )
        saved = torch.load(path, weights_only=True)
        restored = [torch.nn.Parameter(p.clone()) for p in saved["params"]]
        resumed = _optimizer(restored)
        resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(resumed, gamma=0.5)
        resumed.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        assert [g["lr"] for g in resumed.param_groups] == pytest.approx([0.01, 0.001])
        # After load, child optimizers must see scheduler changes to these groups.
        for opt, ps in [(optimizer, params), (resumed, restored)]:
            for p in ps:
                p.grad = torch.full_like(p, 0.25)
            opt.step()
        scheduler.step()
        resumed_scheduler.step()
        for p, ref in zip(params, restored):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
        assert [g["lr"] for g in resumed.param_groups] == pytest.approx([0.005, 0.0005])

    def test_zero_lr_freezes_muon_and_adamw_parameters(self):
        params = _parameters()
        groups = [
            {
                "params": params,
                "param_names": [
                    "interactions.0.conv_tp_weights.layer0.weight",
                    "readouts.bias",
                    "tp_mix.weight",
                ],
                "lr": 0.0,
                "weight_decay": 0.5,
            }
        ]
        optimizer = MuonAdamW(groups, lr=0.001, muon_lr=0.02)
        before = [p.detach().clone() for p in params]
        for p in params:
            p.grad = torch.ones_like(p)
        optimizer.step()
        for p, ref in zip(params, before):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)

    def test_closure_runs_once(self):
        params = _parameters()
        optimizer = _optimizer(params)
        calls = []

        def closure():
            calls.append(1)
            optimizer.zero_grad()
            loss = sum(p.square().sum() for p in params)
            loss.backward()
            return loss

        assert optimizer.step(closure).isfinite()
        assert len(calls) == 1

    def test_mace_scheduler_and_stage_two_support(self):
        model = build_scale_shift_mace().double()
        args = _args()
        optimizer = get_optimizer(args, get_params_options(args, model))
        scheduler = LRScheduler(optimizer, args)
        scheduler.step(metrics=1.0)
        swa = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0001, anneal_epochs=1)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        swa.step()
        assert all(g["lr"] == pytest.approx(0.0001) for g in optimizer.param_groups)

    @pytest.mark.parametrize("steps", [0, -1, 100])
    def test_invalid_newton_schulz_steps_are_rejected(self, steps):
        with pytest.raises(ValueError, match="muon_ns_steps"):
            _optimizer(_parameters(), ns_steps=steps)

    def test_wrong_optimizer_checkpoint_is_rejected(self):
        optimizer = _optimizer(_parameters())
        state = copy.deepcopy(optimizer.state_dict())
        state["param_groups"][0].pop("use_muon")
        with pytest.raises(ValueError, match="do not match"):
            optimizer.load_state_dict(state)
