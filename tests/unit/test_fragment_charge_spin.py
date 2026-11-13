"""Fragment sums, signed spin conventions, gradients and opt-in regressions."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mace.data.atomic_data import AtomicData
from mace.data.fragment_dataset import (
    FragmentDataset,
    FragmentSample,
    collate_fragments,
)
from mace.data.hdf5_dataset import HDF5Dataset
from mace.data.utils import Configuration, save_configurations_as_HDF5
from mace.modules.loss import (
    FragmentChargeSpinLoss,
    WeightedEnergyForcesLoss,
    fragment_charge_spin_error,
    fragment_charge_spin_l2,
)
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.train import MACELoss
from mace.tools.utils import AtomicNumberTable


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def config(n=4):
    return Configuration(
        np.ones(n, dtype=int),
        np.arange(3 * n).reshape(n, 3).astype(float),
        dict(
            energy=0.0,
            forces=np.zeros((n, 3)),
            total_charge=0.0,
            total_spin=1.0,
            head="Default",
        ),
        dict(energy=1.0, forces=1.0),
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
    )


def sample():
    graph = AtomicData.from_config(
        config(), z_table=AtomicNumberTable([1]), cutoff=5.0, heads=["Default"]
    )
    return FragmentSample(
        graph, np.array([0, 0, 1, 1]), np.array([1.0, -1.0]), np.array([0.0, 0.0])
    )


def prediction(batch):
    return dict(
        energy=batch.energy.clone(),
        forces=batch.forces.clone(),
        charges=torch.tensor([0.3, 0.5, -0.2, -0.6], requires_grad=True),
        spins=torch.tensor([0.1, 0.2, -0.1, -0.2], requires_grad=True),
    )


def test_hand_calculated_fragment_totals_l2_and_gradients():
    batch = collate_fragments([sample()])
    pred = prediction(batch)
    parts = FragmentChargeSpinLoss(WeightedEnergyForcesLoss()).weighted_components(
        batch, pred, ddp=False
    )
    # Fragment Q=.8,-.8 vs +1,-1; S=.3,-.3 vs 0,0.
    assert parts["fragment_charge"].item() == pytest.approx(40.0)
    assert parts["fragment_spin"].item() == pytest.approx(90.0)
    sum(parts.values()).backward()
    torch.testing.assert_close(
        pred["charges"].grad, torch.tensor([-200.0, -200.0, 200.0, 200.0])
    )
    torch.testing.assert_close(
        pred["spins"].grad, torch.tensor([300.0, 300.0, -300.0, -300.0])
    )
    assert pred["charges"].sum().item() == pytest.approx(0.0)
    assert pred["spins"].sum().item() == pytest.approx(0.0)


def test_no_atom_count_division_and_no_cross_graph_group_mixing():
    first = sample()
    first.ids = np.array([0, 1, 1, 1])
    first.charges = np.array([0.0, 0.0])
    batch = collate_fragments([first, sample()])
    assert batch.fragment_ids.tolist() == [0, 1, 1, 1, 2, 2, 3, 3]
    pred = dict(
        charges=torch.tensor([0.3, 0.2, 0.1, -0.6, 0.5, 0.5, -0.5, -0.5]),
        spins=torch.zeros(8),
    )
    charge, spin = fragment_charge_spin_l2(batch, pred, ddp=False)
    # First two fragment errors +/- .3, last two exactly zero: mean=.045.
    assert charge.item() == pytest.approx(0.045)
    assert spin.item() == 0


def test_ordinary_energy_forces_losses_unchanged():
    batch = collate_fragments([sample()])
    pred = prediction(batch)
    pred["energy"] = batch.energy + 4.0
    pred["forces"] = batch.forces + 2.0
    parts = FragmentChargeSpinLoss(WeightedEnergyForcesLoss()).weighted_components(
        batch, pred, ddp=False
    )
    assert parts["energy"].item() == pytest.approx(1.0)
    assert parts["forces"].item() == pytest.approx(4.0)


def test_validation_has_correct_charge_spin_metrics():
    batch = collate_fragments([sample()])
    metric = MACELoss(FragmentChargeSpinLoss(WeightedEnergyForcesLoss()))
    metric.update(batch, prediction(batch))
    _, result = metric.compute()
    assert result["rmse_fragment_charge"] == pytest.approx(0.2)
    assert result["rmse_fragment_spin"] == pytest.approx(0.3)
    assert result["loss_fragment_charge"] == pytest.approx(40.0)
    assert result["loss_fragment_spin"] == pytest.approx(90.0)
    assert result["loss"] == pytest.approx(130.0)


def make_dataset(path):
    with h5py.File(path, "w") as h:
        save_configurations_as_HDF5([config(), config(2)], None, h)
        meta = h["config_batch_0/config_0"].create_group("metadata")
        meta["molecule_group_index"] = [0, 0, 0, 1]
        meta["molecule_charges"] = [1, -1]
        meta["molecule_multiplicities"] = [1, 1]
    return HDF5Dataset(
        str(path), r_max=5.0, z_table=AtomicNumberTable([1]), heads=["Default"]
    )


def test_closed_shell_multiplicity_one_maps_to_zero_spin(tmp_path):
    ds = FragmentDataset(make_dataset(tmp_path / "test.h5"))
    row = ds[0]
    assert row.spins.tolist() == [0.0, 0.0]
    batch = collate_fragments([ds[1], row])
    assert batch.num_graphs == 2 and batch.num_nodes == 6
    assert batch.fragment_node_indices.tolist() == [2, 3, 4, 5]
    assert batch.fragment_ids.tolist() == [0, 0, 0, 1]
    assert float(batch.weight.sum()) == 2


def test_signed_opposite_fragment_spins(tmp_path):
    path = tmp_path / "test.h5"
    ds = make_dataset(path)
    ds.file.close()
    ds._file = None
    with h5py.File(path, "r+") as h:
        h["config_batch_0/config_0/metadata/molecule_spins"] = [1, -1]
    assert FragmentDataset(ds)[0].spins.tolist() == [1, -1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("molecule_charges", [1, 1]),
        ("molecule_multiplicities", [2, 2]),
        ("molecule_group_index", [0, 0, 0, 0]),
    ],
)
def test_invalid_group_targets_fail_early(tmp_path, key, value):
    path = tmp_path / "test.h5"
    ds = make_dataset(path)
    ds.file.close()
    ds._file = None
    with h5py.File(path, "r+") as h:
        h["config_batch_0/config_0/metadata/" + key][:] = value
    with pytest.raises(ValueError):
        FragmentDataset(ds)[0]


def test_no_labels_is_finite_and_cli_defaults_off():
    row = sample()
    batch = collate_fragments(
        [FragmentSample(row.graph, np.empty(0, dtype=int), np.empty(0), np.empty(0))]
    )
    pred = prediction(batch)
    charge, spin = fragment_charge_spin_l2(batch, pred, ddp=False)
    (charge + spin).backward()
    assert charge.item() == spin.item() == 0
    assert torch.all(pred["charges"].grad == 0) and torch.all(pred["spins"].grad == 0)
    args = build_default_arg_parser().parse_args(["--name=test"])
    assert not args.fragment_charge_spin_training
    assert args.fragment_charge_weight == args.fragment_spin_weight == 1000


def _ddp_worker(rank, rendezvous, output):
    torch.set_default_dtype(torch.float64)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    parameter = torch.tensor(0.2, requires_grad=True)
    for empty_all in (False, True):
        labeled = rank == 0 and not empty_all
        ref = SimpleNamespace(
            positions=torch.zeros(2, 3),
            fragment_ids=torch.tensor([0, 1] if labeled else [], dtype=torch.long),
            fragment_node_indices=torch.tensor(
                [0, 1] if labeled else [], dtype=torch.long
            ),
            fragment_charge_targets=torch.zeros(2 if labeled else 0),
            fragment_spin_targets=torch.zeros(2 if labeled else 0),
            fragment_weights=torch.ones(2 if labeled else 0),
        )
        pred = {
            "charges": torch.stack([parameter, -parameter]),
            "spins": torch.stack([2 * parameter, -2 * parameter]),
        }
        q, s = fragment_charge_spin_l2(ref, pred)
        (1000 * (q + s)).backward()
        dist.all_reduce(parameter.grad)
        parameter.grad /= 2
        assert parameter.grad.item() == pytest.approx(0.0 if empty_all else 2000.0)
        parameter.grad = None
    dist.destroy_process_group()
    with open(output + str(rank), "w") as f:
        json.dump({"passed": True}, f)


def test_ddp_rank_without_fragments_and_all_unlabeled(tmp_path):
    mp.spawn(
        _ddp_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "rank")),
        nprocs=2,
        join=True,
    )
    assert all(
        json.loads((tmp_path / f"rank{i}").read_text())["passed"] for i in range(2)
    )
