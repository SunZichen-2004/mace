"""Independent numerical checks for grouped interaction supervision."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler

from mace.data.atomic_data import AtomicData
from mace.data.hdf5_dataset import HDF5Dataset
from mace.data.interaction_dataset import (
    InteractionDataset,
    InteractionDataLoader,
    InteractionSample,
    collate_interactions,
)
from mace.data.utils import Configuration, save_configurations_as_HDF5
from mace.modules.loss import (
    InteractionEnergyLoss,
    WeightedEnergyForcesLoss,
    interaction_energy_error,
    interaction_energy_l2,
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


def graph(n, energy=0.0, weight=1.0):
    config = Configuration(
        np.full(n, 2),
        np.arange(n * 3).reshape(n, 3).astype(float),
        dict(
            energy=energy,
            forces=np.zeros((n, 3)),
            total_charge=0.0,
            total_spin=1.0,
            head="Default",
        ),
        dict(energy=1.0, forces=1.0),
        weight=weight,
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
    )
    return AtomicData.from_config(
        config, z_table=AtomicNumberTable([2]), cutoff=5.0, heads=["Default"]
    )


def pair_sample():
    return InteractionSample(
        [graph(4, weight=0), graph(2, weight=0), graph(2, weight=0)], target=2.0
    )


def test_hand_calculated_l2_and_all_three_energy_gradients():
    batch = collate_interactions([pair_sample()])
    energy = torch.tensor([11.0, 3.0, 4.0], requires_grad=True)
    pred = {"energy": energy, "forces": batch.forces.clone()}
    loss = InteractionEnergyLoss(WeightedEnergyForcesLoss(), 100)(batch, pred)
    # E_int=4 eV, target=2 eV, N=4: 100*(2/4)^2 = 25.
    assert loss.item() == pytest.approx(25.0)
    loss.backward()
    torch.testing.assert_close(energy.grad, torch.tensor([25.0, -25.0, -25.0]))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_any_fragment_count_cancels_local_additive_energy(n):
    batch = collate_interactions(
        [
            InteractionSample(
                [graph(2 * n, weight=0)] + [graph(2, weight=0) for _ in range(n)],
                target=0.0,
            )
        ]
    )
    local = torch.tensor(7.0, requires_grad=True)
    interaction = torch.tensor(2.0, requires_grad=True)
    energy = torch.stack(
        [local * (2 * n) + interaction] + [local * 2 for _ in range(n)]
    )
    loss = 100 * interaction_energy_l2(batch, {"energy": energy}, ddp=False)
    loss.backward()
    assert local.grad.item() == pytest.approx(0.0, abs=1e-12)
    assert interaction.grad.item() > 0


def test_auxiliary_graphs_do_not_dilute_original_losses():
    isolated = graph(2, energy=0.0)
    batch = collate_interactions([pair_sample(), InteractionSample([isolated])])
    energy = torch.tensor([11.0, 3.0, 4.0, 2.0], requires_grad=True)
    pred = {"energy": energy, "forces": batch.forces + 1.0}
    parts = InteractionEnergyLoss(WeightedEnergyForcesLoss(), 100).weighted_components(
        batch, pred, ddp=False
    )
    # Ordinary base batch contains the masked 4-atom complex and 2-atom source.
    assert parts["energy"].item() == pytest.approx(0.5)
    assert parts["forces"].item() == pytest.approx(1 / 3)
    assert parts["interaction_energy"].item() == pytest.approx(25.0)


def test_isolated_only_batch_has_finite_zero_interaction_gradient():
    batch = collate_interactions([InteractionSample([graph(2)])])
    energy = torch.tensor([2.0], requires_grad=True)
    loss = interaction_energy_l2(batch, {"energy": energy}, ddp=False)
    loss.backward()
    assert loss.item() == 0 and energy.grad.item() == 0


def test_validation_reports_actual_interaction_l2_and_units():
    batch = collate_interactions([pair_sample(), InteractionSample([graph(2)])])
    pred = {"energy": torch.tensor([11.0, 3.0, 4.0, 0.0]), "forces": batch.forces}
    metric = MACELoss(InteractionEnergyLoss(WeightedEnergyForcesLoss(), 100))
    metric.update(batch, pred)
    _, result = metric.compute()
    assert result["rmse_interaction_energy"] == pytest.approx(2.0)
    assert result["rmse_interaction_energy_per_atom"] == pytest.approx(0.5)
    assert result["loss_interaction_energy"] == pytest.approx(25.0)
    assert result["loss"] == pytest.approx(25.0)


def make_dataset(path):
    configs = []
    for n in (4, 2):
        configs.append(
            Configuration(
                np.full(n, 2),
                np.arange(n * 3).reshape(n, 3).astype(float),
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
        )
    with h5py.File(path, "w") as h:
        save_configurations_as_HDF5(configs, None, h)
        meta = h["config_batch_0/config_0"].create_group("metadata")
        meta["molecule_group_index"] = [0, 0, 1, 1]
        meta["molecule_charges"] = [1, -1]
        meta["molecule_multiplicities"] = [2, 2]
        meta.attrs["interaction_energy"] = 2.0
    return HDF5Dataset(
        str(path), r_max=5.0, z_table=AtomicNumberTable([2]), heads=["Default"]
    )


def test_hdf5_fragment_geometry_charge_spin_and_complete_sampling(tmp_path):
    ds = InteractionDataset(make_dataset(tmp_path / "data.h5"))
    sample = ds[0]
    assert [float(g.total_charge) for g in sample.graphs] == [0.0, 1.0, -1.0]
    assert [float(g.total_spin) for g in sample.graphs] == [1.0, 2.0, 2.0]
    torch.testing.assert_close(
        sample.graphs[1].positions, sample.graphs[0].positions[:2]
    )
    torch.testing.assert_close(
        sample.graphs[2].positions, sample.graphs[0].positions[2:]
    )
    assert float(sample.graphs[0].weight) == 0
    assert (
        float(InteractionDataset(ds.dataset, keep_total_loss=True)[0].graphs[0].weight)
        == 1
    )
    for rank in range(2):
        sampler = DistributedSampler(ds, num_replicas=2, rank=rank, shuffle=False)
        batch = next(iter(InteractionDataLoader(ds, batch_size=1, sampler=sampler)))
        assert batch.num_graphs == (3 if rank == 0 else 1)
        assert batch.interaction_targets.numel() == (1 if rank == 0 else 0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("molecule_group_index", [0, 0, 0, 0]),
        ("molecule_charges", [1, 1]),
        ("molecule_multiplicities", [0, 2]),
    ],
)
def test_malformed_fragments_rejected(tmp_path, key, value):
    path = tmp_path / "data.h5"
    ds = make_dataset(path)
    ds.file.close()
    ds._file = None
    with h5py.File(path, "r+") as h:
        h["config_batch_0/config_0/metadata/" + key][:] = value
    with pytest.raises(ValueError):
        InteractionDataset(ds)[0]


def test_cli_is_opt_in():
    parser = build_default_arg_parser()
    assert not parser.parse_args(["--name=test"]).interaction_energy_training
    args = parser.parse_args(["--name=test", "--interaction_energy_training=True"])
    assert (
        args.interaction_energy_weight == 100
        and not args.interaction_energy_keep_total_loss
    )


def _distributed_worker(rank, rendezvous, output):
    torch.set_default_dtype(torch.float64)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    parameter = torch.tensor(2.0, requires_grad=True)
    for empty_all in (False, True):
        has_pair = rank == 0 and not empty_all
        ref = SimpleNamespace(
            interaction_indices=torch.tensor(
                [0, 1, 2] if has_pair else [], dtype=torch.long
            ),
            interaction_coefficients=torch.tensor(
                [1.0, -1.0, -1.0] if has_pair else []
            ),
            interaction_sample_ids=torch.tensor(
                [0, 0, 0] if has_pair else [], dtype=torch.long
            ),
            interaction_targets=torch.tensor([0.0] if has_pair else []),
            interaction_num_atoms=torch.tensor([4.0] if has_pair else []),
            interaction_weights=torch.tensor([1.0] if has_pair else []),
        )
        energy = (
            torch.stack([parameter, parameter * 0, parameter * 0])
            if has_pair
            else parameter.reshape(1)
        )
        loss = 100 * interaction_energy_l2(ref, {"energy": energy})
        loss.backward()
        dist.all_reduce(parameter.grad)
        parameter.grad /= 2
        expected = 0.0 if empty_all else 25.0
        assert parameter.grad.item() == pytest.approx(expected)
        parameter.grad = None
    dist.destroy_process_group()
    with open(output + str(rank), "w") as f:
        json.dump({"passed": True}, f)


def test_ddp_handles_one_empty_rank_and_all_empty(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "rank")),
        nprocs=2,
        join=True,
    )
    assert all(
        json.loads((tmp_path / f"rank{i}").read_text())["passed"] for i in range(2)
    )
