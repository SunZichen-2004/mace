import math

import pytest
import torch

from mace.modules.mm_embedding import MM_potential_features
from tests.extensions.polar.test_polar_models import (
    _build_minimal_batch,
    _build_minimal_model,
)


def system(periodic=False):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(4)
    model = _build_minimal_model("cpu", torch.float64, realspace_method="analytical")
    data = _build_minimal_batch("cpu", torch.float64)
    data["pbc"][:] = periodic
    data["volume"] = torch.linalg.det(data["cell"].view(-1, 3, 3))
    data["total_spin"][:] = 1
    mm = torch.tensor([[3.2, 1.1, -0.7], [-2.5, -1.3, 1.5]])
    charges = torch.tensor([0.4, -0.4])
    return model, data, mm, charges


@pytest.mark.parametrize("periodic", [False, True])
def test_mm_forces_are_energy_gradients(periodic):
    model, data, mm, charges = system(periodic)
    result = model(dict(data), MM_coordinate=mm, MM_charges=charges)
    eps = 1e-5
    for kind, index in (("positions", 0), ("MM_coordinate", 1)):
        values = []
        for sign in (1, -1):
            current = {k: v.clone() for k, v in data.items()}
            external = mm.clone()
            (current["positions"] if kind == "positions" else external)[index, 0] += (
                sign * eps
            )
            values.append(
                model(
                    current,
                    compute_force=False,
                    MM_coordinate=external,
                    MM_charges=charges,
                )["energy"]
                .sum()
                .detach()
            )
        expected = -(values[0] - values[1]) / (2 * eps)
        force = result["forces" if kind == "positions" else "MM_forces"][index, 0]
        torch.testing.assert_close(force, expected, atol=1e-6, rtol=1e-4)


def test_zero_mm_and_force_training():
    model, data, mm, charges = system()
    vacuum = model(dict(data))
    zero = model(dict(data), MM_coordinate=mm, MM_charges=torch.zeros_like(charges))
    torch.testing.assert_close(zero["energy"], vacuum["energy"])
    torch.testing.assert_close(zero["forces"], vacuum["forces"])
    result = model(dict(data), training=True, MM_coordinate=mm, MM_charges=charges)
    loss = (
        result["energy"].abs().mean()
        + result["forces"].square().mean()
        + result["MM_forces"].square().mean()
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )

    torch.testing.assert_close(
        result["forces"].sum(0) + result["MM_forces"].sum(0),
        torch.zeros(3),
        atol=1e-9,
        rtol=0,
    )


def test_separate_force_losses():
    from mace.modules.loss import WeightedEnergyForcesL1L2Loss
    from mace.tools.torch_geometric.batch import Batch

    ref = Batch(
        ptr=torch.tensor([0, 2]),
        weight=torch.ones(1),
        energy_weight=torch.ones(1),
        forces_weight=torch.ones(1),
        energy=torch.tensor([2.0]),
        forces=torch.ones(2, 3),
        MM_forces=torch.full((5, 3), 2.0),
    )
    pred = {
        "energy": torch.zeros(1),
        "forces": torch.zeros(2, 3),
        "MM_forces": torch.zeros(5, 3),
    }
    loss = WeightedEnergyForcesL1L2Loss(1, 2, 3)(ref, pred)
    torch.testing.assert_close(loss, torch.tensor(1.0 + 2.0 + 12.0))


def test_ragged_mm_batch_matches_separate_graphs():
    model, data, mm, charges = system()
    positions = torch.cat((data["positions"], data["positions"][:1] + 10))
    batch = torch.tensor([0] * len(data["positions"]) + [1])
    external = torch.cat((mm[:1] + 10, mm)).requires_grad_()
    MM_batch = torch.tensor([1, 0, 0])  # Deliberately not grouped by graph.
    q = torch.cat((charges[:1], charges))
    cell = data["cell"].view(-1, 3, 3).repeat(2, 1, 1)

    def evaluate(x, b, y, charge, mb, c):
        return MM_potential_features(
            model,
            x,
            b,
            y,
            charge,
            mb,
            c,
            torch.zeros(len(c), 3, dtype=torch.bool),
            None,
            None,
            None,
            None,
        )

    together = evaluate(positions, batch, external, q, MM_batch, cell)
    separate = [
        evaluate(
            positions[batch == i],
            torch.zeros_like(batch[batch == i]),
            external[MM_batch == i],
            q[MM_batch == i],
            torch.zeros_like(MM_batch[MM_batch == i]),
            cell[i : i + 1],
        )
        for i in range(2)
    ]
    for index in range(2):
        expected = torch.cat([result[index] for result in separate])
        torch.testing.assert_close(together[index], expected)
        actual_grad = torch.autograd.grad(
            together[index].sum(), external, retain_graph=True
        )[0]
        expected_grad = torch.autograd.grad(
            expected.sum(), external, retain_graph=True
        )[0]
        torch.testing.assert_close(actual_grad, expected_grad)


def test_legacy_config_preserves_absent_tp_mix():
    from mace.tools.scripts_utils import extract_config_mace_model

    model, _, _, _ = system()
    model._fixedpoint_update_config.pop("use_tp_mix", None)
    for update in model.field_dependent_charges_maps:
        if hasattr(update, "tp_mix"):
            del update.tp_mix
    config = extract_config_mace_model(model)
    assert config["fixedpoint_update_config"]["use_tp_mix"] is False
    rebuilt = type(model)(**config)
    assert all(
        not hasattr(update, "tp_mix") for update in rebuilt.field_dependent_charges_maps
    )


def test_point_charge_cross_energy_and_pme():
    from graph_longrange.kspace import compute_k_vectors_flat
    from graph_longrange.utils import FIELD_CONSTANT

    model, data, mm, charges = system()
    cell = data["cell"].view(-1, 3, 3)
    k, norm, batch, zero = compute_k_vectors_flat(
        model.kspace_cutoff, cell, data["rcell"].view(-1, 3, 3)
    )
    _, potential = MM_potential_features(
        model,
        data["positions"],
        data["batch"],
        mm,
        charges,
        torch.zeros(2, dtype=torch.long),
        cell,
        data["pbc"],
        k,
        norm,
        batch,
        zero,
    )
    distance = (data["positions"][0] - mm).norm(dim=-1)
    sigma = model.coulomb_energy.density_smearing_width
    expected = (
        FIELD_CONSTANT
        / (4 * math.pi)
        * (charges * torch.erf(distance / (math.sqrt(2) * sigma)) / distance).sum()
    )
    torch.testing.assert_close(potential[0, 0], expected)
    pytest.importorskip("torchpme")
    data["pbc"][:] = True
    direct = model(dict(data), MM_coordinate=mm, MM_charges=charges)
    pme = model(dict(data), MM_coordinate=mm, MM_charges=charges, MM_method="PME")
    for name in ("energy", "forces", "MM_forces"):
        torch.testing.assert_close(pme[name], direct[name], atol=2e-5, rtol=2e-4)
    trained = model(
        dict(data), training=True, MM_coordinate=mm, MM_charges=charges, MM_method="PME"
    )
    (
        trained["forces"].square().mean() + trained["MM_forces"].square().mean()
    ).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_cross_energy_matches_existing_coulomb_block(periodic):
    from graph_longrange.kspace import compute_k_vectors_flat

    model, data, mm, charges = system(periodic)
    cell = data["cell"].view(-1, 3, 3)
    k, norm, kb, zero = compute_k_vectors_flat(
        model.kspace_cutoff, cell, data["rcell"].view(-1, 3, 3)
    )
    mm_batch = torch.zeros(2, dtype=torch.long)
    _, conjugate = MM_potential_features(
        model,
        data["positions"],
        data["batch"],
        mm,
        charges,
        mm_batch,
        cell,
        data["pbc"],
        k,
        norm,
        kb,
        zero,
        MM_charge_width=model.coulomb_energy.density_smearing_width,
    )
    qm_density = torch.tensor([[0.2, 0.03, -0.02, 0.01], [-0.2, 0.01, 0.02, -0.04]])
    mm_density = torch.zeros_like(qm_density)
    mm_density[:, 0] = charges

    def energy(coefficients, coordinates):
        return model.coulomb_energy(
            k_vectors=k,
            k_norm2=norm,
            k_vector_batch=kb,
            k0_mask=zero,
            source_feats=coefficients,
            node_positions=coordinates,
            batch=torch.zeros(len(coordinates), dtype=torch.long),
            volume=data["volume"],
            pbc=data["pbc"],
        ).sum()

    reference = (
        energy(torch.cat((qm_density, mm_density)), torch.cat((data["positions"], mm)))
        - energy(qm_density, data["positions"])
        - energy(mm_density, mm)
    )
    torch.testing.assert_close(
        (qm_density * conjugate).sum(), reference, atol=1e-9, rtol=1e-8
    )


def test_bms_variable_mm_counts_and_missing_gradients(tmp_path):
    import gzip
    import hashlib
    import shutil
    import h5py
    import numpy as np
    from mace.data.bms_dataset import BMSBatches, BMSDataset, collate_bms

    source = tmp_path / "sample.h5"
    with h5py.File(source, "w") as handle:
        for name, count, missing in (
            ("first", 2, False),
            ("second", 3, False),
            ("missing", 4, True),
        ):
            while (
                int(hashlib.sha256(f"1:TEST:{name}".encode()).hexdigest()[:8], 16) % 20
                == 0
            ):
                name += "x"
            group = handle.create_group(f"{name}/0")
            values = dict(
                Z=[1, 8],
                coords_qm=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                charge=0,
                e_qmmm=-100.0,
                grad_qmmm=np.ones((2, 3)),
                coords_mm=np.ones((count, 3)) * 3,
                charges_mm=np.ones(count),
                grad_mm=np.empty(0) if missing else np.ones((count, 3)),
            )
            for key, value in values.items():
                group[key] = value
    with source.open("rb") as src, gzip.open(tmp_path / "TEST_0.h5.gz", "wb") as dest:
        shutil.copyfileobj(src, dest)
    with pytest.warns(UserWarning, match="Skipping inconsistent BMS25 record"):
        graphs = list(BMSDataset(tmp_path, [1, 8], 4.0))
    assert len(graphs) == 2
    batch = collate_bms(graphs)
    assert batch.MM_coordinate.shape == batch.MM_forces.shape == (5, 3)
    assert batch.MM_batch.shape == (5,)
    assert torch.all(batch.MM_forces < 0)
    batches = list(BMSBatches(graphs, batch_size=320, max_pairs=10))
    assert [len(items) for items in batches] == [1, 1]
    assert [item.sample_id for items in batches for item in items] == [
        graph.sample_id for graph in graphs
    ]
    # Node-local, already unpacked shards follow exactly the same split.
    with pytest.warns(UserWarning, match="Skipping inconsistent BMS25 record"):
        unpacked = list(BMSDataset(tmp_path / "uncompressed", [1, 8], 4.0))
    assert {g.sample_id.split(":", 1)[1] for g in unpacked} == {
        g.sample_id.split(":", 1)[1] for g in graphs
    }


def test_recompute_final_field_with_mm_force_training():
    model, data, mm, charges = system()
    model.fukui_regularisation = 0.1
    model.recompute_field_after_scf = True
    result = model(dict(data), training=True, MM_coordinate=mm, MM_charges=charges)
    loss = (
        result["energy"].square().mean()
        + result["forces"].square().mean()
        + result["MM_forces"].square().mean()
    )
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    eps = 1e-5
    energies = []
    for sign in [1, -1]:
        external = mm.clone()
        external[0, 0] += sign * eps
        energies.append(
            model(
                dict(data),
                compute_force=False,
                MM_coordinate=external,
                MM_charges=charges,
            )["energy"]
            .sum()
            .detach()
        )
    torch.testing.assert_close(
        result["MM_forces"][0, 0].detach(),
        -(energies[0] - energies[1]) / (2 * eps),
        atol=1e-6,
        rtol=1e-4,
    )
