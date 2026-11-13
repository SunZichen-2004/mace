import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mace.modules.polar2_loss import POLAR2Loss
from tests.extensions.polar.test_polar_block_configs import build
from tests.extensions.polar.test_polar_models import _build_model_batch


def test_dual_response_matches_field_finite_difference_and_backpropagates():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(31)
    model = build(
        "{'type':'PairwiseGatedMultipoleUpdate'}",
        "{'type':'OneBodyMLPFieldReadout'}",
        extra_argv=[
            "--realspace_method",
            "analytical",
            "--potential_feature_mode",
            "charge_spin",
        ],
    )
    model.compute_dipole_from_electric_field = True
    model.compute_spin_multipoles = True
    model.recompute_field_after_scf = True
    batch = _build_model_batch(model)
    out = model(
        copy.deepcopy(batch),
        training=True,
        compute_force=False,
        compute_polar2_response=True,
    )
    for axis in range(3):
        plus = copy.deepcopy(batch)
        minus = copy.deepcopy(batch)
        plus["external_field"][:, axis] += 1e-5
        minus["external_field"][:, axis] -= 1e-5
        op = model(plus, training=True, compute_force=False)
        om = model(minus, training=True, compute_force=False)
        for name, mu in [
            ("polar2_alpha_field", "total_dipole_from_electric_field"),
            ("polar2_alpha_multipole", "dipole"),
        ]:
            fd = -(op[mu] - om[mu]) / 2e-5
            torch.testing.assert_close(out[name][:, :, axis], fd, atol=1e-6, rtol=1e-4)
    loss = (
        out["polar2_alpha_field"].square().sum()
        + out["polar2_alpha_multipole"].square().sum()
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters()
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    ordinary = model(copy.deepcopy(batch), compute_force=False)
    assert ordinary["polar2_alpha_field"] is None


def test_masks_dual_branches_and_normalization():
    torch.set_default_dtype(torch.float64)
    b = SimpleNamespace(
        ptr=torch.tensor([0, 2, 4]),
        batch=torch.tensor([0, 0, 1, 1]),
        MM_batch=torch.empty(0, dtype=torch.long),
        weight=torch.ones(2),
    )
    p = {}
    shapes = {
        "energy": (2,),
        "forces": (4, 3),
        "MM_forces": (0, 3),
        "dipole": (2, 3),
        "quadrupole": (2, 3, 3),
        "spin_dipole": (2, 3),
        "spin_quadrupole": (2, 3, 3),
        "polarizability": (2, 3, 3),
    }
    for k, shape in shapes.items():
        setattr(b, k, torch.zeros(shape))
        setattr(b, k + "_weight", torch.zeros(2))
        p[k] = None
    b.energy_weight = torch.ones(2)
    p["energy"] = torch.tensor([2.0, -4.0], requires_grad=True)
    b.dipole_weight = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    b.dipole[1] = float("nan")
    p["dipole"] = torch.ones(2, 3, requires_grad=True) * 2
    p["total_dipole_from_electric_field"] = torch.ones(2, 3, requires_grad=True) * 4
    loss = POLAR2Loss(energy_weight=10, dipole_weight=1)
    parts = loss.weighted_components(b, p)
    torch.testing.assert_close(parts["energy"], torch.tensor(15.0))
    torch.testing.assert_close(parts["dipole"], torch.tensor(2.5))
    assert all(torch.isfinite(v) for v in parts.values())
    sum(parts.values()).backward()
    assert torch.isfinite(p["energy"].grad).all()
    p["total_dipole_from_electric_field"] = None
    with pytest.raises(ValueError, match="Missing active"):
        loss(b, p)


def test_polarizability_storage_and_model_units(tmp_path):
    import json
    import h5py
    from mace.data.polar2_dataset import UnifiedDataset

    source = tmp_path / "alpha.hdf5"
    alpha = np.array([[2.0, 0.2, 0.1], [0.2, 3.0, -0.3], [0.1, -0.3, 4.0]])
    with h5py.File(source, "w") as f:
        f.attrs["polarizability_unit"] = "e*Angstrom^2/volt"
        g = f.create_group("0")
        g.attrs.update(
            total_charge=0,
            total_spin=1,
            source_group="test",
            original_spice_group="test",
        )
        for k, v in dict(
            atomic_numbers=[1, 1],
            positions=[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]],
            energy=[0.0],
            forces=np.zeros((1, 2, 3)),
            dipole=np.zeros((1, 3)),
            polarizability=alpha[None],
            source_row=[0],
            original_conformer=[0],
        ).items():
            g[k] = v
    np.save(tmp_path / "index.npy", np.array([[0, 0]]))
    spec = dict(
        heads=["omol"],
        datasets={
            "spice_alpha": dict(
                head="omol",
                splits=["all"],
                kind="grouped_hdf5",
                shards=[
                    dict(
                        source=str(source),
                        index=str(tmp_path / "index.npy"),
                        counts={"all": 1},
                    )
                ],
            )
        },
    )
    (tmp_path / "catalog.json").write_text(json.dumps(spec))
    dataset = UnifiedDataset("spice_alpha", "all", graph=False, root=tmp_path)
    record = dataset[0]
    dataset.close()
    np.testing.assert_allclose(
        record["properties"]["polarizability"], alpha * 14.3996454784255
    )
    assert (
        record["property_weights"]["energy"]
        == record["property_weights"]["forces"]
        == 0
    )
    assert record["property_weights"]["polarizability"] == 1
    from mace.data.polar2_dataset import collate_graphs

    dataset = UnifiedDataset(
        "spice_alpha", "all", graph=True, root=tmp_path, atomic_numbers=[1]
    )
    batch = collate_graphs([dataset[0]])
    dataset.close()
    pred = {
        "energy": torch.zeros(1, requires_grad=True),
        "polar2_alpha_field": torch.tensor(alpha)[None],
        "polar2_alpha_multipole": torch.tensor(alpha)[None],
    }
    parts = POLAR2Loss().weighted_components(batch, pred, ddp=False)
    torch.testing.assert_close(
        parts["polarizability"], torch.zeros_like(parts["polarizability"])
    )


def test_positive_physical_response_sign():
    from mace.modules.polar2_response import physical_polarizability

    field = torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float64, requires_grad=True)
    # Stable anisotropic response, including physically valid negative off-diagonals.
    alpha = torch.tensor(
        [[2.0, -0.4, 0.1], [-0.4, 3.0, 0.2], [0.1, 0.2, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mu0 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    energy = (mu0 * field).sum() - 0.5 * torch.einsum("bi,ij,bj->", field, alpha, field)
    field_mu = torch.autograd.grad(energy, field, create_graph=True)[0]
    multipole_mu = mu0 - field @ alpha.T
    for mu in [field_mu, multipole_mu]:
        actual = physical_polarizability(mu, field, True)
        torch.testing.assert_close(actual, alpha[None])
        assert torch.linalg.eigvalsh(actual).min() > 0
        assert actual[0, 0, 1] < 0
        (actual.square().sum()).backward(retain_graph=True)
    assert torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum() > 0


def test_source_sampler_stays_synchronized_and_disjoint():
    from mace.data.polar2_dataset import Polar2Dataset, Polar2BatchSampler
    from torch.utils.data import ConcatDataset

    class SmallDataset(Polar2Dataset):
        def __init__(self, n, b, p):
            self.n = n
            self.batch_size = b
            self.sample_probability = p

        def __len__(self):
            return self.n

    d = ConcatDataset([SmallDataset(100, 3, 0.7), SmallDataset(200, 5, 0.3)])
    samplers = [
        Polar2BatchSampler(d, steps=30, rank=r, world_size=4, seed=17) for r in range(4)
    ]
    batches = [list(s) for s in samplers]
    for step in range(30):
        source = [int(batches[r][step][0] >= 100) for r in range(4)]
        assert len(set(source)) == 1
        flat = [i for r in range(4) for i in batches[r][step]]
        assert len(flat) == len(set(flat))
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != batches[0]
    vd = ConcatDataset([SmallDataset(23, 3, 0.7), SmallDataset(41, 3, 0.3)])
    vb = [
        list(Polar2BatchSampler(vd, rank=r, world_size=4, validation=True))
        for r in range(4)
    ]
    assert len(set(len(x) for x in vb)) == 1
    flat = [i for rank in vb for batch in rank for i in batch]
    assert len(flat) == len(set(flat)) == len(vd)
    assert set(flat) == set(range(len(vd)))


def test_cache_resolution_fails_closed(tmp_path):
    import json
    from mace.data.polar2_dataset import UnifiedDataset

    reader = UnifiedDataset.__new__(UnifiedDataset)
    reader.node_cache = {"source": {"path": str(tmp_path / "cached"), "verified": True}}
    assert reader._cached_path("source") == str(tmp_path / "cached")
    with pytest.raises(ValueError, match="Missing verified"):
        reader._cached_path("missing")
    reader.node_cache["source"]["verified"] = False
    with pytest.raises(ValueError, match="Missing verified"):
        reader._cached_path("source")
