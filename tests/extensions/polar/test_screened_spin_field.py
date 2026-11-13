"""Screened Gaussian convolution, periodic zero mode, and training derivatives."""

import io
import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from scipy.integrate import quad

pytest.importorskip("graph_longrange")
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.gto_utils import get_Cl_sigma
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.utils import FIELD_CONSTANT

from mace.modules.extensions import PolarMACE
from mace.modules.screened_field import ScreenedGTOElectrostaticFeatures
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.torch_tools import default_dtype
from tests.extensions.polar.test_polar_block_configs import build
from tests.extensions.polar.test_polar_models import _build_model_batch

pytestmark = pytest.mark.polar


@pytest.fixture(autouse=True)
def double_precision():
    with default_dtype(torch.float64):
        torch.manual_seed(17)
        yield


def descriptor(screened=True, **overrides):
    config = dict(
        density_max_l=1,
        density_smearing_width=1.5,
        feature_max_l=1,
        feature_smearing_widths=[1.5, 3.0],
        include_self_interaction=False,
        kspace_cutoff=5.0,
        realspace_method="analytical",
        pbc_handling="auto",
    )
    config.update(overrides)
    cls = ScreenedGTOElectrostaticFeatures if screened else GTOElectrostaticFeatures
    return cls(**config)


def geometry(positions, batch=None, cell=None):
    if batch is None:
        batch = torch.zeros(len(positions), dtype=torch.long)
    if cell is None:
        k = positions.new_empty((0, 3))
        k2 = positions.new_empty(0)
        kb = batch.new_empty(0)
        k0 = positions.new_empty(0)
        pbc = torch.zeros((int(batch.max()) + 1, 3), dtype=torch.bool)
        volume = positions.new_ones(len(pbc))
    else:
        cell = cell.reshape(-1, 3, 3)
        rcell = 2 * torch.pi * torch.linalg.inv(cell.transpose(-1, -2))
        k, k2, kb, k0 = compute_k_vectors_flat(5.0, cell, rcell)
        pbc = torch.ones((len(cell), 3), dtype=torch.bool)
        volume = torch.linalg.det(cell)
    return dict(
        k_vectors=k,
        k_norm2=k2,
        k_vector_batch=kb,
        k0_mask=k0,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )


def evaluate(block, sources, positions, **kwargs):
    cache = block.precompute_geometry(**geometry(positions, **kwargs))
    return block.forward_dynamic(source_feats=sources, cache=cache)


@pytest.mark.parametrize("distance", [0.0, 0.01, 1.0, 6.0, 12.0])
def test_scalar_convolution_matches_independent_fourier_quadrature(distance):
    block = descriptor()
    sources = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    positions = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    result = evaluate(block, sources, positions)[1, :2]
    expected = []
    for width in (1.5, 3.0):
        variance = 1.5**2 + width**2

        def integrand(k):
            sinc = 1.0 if k * distance == 0 else math.sin(k * distance) / (k * distance)
            return math.exp(-variance * k * k / 2) * (-math.expm1(-k * k)) * sinc

        radial = 2 / math.pi * quad(integrand, 0, math.inf, epsabs=1e-13)[0]
        receiver_norm = get_Cl_sigma(0, width, "receiver") / get_Cl_sigma(
            0, width, "multipoles"
        )
        expected.append(radial * FIELD_CONSTANT / (4 * math.pi) * receiver_norm)
    torch.testing.assert_close(result, torch.tensor(expected), atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("distance", [0.0, 1.7])
def test_multipole_first_and_second_derivatives(distance):
    block = descriptor()
    sources = torch.randn(2, 4, requires_grad=True)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.1 * distance, 0.0]], requires_grad=True
    )
    fn = lambda q, r: evaluate(block, q, r)
    assert torch.autograd.gradcheck(fn, (sources, positions), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (sources, positions), fast_mode=True)


@pytest.mark.parametrize("include_self", [False, True])
def test_periodic_multipoles_match_explicit_realspace_image_sum(include_self):
    periodic = descriptor(include_self_interaction=include_self)
    isolated = descriptor(include_self_interaction=include_self)
    moment = torch.tensor([[1.0, 0.2, -0.1, 0.3]])
    actual = evaluate(periodic, moment, torch.zeros(1, 3), cell=12 * torch.eye(3))[0]
    positions = 12 * torch.tensor(list(itertools.product(range(-2, 3), repeat=3)))
    reference = evaluate(isolated, moment.expand(len(positions), -1), positions)
    center = (positions == 0).all(dim=-1).nonzero().item()
    torch.testing.assert_close(actual, reference[center], atol=2e-8, rtol=2e-8)


def test_periodic_kernel_zero_mode_is_finite_and_nonzero():
    block = descriptor()
    cache = block.precompute_geometry(
        **geometry(torch.zeros(1, 3), cell=12 * torch.eye(3))
    )
    zero = cache["k0_mask"] > 0
    assert zero.sum() == 1
    torch.testing.assert_close(cache["k_factor_coulomb"][zero], torch.ones(1))


@pytest.mark.parametrize("mode", ["alpha_beta", "charge_spin"])
def test_field_channels_external_field_and_spin_flip(mode):
    charge = descriptor(screened=False)
    spin = descriptor()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    geo = geometry(positions)
    charge_cache, spin_cache = charge.precompute_geometry(
        **geo
    ), spin.precompute_geometry(**geo)
    density = torch.randn(2, 2, 4)
    norms = torch.arange(1.0, 9.0)
    external = torch.randn(2, 8)
    model = SimpleNamespace(
        potential_feature_mode=mode,
        electric_potential_descriptor=charge,
        spin_potential_descriptor=spin,
        field_feature_norms=norms,
    )
    fn = lambda d, e: PolarMACE._compute_potential_features(
        model, d, charge_cache, spin_cache, e
    )
    result = fn(density, external)
    if mode == "alpha_beta":
        expected = torch.cat(
            [
                (
                    charge.forward_dynamic(
                        source_feats=density[:, s], cache=charge_cache
                    )
                    + external
                )
                / norms
                for s in (0, 1)
            ],
            dim=-1,
        )
        torch.testing.assert_close(result, expected)
    else:
        q, s = result.chunk(2, dim=-1)
        expected_q = (
            evaluate(charge, density.sum(1), positions) / 2 + external
        ) / norms
        expected_s = (
            evaluate(spin, density[:, 0] - density[:, 1], positions) / 2 / norms
        )
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(s, expected_s)
        flipped_q, flipped_s = fn(density.flip(1), external).chunk(2, dim=-1)
        torch.testing.assert_close(flipped_q, q)
        torch.testing.assert_close(flipped_s, -s)
        torch.testing.assert_close(fn(density, torch.zeros_like(external))[:, 8:], s)


def test_charge_spin_model_force_training_and_serialization():
    model = build(
        fixedpoint="{'type':'PairwiseGatedMultipoleUpdate'}",
        readout="{'type':'OneBodyNoFeildTPMLPFieldReadout'}",
        extra_argv=[
            "--potential_feature_mode",
            "charge_spin",
            "--spin_field_alpha",
            "0.5",
            "--realspace_method",
            "analytical",
            "--num_recursion_steps",
            "2",
        ],
    )
    batch = _build_model_batch(model)
    reference = model(batch, training=True)
    loss = reference["energy"].square().sum() + reference["forces"].square().sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.recompute_field_after_scf = True
    refreshed = model(_build_model_batch(model))
    for key in ("energy", "forces", "density_coefficients"):
        torch.testing.assert_close(refreshed[key], reference[key])
    config = extract_config_mace_model(model)
    assert config["potential_feature_mode"] == "charge_spin"
    assert config["spin_field_alpha"] == 0.5
    restored = PolarMACE(**config)
    restored.load_state_dict(model.state_dict(), strict=True)
    buffer = io.BytesIO()
    torch.save(restored, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    restored.set_realspace_method("analytical")
    torch.testing.assert_close(
        restored(_build_model_batch(restored))["energy"], reference["energy"]
    )


@pytest.mark.parametrize("alpha", [0.0, -0.5, float("nan"), float("inf")])
def test_invalid_screening_is_rejected(alpha):
    with pytest.raises(ValueError, match="finite and positive"):
        descriptor(screening_alpha=alpha)
