"""Physical and training checks for the pure edge cluster expansion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, tools
from mace.modules.mece import MECE, CylindricalBasis
from mace.tools.scripts_utils import get_params_options
from mace.tools.torch_geometric.batch import Batch


@pytest.fixture(autouse=True)
def precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    yield
    torch.set_default_dtype(previous)


def model(union=False, **kwargs):
    config = dict(
        r_max=2.5,
        num_bessel=3,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_interactions=3,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=4.0,
        atomic_numbers=[1, 8],
        correlation=3,
        num_density_channels=3,
        element_embedding_dim=3,
        bond_neighbor_union=union,
        bond_neighbor_middle=not union,
    )
    config.update(kwargs)
    return MECE(**config)


POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [1.2, 0.2, -0.1], [0.3, 1.1, 0.7], [-0.7, 0.4, 1.3]]
)


def batch(positions=POSITIONS, periodic=False, cell=None, numbers=None):
    positions = np.asarray(positions)
    if cell is None:
        cell = np.eye(3) * 4.5
    config = data.Configuration(
        atomic_numbers=np.asarray(numbers if numbers is not None else [1, 8, 1, 8]),
        positions=positions,
        properties={},
        property_weights={},
        cell=np.asarray(cell),
        pbc=(periodic,) * 3,
    )
    atomic = data.AtomicData.from_config(
        config, z_table=tools.AtomicNumberTable([1, 8]), cutoff=2.5
    )
    return Batch.from_data_list([atomic]).to_dict()


def test_phase_has_unit_modulus_and_no_rho_prefactor():
    basis = CylindricalBasis(2.5, 3, 2)
    features, phase = basis(torch.tensor([[0.3, 0.4, 0.2], [0.9, 1.2, 0.2]]))
    torch.testing.assert_close(phase.abs(), torch.ones_like(phase.real))
    torch.testing.assert_close(phase[0], phase[1])
    assert features.shape == (2, 7)


def test_independent_signed_radial_weights_and_exact_neighbor_sum():
    net = model()
    expansion = net.edge_cluster_expansion
    with torch.no_grad():
        for param in expansion.env_radial_mlp.parameters():
            param.zero_()
        expansion.env_radial_mlp[-1].bias.copy_(
            torch.arange(1, 16, dtype=torch.float64)
        )
    geometry = dict(
        env_features=torch.zeros(2, 7),
        env_species=torch.zeros(2, 9),
        env_cutoff=torch.tensor([[0.5], [0.25]]),
        env_phase=torch.ones(2, 5, dtype=torch.complex128),
        bond_index=torch.tensor([0, 0]),
        number_of_edges=2,
    )
    g = expansion._environment_radial(geometry)
    assert not torch.allclose(g[:, :, 0], g[:, :, -1].conj())
    narrow, _ = expansion(geometry)
    torch.testing.assert_close(narrow[0], g.sum(0))
    torch.testing.assert_close(narrow[1], torch.zeros_like(narrow[1]))


@pytest.mark.parametrize("union", [False, True])
def test_neighbor_modes_and_cutoff(union):
    net = model(union=union)
    positions = torch.tensor(
        [[-1.0, 0, 0], [1.0, 0, 0], [-3.1, 0.2, 0.1], [0, 2.3, 0.2]]
    )
    edges = torch.tensor([[1, 0], [0, 1]])
    vectors = positions[edges[1]] - positions[edges[0]]
    geometry = net._geometry(
        positions,
        edges,
        vectors,
        vectors.norm(dim=-1, keepdim=True),
        torch.zeros(4, dtype=torch.long),
        torch.eye(2)[torch.tensor([0, 1, 0, 1])],
    )
    atoms = geometry["bond_neighbor_atom_index"][geometry["bond_index"] == 0]
    assert set(atoms.tolist()) == ({2} if union else {3})
    expected_distance = (
        (positions[2] - positions[0]).norm() if union else positions[3].norm()
    )
    torch.testing.assert_close(
        geometry["env_cutoff"], net.cutoff(expected_distance.reshape(1, 1)).expand(2, 1)
    )


@pytest.mark.parametrize("union", [False, True])
def test_energy_invariant_and_forces_equivariant(union):
    net = model(union=union)
    output = net(batch(), training=True)
    rotation = o3.rand_matrix().numpy()
    rotated = net(batch(POSITIONS @ rotation.T), training=True)
    reflected = net(batch(POSITIONS * np.array([1.0, -1.0, 1.0])), training=True)
    translated = net(batch(POSITIONS + np.array([3.0, -2.0, 1.0])), compute_force=False)
    torch.testing.assert_close(
        output["energy"], rotated["energy"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        output["energy"], reflected["energy"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        output["energy"], translated["energy"], atol=1e-10, rtol=1e-9
    )
    torch.testing.assert_close(
        rotated["forces"],
        output["forces"] @ torch.tensor(rotation.T),
        atol=1e-9,
        rtol=1e-8,
    )
    torch.testing.assert_close(output["node_energy"].sum(), output["energy"].sum())
    assert output["bond_feature"].shape[1:] == (12, 5)
    assert not hasattr(net, "node_embedding") and not hasattr(net, "interactions")


@pytest.mark.parametrize("union", [False, True])
def test_force_finite_difference_and_force_training(union):
    net = model(union=union)
    output = net(batch(), training=True)
    epsilon = 1e-5
    plus = POSITIONS.copy()
    plus[2, 1] += epsilon
    minus = POSITIONS.copy()
    minus[2, 1] -= epsilon
    numerical = -(
        net(batch(plus), compute_force=False)["energy"]
        - net(batch(minus), compute_force=False)["energy"]
    ) / (2 * epsilon)
    torch.testing.assert_close(
        output["forces"][2, 1], numerical.squeeze(), atol=1e-7, rtol=1e-5
    )
    loss = output["energy"].square().sum() + output["forces"].square().sum()
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert net.edge_cluster_expansion.env_radial_mlp[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("union", [False, True])
def test_periodic_image_wrapping_and_stress(union):
    net = model(union=union)
    positions = POSITIONS + np.array([3.6, 0.0, 0.0])
    wrapped = positions.copy()
    wrapped[:, 0] %= 4.5
    output = net(batch(positions, periodic=True), training=True, compute_stress=True)
    other = net(batch(wrapped, periodic=True), training=True, compute_stress=True)
    torch.testing.assert_close(output["energy"], other["energy"], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(output["forces"], other["forces"], atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(output["stress"], other["stress"], atol=1e-9, rtol=1e-8)
    epsilon = 1e-5
    strain = np.eye(3)
    strain[0, 0] += epsilon
    plus = net(
        batch(positions @ strain, periodic=True, cell=np.eye(3) * 4.5 @ strain),
        compute_force=False,
    )["energy"]
    strain[0, 0] -= 2 * epsilon
    minus = net(
        batch(positions @ strain, periodic=True, cell=np.eye(3) * 4.5 @ strain),
        compute_force=False,
    )["energy"]
    numerical = (plus - minus) / (2 * epsilon * 4.5**3)
    torch.testing.assert_close(
        output["stress"][0, 0, 0], numerical.squeeze(), atol=1e-7, rtol=1e-5
    )


def test_empty_graph_and_pair_without_environment():
    net = model()
    single = net(batch([[0.0, 0.0, 0.0]], numbers=[1]), training=True)
    torch.testing.assert_close(single["forces"], torch.zeros(1, 3))
    pair = net(batch([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], numbers=[1, 8]), training=True)
    assert pair["bond_feature_narrow"].shape == (2, 3, 5)
    assert torch.count_nonzero(pair["bond_feature_narrow"]) == 0
    assert torch.isfinite(pair["energy"]).all() and torch.isfinite(pair["forces"]).all()


def test_optimizer_covers_every_parameter():
    net = model()
    args = SimpleNamespace(weight_decay=0.01, lr=0.001, amsgrad=False, beta=0.9)
    options = get_params_options(args, net)
    registered = [p for group in options["params"] for p in group["params"]]
    assert {id(p) for p in registered} == {
        id(p) for p in net.parameters() if p.requires_grad
    }
    torch.optim.AdamW(**options)


def test_neighbor_flags_and_cli_factory():
    from mace.tools.model_script_utils import _build_model
    from mace.tools.scripts_utils import extract_config_mace_model

    parser = tools.build_default_arg_parser()
    args = parser.parse_args(
        ["--name=ece", "--model=MECE", "--bond_neighbor_union=True"]
    )
    args.std = 1.0
    args.mean = 0.0
    net = model()
    config = dict(
        r_max=2.5,
        num_bessel=3,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=4.0,
        atomic_numbers=[1, 8],
    )
    built = _build_model(args, config, None, ["Default"])
    assert built.bond_neighbor_union and not built.bond_neighbor_middle
    extracted = extract_config_mace_model(built)
    assert extracted["bond_neighbor_union"] and extracted["max_ell"] == 2
    with pytest.raises(ValueError, match="exactly one"):
        model(bond_neighbor_union=True, bond_neighbor_middle=True)


@pytest.mark.parametrize("z_basis", ["bessel", "legendre"])
def test_longitudinal_basis_contains_odd_information(z_basis):
    basis = CylindricalBasis(2.5, 3, 2, z_basis=z_basis)
    points = torch.tensor([[0.3, 0.4, 0.7], [0.3, 0.4, -0.7]])
    features, _ = basis(points)
    torch.testing.assert_close(features[0, :3], features[1, :3])
    if z_basis == "bessel":
        assert features.shape[-1] == 7
        torch.testing.assert_close(features[0, 3:6], features[1, 3:6])
        torch.testing.assert_close(features[0, -1], -features[1, -1])
        torch.testing.assert_close(features[0, -1], torch.tensor(0.7 / 2.5))
    else:
        assert features.shape[-1] == 6
        x = torch.tensor(0.7 / 2.5)
        torch.testing.assert_close(
            features[0, 3:], torch.stack((torch.ones_like(x), x, (3 * x * x - 1) / 2))
        )
        torch.testing.assert_close(features[0, 4], -features[1, 4])
        torch.testing.assert_close(features[0, 5], features[1, 5])


def test_union_legendre_normalization_and_invalid_basis():
    basis = CylindricalBasis(2.5, 3, 2, z_basis="legendre", bond_neighbor_union=True)
    features, _ = basis(torch.tensor([[0.1, 0.2, 3.0]]))
    torch.testing.assert_close(features[0, 4], torch.tensor(3.0 / 3.75))
    with pytest.raises(ValueError, match="z_basis"):
        CylindricalBasis(2.5, 3, 2, z_basis="invalid")
    with pytest.raises(ValueError, match="odd degree"):
        CylindricalBasis(2.5, 1, 2, z_basis="legendre")


@pytest.mark.parametrize("z_basis", ["bessel", "legendre"])
def test_smooth_embedding_matches_formula_and_has_finite_axis_derivatives(z_basis):
    basis = CylindricalBasis(2.5, 3, 3, z_basis=z_basis, smooth_theta_embedding=True)
    points = torch.tensor([[0.3, 0.4, 0.7], [0.0, 0.0, 0.7]], requires_grad=True)
    features, angular = basis(points)
    theta = torch.atan2(points[0, 1], points[0, 0])
    expected = (0.5 / 2.5) ** basis.orders.abs() * torch.exp(1j * basis.orders * theta)
    torch.testing.assert_close(angular[0], expected)
    torch.testing.assert_close(angular[1, 3], torch.ones((), dtype=torch.complex128))
    assert torch.count_nonzero(angular[1]) == 1
    loss = features.square().sum() + angular.real.sum() + angular.imag.sum()
    first = torch.autograd.grad(loss, points, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), points)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.parametrize("z_basis", ["bessel", "legendre"])
@pytest.mark.parametrize("smooth", [False, True])
def test_encoding_options_preserve_off_axis_energy_and_force_symmetry(z_basis, smooth):
    net = model(z_basis=z_basis, smooth_theta_embedding=smooth)
    out = net(batch(), training=True)
    rotation = o3.rand_matrix()
    rotated = net(batch(POSITIONS @ rotation.numpy().T), training=True)
    torch.testing.assert_close(out["energy"], rotated["energy"], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        out["forces"] @ rotation.T, rotated["forces"], atol=1e-9, rtol=1e-8
    )
    reflected = net(batch(POSITIONS * np.array([1.0, -1.0, 1.0])), compute_force=False)
    torch.testing.assert_close(
        out["energy"], reflected["energy"], atol=1e-10, rtol=1e-9
    )
    (out["forces"].square().sum() + out["energy"].square().sum()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )


@pytest.mark.parametrize("z_basis", ["bessel", "legendre"])
def test_smooth_axis_environment_is_invariant_and_force_trainable(z_basis):
    net = model(z_basis=z_basis, smooth_theta_embedding=True)
    # The third atom is exactly on the first bond's axis, alongside an off-axis neighbor.
    positions = np.array(
        [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [1.4, 0.0, 0.0], [0.1, 0.5, 0.6]]
    )
    out = net(batch(positions), training=True)
    rotation = o3.rand_matrix()
    rotated = net(batch(positions @ rotation.numpy().T), training=True)
    torch.testing.assert_close(out["energy"], rotated["energy"], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(
        out["forces"] @ rotation.T, rotated["forces"], atol=1e-9, rtol=1e-8
    )
    (out["energy"].square().sum() + out["forces"].square().sum()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )


def test_new_encoding_options_reach_factory_and_metadata():
    from mace.tools.model_script_utils import _build_model
    from mace.tools.scripts_utils import extract_config_mace_model

    parser = tools.build_default_arg_parser()
    args = parser.parse_args(
        [
            "--name=ece",
            "--model=MECE",
            "--z_basis=legendre",
            "--smooth_theta_embedding=True",
        ]
    )
    args.std = 1.0
    args.mean = 0.0
    config = dict(
        r_max=2.5,
        num_bessel=3,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(2),
        avg_num_neighbors=4.0,
        atomic_numbers=[1, 8],
    )
    net = _build_model(args, config, None, ["Default"])
    assert net.z_basis == "legendre" and net.smooth_theta_embedding
    extracted = extract_config_mace_model(net)
    assert extracted["z_basis"] == "legendre" and extracted["smooth_theta_embedding"]
    defaults = parser.parse_args(["--name=ece"])
    assert defaults.z_basis == "bessel" and not defaults.smooth_theta_embedding
