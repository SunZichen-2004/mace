"""Physical invariance and response gradients of the self-TP field readout."""

import pytest
import torch
from e3nn import o3

from mace.modules.field_blocks import (
    OneBodyNoFeildTPMLPFieldReadout,
    OneBodyTPMLPFieldReadout,
    field_readout_blocks,
)
from mace.modules.wrapper_ops import CuEquivarianceConfig


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    yield
    torch.set_default_dtype(previous)


def make_readout(
    layer_norm=False,
    node_irreps="4x0e+4x1o+4x2e",
    cueq=None,
    readout_cls=OneBodyTPMLPFieldReadout,
):
    irreps = o3.Irreps(node_irreps)
    return readout_cls(
        node_attrs_irreps=o3.Irreps("2x0e"),
        node_feats_irreps=irreps,
        edge_attrs_irreps=o3.Irreps.spherical_harmonics(2),
        edge_feats_irreps=o3.Irreps("3x0e"),
        target_irreps=irreps,
        hidden_irreps=irreps,
        avg_num_neighbors=3.0,
        potential_irreps=o3.Irreps("4x0e+4x1o"),
        charges_irreps=2 * o3.Irreps.spherical_harmonics(1),
        cueq_config=cueq,
        layer_norm=layer_norm,
    )


def random_inputs(block, nodes=3):
    return (
        block.node_feats_irreps.randn(nodes, -1),
        block.potential_irreps.randn(nodes, -1),
        block.charges_irreps.randn(nodes, -1),
        block.charges_irreps.randn(nodes, -1),
    )


def energy(block, inputs):
    node, field, initial, induced = inputs
    return block(
        node_attrs=node.new_zeros((node.shape[0], 2)),
        node_feats=node,
        edge_attrs=node.new_zeros((0, 9)),
        edge_feats=node.new_zeros((0, 3)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        field_feats=field,
        charges_0=initial,
        charges_induced=induced,
    )


@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("inversion", [False, True])
def test_energy_and_response_gradients_transform_correctly(layer_norm, inversion):
    block = make_readout(layer_norm)
    inputs = tuple(x.requires_grad_() for x in random_inputs(block))
    rotation = o3.rand_matrix()
    if inversion:
        rotation = -rotation
    irreps = (
        block.node_feats_irreps,
        block.potential_irreps,
        block.charges_irreps,
        block.charges_irreps,
    )
    matrices = [ir.D_from_matrix(rotation) for ir in irreps]
    rotated = tuple(
        (x.detach() @ matrix.T).requires_grad_() for x, matrix in zip(inputs, matrices)
    )
    reference = energy(block, inputs)
    actual = energy(block, rotated)
    torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-8)
    reference_grads = torch.autograd.grad(reference.sum(), inputs)
    actual_grads = torch.autograd.grad(actual.sum(), rotated)
    for got, want, matrix in zip(actual_grads, reference_grads, matrices):
        torch.testing.assert_close(got, want @ matrix.T, atol=1e-8, rtol=1e-8)


def test_vector_response_survives_without_geometric_vectors():
    block = make_readout()
    node, field, initial, induced = random_inputs(block)
    node.zero_()
    initial.zero_()
    # Only vector electronics remain, so a scalar input cannot explain a
    # response through the old geometry-dot-electronics path.
    masks = []
    for ir in (block.potential_irreps, block.charges_irreps):
        mask = torch.zeros(ir.dim)
        for (_, component), part in zip(ir, ir.slices()):
            if component.l == 1:
                mask[part] = 1
        masks.append(mask)
    field = (field * masks[0]).requires_grad_()
    induced = (induced * masks[1]).requires_grad_()
    grads = torch.autograd.grad(
        energy(block, (node, field, initial, induced)).sum(), (field, induced)
    )
    for grad, mask in zip(grads, masks):
        assert (grad * mask).abs().max().item() > 1e-10


@pytest.mark.parametrize("layer_norm", [False, True])
def test_first_and_second_derivatives_match_finite_differences(layer_norm):
    block = make_readout(layer_norm, "2x0e+2x1o")
    inputs = tuple(x.requires_grad_() for x in random_inputs(block, nodes=1))

    def fn(*args):
        return energy(block, args)

    assert torch.autograd.gradcheck(fn, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, inputs, fast_mode=True)


@pytest.mark.parametrize("layer_norm", [False, True])
def test_zero_inputs_have_finite_second_derivatives(layer_norm):
    block = make_readout(layer_norm)
    inputs = tuple(torch.zeros_like(x).requires_grad_() for x in random_inputs(block))
    first = torch.autograd.grad(energy(block, inputs).sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(x.sum() for x in first), inputs)
    assert all(torch.isfinite(x).all() for x in (*first, *second))


@pytest.mark.parametrize("node_irreps", ["4x0e", "3x1o+4x0e+2x2e", "2x0e+2x0o+2x1o"])
def test_scalar_selection_and_nonuniform_irreps(node_irreps):
    block = make_readout(node_irreps=node_irreps)
    inputs = random_inputs(block)
    assert energy(block, inputs).shape == (3,)
    assert block._scalar_indices.numel() == block.node_feats_irreps.count("0e")


@pytest.mark.cueq
def test_ir_mul_inputs_match_mul_ir_on_cpu():
    reference = make_readout()
    config = CuEquivarianceConfig(enabled=True, layout="ir_mul")
    actual = make_readout(cueq=config)
    actual.load_state_dict(reference.state_dict(), strict=False)
    inputs = random_inputs(reference)
    irreps = (
        reference.node_feats_irreps,
        reference.potential_irreps,
        reference.charges_irreps,
        reference.charges_irreps,
    )
    transposed = []
    for x, ir in zip(inputs, irreps):
        transposed.append(
            torch.cat(
                [
                    x[:, part]
                    .reshape(x.shape[0], mul, component.dim)
                    .transpose(1, 2)
                    .reshape(x.shape[0], -1)
                    for (mul, component), part in zip(ir, ir.slices())
                ],
                dim=-1,
            )
        )
    torch.testing.assert_close(energy(actual, transposed), energy(reference, inputs))


def test_layer_norm_defaults_off_and_rejects_single_channel_collapse():
    assert isinstance(make_readout().tp_norm, torch.nn.Identity)
    with pytest.raises(ValueError, match="at least two"):
        make_readout(layer_norm=True, node_irreps="1x0e+1x1o")


def test_no_field_readout_ignores_field_and_keeps_charge_response():
    assert (
        field_readout_blocks["OneBodyNoFeildTPMLPFieldReadout"]
        is OneBodyNoFeildTPMLPFieldReadout
    )
    block = make_readout(readout_cls=OneBodyNoFeildTPMLPFieldReadout)
    node, field, initial, induced = random_inputs(block)
    field.requires_grad_()
    induced.requires_grad_()
    reference = energy(block, (node, field, initial, induced))
    changed_field = field.detach() * 7.0 + 3.0
    torch.testing.assert_close(
        energy(block, (node, changed_field, initial, induced)), reference
    )
    field_grad, charge_grad = torch.autograd.grad(
        reference.sum(), (field, induced), allow_unused=True
    )
    assert field_grad is None
    assert charge_grad is not None and charge_grad.abs().max().item() > 1e-10
    assert not any("linear_up_v" in name for name, _ in block.named_parameters())
