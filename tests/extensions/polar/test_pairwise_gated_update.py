"""Symmetry and field-response checks for the opt-in multipole update."""

import torch
from e3nn import o3

from mace.modules.field_blocks import (
    PairwiseGatedMultipoleUpdate,
    field_update_blocks,
)


def _block():
    node_irreps = o3.Irreps("4x0e + 4x1o")
    field_irreps = o3.Irreps("2x0e + 2x1o")
    return PairwiseGatedMultipoleUpdate(
        node_attrs_irreps=o3.Irreps("2x0e"),
        node_feats_irreps=node_irreps,
        edge_attrs_irreps=o3.Irreps("1x0e"),
        edge_feats_irreps=o3.Irreps("1x0e"),
        target_irreps=node_irreps,
        hidden_irreps=node_irreps,
        avg_num_neighbors=2.0,
        potential_irreps=field_irreps,
        charges_irreps=field_irreps,
        field_norm_factor=1.0,
    ).double()


def _forward(block, attrs, h, v, p):
    empty = h.new_empty((0,))
    return block(attrs, h, empty, empty, empty.long(), v, p)


def test_direct_field_path_and_rotation_equivariance():
    torch.manual_seed(12)
    assert (
        field_update_blocks["PairwiseGatedMultipoleUpdate"]
        is PairwiseGatedMultipoleUpdate
    )
    block = _block()
    attrs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    h = torch.randn(1, 16, dtype=torch.float64)
    h[:, 4:] = 0.0  # no local vector direction
    p = torch.randn(1, 8, dtype=torch.float64)
    p[:, 2:] = 0.0  # no pre-existing dipole
    v = torch.randn(1, 8, dtype=torch.float64)
    y = _forward(block, attrs, h, v, p)
    assert y[:, 2:8].norm() > 1e-8

    for transform in (o3.rand_matrix().double(), -torch.eye(3, dtype=torch.float64)):
        node_d = o3.Irreps("4x0e + 4x1o").D_from_matrix(transform)
        field_d = o3.Irreps("2x0e + 2x1o").D_from_matrix(transform)
        out_d = o3.Irreps("2x0e + 2x1o + 2x0e").D_from_matrix(transform)
        y_transformed = _forward(
            block, attrs, h @ node_d.T, v @ field_d.T, p @ field_d.T
        )
        torch.testing.assert_close(y_transformed, y @ out_d.T, rtol=1e-9, atol=1e-9)


class _SelectStream(torch.nn.Module):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def forward(self, descriptors):
        gates = descriptors.new_zeros((descriptors.shape[0], 3, 8))
        gates[:, self.stream, :] = 1.0
        return gates.flatten(1)


def _reference_tp(tp):
    reference = o3.TensorProduct(
        tp.irreps_in1, tp.irreps_in2, tp.irreps_out,
        instructions=[
            (ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight)
            for ins in tp.instructions
        ],
    ).double()
    with torch.no_grad():
        reference.weight.copy_(tp.weight)
    return reference


def test_three_paths_use_raw_geometry_shared_field_and_original_tp():
    """Isolate each path and check the TP against e3nn, not our sparse kernel."""
    torch.manual_seed(19)
    block = _block()
    attrs = torch.eye(2, dtype=torch.float64)
    h = torch.randn(2, 16, dtype=torch.float64)
    v = torch.randn(2, 8, dtype=torch.float64)
    p = torch.randn(2, 8, dtype=torch.float64)
    # A nonzero scalar embedding bias belongs to V, never to pure F.
    with torch.no_grad():
        block.h_linear.bias.fill_(0.37)
    field = block.v_linear(v)
    mixed = field + block.h_linear(h) + block.p_linear(p)
    dot_reference = _reference_tp(block.dot_products)
    tp_reference = _reference_tp(block.tp_mix)
    torch.testing.assert_close(block.dot_products(h, mixed), dot_reference(h, mixed))
    expected_paths = (h, field, tp_reference(mixed, h))
    captured = []

    def capture(_module, args):
        captured.append(args[0])

    hook = block.readout.register_forward_pre_hook(capture)
    try:
        for stream, expected in enumerate(expected_paths):
            block.gate_mlp = _SelectStream(stream)
            _forward(block, attrs, h, v, p)
            torch.testing.assert_close(captured[-1], expected)
        # With only F selected, neither density nor geometry can affect g.
        block.gate_mlp = _SelectStream(1)
        _forward(block, attrs, h * 2, v, p + 1)
        torch.testing.assert_close(captured[-1], field)
    finally:
        hook.remove()


def test_second_derivatives_and_new_checkpoint_roundtrip():
    """Force training needs differentiable response through all three inputs."""
    torch.manual_seed(23)
    block = _block()
    attrs = torch.eye(2, dtype=torch.float64)
    inputs = [
        torch.randn(2, width, dtype=torch.float64, requires_grad=True)
        for width in (16, 8, 8)
    ]
    output = _forward(block, attrs, *inputs)
    first = torch.autograd.grad(output.square().sum(), inputs, create_graph=True)
    for gradient in first:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    second = torch.autograd.grad(sum(g.square().sum() for g in first), inputs)
    assert all(torch.isfinite(g).all() for g in second)
    restored = _block()
    restored.load_state_dict(block.state_dict(), strict=True)
    torch.testing.assert_close(_forward(restored, attrs, *inputs), output)
