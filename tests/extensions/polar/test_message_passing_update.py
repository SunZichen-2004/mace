"""Response must propagate between atoms and preserve force differentiability."""

import torch
from e3nn import o3
from mace.modules.field_blocks import MessagePassingMultipoleUpdate


def test_message_passing_symmetry_and_nonlocal_response():
    torch.manual_seed(23)
    torch.set_default_dtype(torch.float64)
    block = MessagePassingMultipoleUpdate(
        node_attrs_irreps=o3.Irreps("2x0e"),
        node_feats_irreps=o3.Irreps("4x0e + 4x1o"),
        edge_attrs_irreps=o3.Irreps.spherical_harmonics(2),
        edge_feats_irreps=o3.Irreps("2x0e"),
        target_irreps=o3.Irreps("4x0e + 4x1o"),
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        avg_num_neighbors=2.0,
        potential_irreps=o3.Irreps("2x0e + 2x1o + 2x0e + 2x1o"),
        charges_irreps=o3.Irreps("2x0e + 2x1o"),
        field_norm_factor=1.0,
        num_channels=4,
    )
    attrs = torch.eye(2)
    h = torch.randn(2, 16, requires_grad=True)
    v = torch.randn(2, 16, requires_grad=True)
    p = torch.randn(2, 8, requires_grad=True)
    vectors = torch.randn(2, 3, requires_grad=True)
    edges = torch.tensor([[0, 1], [1, 0]])
    radial = torch.randn(2, 2, requires_grad=True)

    def evaluate(h, v, p, vectors):
        sh = o3.spherical_harmonics(
            range(3), vectors, normalize=True, normalization="component"
        )
        return block(attrs, h, sh, radial, edges, v, p)

    out = evaluate(h, v, p, vectors)
    # W^v is shared by the two spin channels in the specified F projection.
    torch.testing.assert_close(
        evaluate(h, torch.cat([v[:, 8:], v[:, :8]], dim=-1), p, vectors), out
    )
    # A field change at atom 0 affects the update at its neighbor atom 1.
    cross = torch.autograd.grad(out[1].square().sum(), v, retain_graph=True)[0][0]
    assert cross.norm() > 1e-10
    for rotation in (o3.rand_matrix(), -torch.eye(3)):
        D = o3.Irreps("4x0e+4x1o").D_from_matrix(rotation)
        Dp = o3.Irreps("2x0e+2x1o").D_from_matrix(rotation)
        Dv = o3.Irreps("2x0e+2x1o+2x0e+2x1o").D_from_matrix(rotation)
        Dout = o3.Irreps("2x0e+2x1o+2x0e").D_from_matrix(rotation)
        rotated = evaluate(h @ D.T, v @ Dv.T, p @ Dp.T, vectors @ rotation.T)
        torch.testing.assert_close(rotated, out @ Dout.T, rtol=1e-7, atol=1e-8)
    inputs = (h, v, p, vectors, radial)
    first = torch.autograd.grad(out.square().sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(x.square().sum() for x in first), inputs)
    assert all(torch.isfinite(x).all() for x in second)
    torch.set_default_dtype(torch.float32)
