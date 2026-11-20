"""Numerical quadrature, equivariance, graph isolation and differentiation."""
import numpy as np
import pytest
import torch
from graph_longrange.gaussian_multipoles import GaussianWeightedMultipoles


def test_independent_gauss_hermite_quadrature():
    rng = np.random.default_rng(91)
    pos = rng.normal(size=(3, 3)) * 2
    coeff = rng.normal(size=(3, 4))
    tau, sigma = 1.5, 6.
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    xyz = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), -1).reshape(-1, 3)
    w = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing='ij'), -1), axis=-1).ravel() / np.pi**1.5
    u = xyz * (2**.5 * tau)
    qref = np.zeros(3); muref = np.zeros((3, 3))
    for center in range(3):
        for j in range(3):
            relative = pos[j] + u - pos[center]
            mu = coeff[j, [3, 1, 2]]
            integrand = w * (coeff[j, 0] + u @ mu / tau**2) * np.exp(-(relative**2).sum(1) / sigma**2)
            qref[center] += integrand.sum()
            muref[center] += integrand @ relative
    calc = GaussianWeightedMultipoles(tau, sigma)
    q, mu = calc(torch.tensor(pos), torch.tensor(coeff))
    np.testing.assert_allclose(q, qref, atol=3e-12, rtol=1e-12)
    np.testing.assert_allclose(mu, muref, atol=3e-12, rtol=1e-12)


def test_graph_isolation_translation_and_coefficient_residual():
    torch.manual_seed(15)
    pos = torch.randn(5, 3, dtype=torch.double)
    ref = torch.randn(5, 4, dtype=torch.double)
    pred = torch.randn(5, 4, dtype=torch.double)
    calc = GaussianWeightedMultipoles(chunk_size=2)
    delta = calc(pos, pred-ref)
    p = calc(pos, pred); r = calc(pos, ref)
    for d, pv, rv in zip(delta, p, r):torch.testing.assert_close(d, pv-rv)
    shifted = calc(pos+torch.tensor([73., -14., 8.]), pred-ref)
    for x,y in zip(delta,shifted):torch.testing.assert_close(x,y)
    batched = calc(pos.repeat(2,1), (pred-ref).repeat(2,1), torch.tensor([0]*5+[1]*5))
    for x,y in zip(delta,batched):torch.testing.assert_close(y,torch.cat([x,x]))
    rotation,_ = torch.linalg.qr(torch.randn(3,3,dtype=torch.double))
    rotated_coeff = pred-ref
    mu = rotated_coeff[:,[3,1,2]]@rotation.T
    rotated_coeff = torch.cat([rotated_coeff[:,:1],mu[:,[1,2,0]]],dim=1)
    rotated = calc(pos@rotation.T,rotated_coeff)
    torch.testing.assert_close(rotated[0],delta[0]);torch.testing.assert_close(rotated[1],delta[1]@rotation.T)


def test_gradients_and_second_derivatives():
    torch.manual_seed(1)
    pos=torch.randn(3,3,dtype=torch.double,requires_grad=True)
    c=torch.randn(3,4,dtype=torch.double,requires_grad=True)
    calc=GaussianWeightedMultipoles(chunk_size=2)
    def f(p,c):
        q,mu=calc(p,c)
        return torch.cat([q[:,None],mu],dim=1)
    assert torch.autograd.gradcheck(f,(pos,c),atol=2e-6)
    assert torch.autograd.gradgradcheck(f,(pos,c),atol=2e-6)


def test_load_mbis_and_validation():
    from ase import Atoms
    a=Atoms('OH',positions=[[0,0,0],[0,0,1]])
    with pytest.raises(KeyError):GaussianWeightedMultipoles.load_mbis(a)
    a.arrays['mbis_charge_coefficients']=np.arange(8).reshape(2,4).astype(float)
    a.arrays['mbis_spin_coefficients']=np.ones((2,4))
    loaded=GaussianWeightedMultipoles.load_mbis(a,dtype=torch.double)
    assert loaded['charge_coefficients'].shape==(2,4)
    for width in [0,-1,float('nan')]:
        with pytest.raises(ValueError):GaussianWeightedMultipoles(tau=width)
