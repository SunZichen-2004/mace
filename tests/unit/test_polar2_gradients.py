import pytest
import torch
from mace.tools.polar2_gradients import backward_scale, clip_scaled_gradients

@pytest.mark.parametrize('size',[.1,1.e10])
def test_scaled_float32_clipping_matches_float64_reference(size):
    x=torch.nn.Parameter(torch.tensor([size,2*size],dtype=torch.float32))
    ref=torch.nn.Parameter(x.detach().double())
    loss=(x.pow(3).double().square()).sum()
    reference=(ref.pow(3).square()).sum();reference.backward()
    torch.nn.utils.clip_grad_norm_([ref],1.)
    scale=backward_scale(loss);(loss*scale).backward()
    clip_scaled_gradients([x],scale,1.)
    torch.testing.assert_close(x.grad.double(),ref.grad,rtol=2e-6,atol=1e-7)


def test_nonfinite_still_raises():
    with pytest.raises(FloatingPointError):backward_scale(torch.tensor(float('nan')))
    p=torch.nn.Parameter(torch.ones(1));p.grad=torch.tensor([float('inf')])
    with pytest.raises(FloatingPointError):clip_scaled_gradients([p],1.,1.)
