"""Overflow-safe backward with the same clipped gradient as the original loss."""
import torch
import torch.distributed as dist


def backward_scale(loss):
    magnitude = loss.detach().abs().double().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(magnitude, op=dist.ReduceOp.MAX)
    if not torch.isfinite(magnitude):
        raise FloatingPointError('Non-finite POLAR-2 objective on at least one rank')
    value = float(magnitude)
    # Ordinary batches retain their original backward. All ranks must use the
    # same factor before DDP averages gradients.
    return 1. / value if value > 1.e4 else 1.


def clip_scaled_gradients(parameters, scale, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.
    norm = torch.stack([p.grad.detach().double().norm() for p in parameters]).norm()
    if not torch.isfinite(norm):
        raise FloatingPointError('Non-finite POLAR-2 gradients after scaled backward')
    # g_scaled = scale*g. Combining unscale and clip avoids ever materializing
    # the unscaled huge gradient in float32. The epsilon matches clip_grad_norm_.
    coefficient = norm.new_tensor(1. / scale)
    if max_norm is not None:
        coefficient = torch.minimum(coefficient, max_norm / (norm + 1.e-6 * scale))
    with torch.no_grad():
        for p in parameters:
            p.grad.copy_((p.grad.double() * coefficient).to(p.grad.dtype))
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError('Non-finite POLAR-2 gradient after unscale/clip')
    return float(norm / scale)
