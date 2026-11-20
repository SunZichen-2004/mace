# Interpolator

This file documents the real-space interpolator used to evaluate a Fourier series at
arbitrary points. It is intended for visualization and analysis rather than fast
production use.

## Function

The interpolator is implemented as a function:

```
evaluate_fourier_series_at_points_flat(
    k_vectors: torch.Tensor,        # [n_k_total, 3]
    k_vector_batch: torch.Tensor,   # [n_k_total]
    fourier_coefficients: torch.Tensor,  # [n_k_total, 2]
    sample_points: torch.Tensor,    # [n_points, 3]
    sample_batch: torch.Tensor,     # [n_points]
    k0_mask: torch.Tensor,          # [n_k_total]
) -> torch.Tensor
```

The Fourier coefficients are the real and imaginary parts of a real-valued series,
stored as `[Re, Im]` per k-vector. The function assumes the same half-space convention
as the main code paths (typically $\mathbf{k}_x>0$) and applies the correct factor of 2,
with a half-weight for the k=0 term.

## Usage example

```python
from graph_longrange.kspace import evaluate_fourier_series_at_points_flat

values = evaluate_fourier_series_at_points_flat(
    k_vectors=k_vectors,
    k_vector_batch=k_vector_batch,
    fourier_coefficients=fourier_coefficients,  # [n_k_total, 2]
    sample_points=probe_points,  # [n_points, 3]
    sample_batch=probe_batch,    # [n_points]
    k0_mask=k0_mask,
)
```

## Notes

- `k_vector_batch` and `sample_batch` must refer to the same graph ids.
- The coefficients should be computed using the same conventions as the energy and
  feature blocks (see `docs/maths/fourier.md` and `docs/kspace.md`).
- This function evaluates a real-valued field only; for complex series, use your own
  reconstruction.
