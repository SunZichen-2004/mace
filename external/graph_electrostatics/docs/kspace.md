# K-space Utilities

This file documents the k-space grid builders and related utilities.
For the Fourier-series conventions used in the code, see `docs/maths/fourier.md`.

## `compute_k_vectors_flat`

```
compute_k_vectors_flat(
    cutoff: float,
    cell_vectors: torch.Tensor,     # [n_graph, 3, 3]
    r_cell_vectors: torch.Tensor,   # [n_graph, 3, 3]
) -> tuple[
    torch.Tensor,  # k_vectors [n_k_total, 3]
    torch.Tensor,  # k_norm2 [n_k_total]
    torch.Tensor,  # k_vector_batch [n_k_total]
    torch.Tensor,  # k0_mask [n_k_total]
]
```

This is the primary k-grid routine used by the energy and feature blocks. It returns a
flattened list of k-vectors grouped by graph, with a `k_vector_batch` index indicating
which graph each vector belongs to. The first vector for each graph is the origin, and
`k0_mask` is `1.0` at those entries and `0.0` elsewhere.

### Inputs

- `cutoff`: reciprocal-space cutoff (same units as `r_cell_vectors`).
- `cell_vectors`: real-space cell matrix with shape `[n_graph, 3, 3]`.
- `r_cell_vectors`: reciprocal cell matrix with shape `[n_graph, 3, 3]`.

The reciprocal cell is defined as

$$
B = 2\pi A^{-T},
$$

so you can compute it as:

```python
r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
```

### Outputs

- `k_vectors`: flattened k-vectors `[n_k_total, 3]`, grouped by graph.
- `k_norm2`: squared norms `[n_k_total]`.
- `k_vector_batch`: graph id for each k-vector `[n_k_total]`.
- `k0_mask`: float mask `[n_k_total]`, `1.0` at k=0.

The search space is a half-space of reciprocal lattice indices (including axis/plane
special cases). This matches the real-valued Fourier-series conventions in
`docs/maths/fourier.md`.

## `compute_k_vectors`

This is the legacy (batched) version that returns a padded tensor and a mask:

```
compute_k_vectors(
    cutoff: float,
    cell_vectors: torch.Tensor,     # [n_graph, 3, 3]
    r_cell_vectors: torch.Tensor,   # [n_graph, 3, 3]
) -> tuple[
    torch.Tensor,  # k_vectors [n_graph, n_max_k, 3]
    torch.Tensor,  # k_norm2 [n_graph, n_max_k]
    torch.Tensor,  # mask [n_graph, n_max_k]
]
```

Use this only if you need the dense `[n_graph, n_max_k, ...]` layout; the current
energy and feature blocks expect the flattened output from `compute_k_vectors_flat`.

## Heuristic cutoff selection

`gto_basis_kspace_cutoff(sigmas, max_l)` provides a simple heuristic for selecting a
reciprocal-space cutoff based on the GTO widths and angular order. This is often a good
starting point when selecting `kspace_cutoff` for the energy and feature blocks.

## Half-space grid and `k0_mask`

The k-grid uses a half-space of reciprocal lattice indices to exploit the symmetry
of real-valued Fourier series. In practice this means storing only one of each
$\pm\mathbf{k}$ pair (typically the $\mathbf{k}_x>0$ half-space), plus the origin.

`compute_k_vectors_flat` always places $\mathbf{k}=\mathbf{0}$ first for each graph and
returns `k0_mask` with `1.0` at those entries and `0.0` elsewhere. The k=0 term is
handled separately in reconstructions and projections (it should not be doubled when
converting from the full lattice to the half-space sum). See further discussion in `maths/`.

## Interpolator formula

The interpolator evaluates a real-valued Fourier series at arbitrary points using

$$
f(\mathbf{r}) = \frac{2}{(2\pi)^3} \sum_{\mathbf{k}_x>0}
\left[\Re\{\tilde{f}(\mathbf{k})\}\cos(\mathbf{k}\cdot\mathbf{r})
-\Im\{\tilde{f}(\mathbf{k})\}\sin(\mathbf{k}\cdot\mathbf{r})\right],
$$

with the k=0 term handled separately. See `interpolator.md` for usage.
