# Gaussian-window observations of atomic multipoles

`graph_longrange.GaussianWeightedMultipoles` analytically convolves l=0,1 CG multipoles with an unnormalized Gaussian measurement window. It supports autograd through coefficients and positions, multiple channels, batching, and chunked evaluation. It currently represents finite, nonperiodic molecules; it does not sum periodic images or truncate source atoms.

Intrinsic atomic coefficients have order `(q, mu_y, mu_z, mu_x)`, using the same real spherical and Cartesian multipole normalization as MACE-POLAR-1-M. MBIS charge coefficients describe nuclear-minus-electron charge; spin coefficients describe n_alpha-n_beta. Partition dipoles must be centered on their own source atom, in e Angstrom or spin Angstrom. The MBIS coefficients themselves do not depend on the reconstruction width tau.

The density and window are

```
rho(r) = sum_j [q_j g_tau(r-R_j) - mu_j . grad g_tau(r-R_j)]
g_tau(u) = exp(-|u|^2/(2 tau^2)) / (2 pi tau^2)^(3/2)
G_i(r) = exp(-|r-R_i|^2 / sigma^2)
```

Define Delta_ij=R_j-R_i, B=sigma^2+2 tau^2, a=sigma^2/B and K_ij=a^(3/2) exp(-|Delta_ij|^2/B). The returned scalar and local Cartesian dipole are

```
Q_i = sum_j K_ij [q_j - 2 mu_j . Delta_ij / B]
D_i = a sum_j K_ij [mu_j + Delta_ij (q_j - 2 mu_j . Delta_ij / B)]
```

Every source atom in the same molecule contributes to each observation center. Overlapping windows do not define an exclusive atomic partition; neither sum_i Q_i nor sum_i spin_i is a molecular population sum.

```python
import torch
from ase.io import read
from graph_longrange import GaussianWeightedMultipoles

atoms = read("frame.extxyz")
window = GaussianWeightedMultipoles(tau=1.5, sigma=6.0)
reference = window.load_mbis(atoms, dtype=torch.float64)
positions = torch.as_tensor(atoms.positions, dtype=torch.float64)
observed = window.observables(positions, reference["charge_coefficients"],
                              reference["spin_coefficients"])
# observed: charge (N,), dipole (N,3), spin (N,), spin_dipole (N,3)
```

The default ASE array names are `mbis_charge_coefficients` and `mbis_spin_coefficients`, both (N,4). Missing or malformed arrays raise errors. `load_mbis` accepts custom keys and device/dtype. Passing `batch` labels to `forward`, `observables`, or `residual` isolates different molecules even if they occupy the same coordinates.

For online supervision, evaluate the difference directly:

```python
delta = window.residual(positions, predicted_charge, reference_charge,
                         predicted_spin, reference_spin, batch)
loss = sum(100.0 * value.square().mean() for value in delta.values())
```

Because the operator is linear in multipole coefficients, this equals integrating model and MBIS densities separately and subtracting. The scalar MSE averages over atoms; each dipole MSE averages over atoms and three Cartesian components. The normalization matches a component-wise force MSE, with no second division by the atom count. In MACE's training integration, configuration weights and DDP reduction follow the force-loss convention.

Tests use independent 3D Gauss-Hermite quadrature, translations/rotations, overlapping graph coordinates, coefficient-residual linearity, and first/second derivative checks. MACE integration tests cover dataset loading, loss gradients, batch replication, unequal batch sizes, zero-weight configurations, and missing-reference/periodic-input errors.
