"""Physical response for PolarMACE's internal, oppositely signed electric field."""

import torch


def physical_polarizability(
    dipole: torch.Tensor, internal_field: torch.Tensor, create_graph: bool = False
) -> torch.Tensor:
    """Return -d(mu_i)/d(F_j) in e A^2/V, without symmetrizing or sign clipping.

    PolarMACE adds +F.mu to E; physical field E_phys=-F. For a stable linear
    response E(F)=E0+mu0.F-1/2 F.alpha.F, this returns the positive alpha.
    Scalar VJPs avoid batched-autograd restrictions in accelerated kernels.
    """
    rows = []
    for axis in range(3):
        derivative = torch.autograd.grad(
            [dipole[:, axis].sum()],
            [internal_field],
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if derivative is None:
            derivative = internal_field * 0.0
        rows.append(-derivative)
    return torch.stack(rows, dim=1)
