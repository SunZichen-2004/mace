"""Differentiable Gaussian windows of l=0,1 atom-centered CG multipoles.

rho(r) = sum_j [q_j g_tau(r-R_j) - mu_j . grad g_tau(r-R_j)],
g_tau(u) = exp(-u²/(2 tau²)) / (2 pi tau²)**(3/2).
The unnormalized measurement window is exp(-|r-R_i|²/sigma²).
Coefficients are MACE-POLAR real spherical (q, mu_y, mu_z, mu_x).
All lengths are in Angstrom; dipoles returned here are centered on R_i.
This finite-molecule operator does not sum periodic images or use a cutoff.
"""
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class GaussianWeightedMultipoles(nn.Module):
    """Integrate CG charge/spin coefficients, or their residual, analytically.

    ``forward`` accepts (N,4) or (N,C,4), returning monopoles (N[,C])
    and Cartesian local dipoles (N[,C],3). ``observables`` handles the
    independent charge and alpha-minus-beta spin channels together.
    Batch labels isolate different molecules even when their coordinates overlap.
    No reference weighted observables need to be stored in a training dataset.
    """

    def __init__(self, tau: float = 1.5, sigma: float = 6.0, chunk_size: int = 256):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("tau and sigma must be positive finite Angstrom values")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.chunk_size = int(chunk_size)

    def forward(self, positions: Tensor, coefficients: Tensor,
                batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N,3)")
        scalar_input = coefficients.ndim == 2
        if scalar_input:
            coefficients = coefficients.unsqueeze(1)
        if coefficients.ndim != 3 or coefficients.shape[0] != len(positions) or coefficients.shape[-1] != 4:
            raise ValueError("l=0,1 coefficients must have shape (N,4) or (N,C,4)")
        if batch is not None and batch.shape != (len(positions),):
            raise ValueError("batch must have shape (N,)")
        B = self.sigma ** 2 + 2 * self.tau ** 2
        a = self.sigma ** 2 / B
        mu = coefficients[..., [3, 1, 2]]
        charges, dipoles = [], []
        for start in range(0, len(positions), self.chunk_size):
            stop = start + self.chunk_size
            delta = positions.unsqueeze(0) - positions[start:stop].unsqueeze(1)
            K = a ** 1.5 * torch.exp(-delta.square().sum(-1) / B)
            if batch is not None:
                K = K * (batch[start:stop, None] == batch[None, :]).to(K.dtype)
            correction = coefficients[None, :, :, 0] - (2 / B) * torch.einsum("ijc,jkc->ijk", delta, mu)
            charges.append(torch.einsum("ij,ijk->ik", K, correction))
            dipoles.append(a * (torch.einsum("ij,jkc->ikc", K, mu)
                                + torch.einsum("ij,ijc,ijk->ikc", K, delta, correction)))
        if not charges:
            scalar = coefficients.new_empty((0, coefficients.shape[1]))
            vector = coefficients.new_empty((0, coefficients.shape[1], 3))
        else:
            scalar, vector = torch.cat(charges), torch.cat(dipoles)
        if scalar_input:
            return scalar[:, 0], vector[:, 0]
        return scalar, vector

    def observables(self, positions: Tensor, charge_coefficients: Tensor,
                    spin_coefficients: Tensor, batch: Optional[Tensor] = None) -> Dict[str, Tensor]:
        scalar, vector = self(positions, torch.stack((charge_coefficients, spin_coefficients), dim=1), batch)
        return dict(charge=scalar[:, 0], dipole=vector[:, 0],
                    spin=scalar[:, 1], spin_dipole=vector[:, 1])

    def residual(self, positions: Tensor, predicted_charge: Tensor, reference_charge: Tensor,
                 predicted_spin: Tensor, reference_spin: Tensor,
                 batch: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Integrate coefficient differences directly, preserving autograd."""
        return self.observables(positions, predicted_charge - reference_charge,
                                predicted_spin - reference_spin, batch)

    @staticmethod
    def load_mbis(atoms, charge_key="mbis_charge_coefficients", spin_key="mbis_spin_coefficients",
                  device=None, dtype=None) -> Dict[str, Tensor]:
        """Read intrinsic MBIS (q,y,z,x) arrays from an ASE frame, no convolution.

        These coefficients themselves are independent of tau. The configured
        tau defines the CG density used by ``observables`` and ``residual``.
        Missing reference arrays fail rather than silently creating zero targets.
        """
        out = {}
        for name, key in [("charge_coefficients", charge_key), ("spin_coefficients", spin_key)]:
            if key not in atoms.arrays:
                raise KeyError("Missing intrinsic MBIS multipoles: " + key)
            value = torch.as_tensor(atoms.arrays[key], dtype=dtype or torch.get_default_dtype(), device=device)
            if value.shape != (len(atoms), 4) or not torch.isfinite(value).all():
                raise ValueError(key + " must be finite with shape (N,4), order q,y,z,x")
            out[name] = value
        return out

    def extra_repr(self):
        return f"tau={self.tau} A, sigma={self.sigma} A, lmax=1, local Cartesian dipoles"
