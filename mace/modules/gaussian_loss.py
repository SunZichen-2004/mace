"""Online Gaussian-window supervision from intrinsic MBIS l=0,1 coefficients."""
import math
from typing import Dict
import torch
from graph_longrange.gaussian_multipoles import GaussianWeightedMultipoles
from .loss import reduce_loss


class GaussianMultipoleLoss(torch.nn.Module):
    """Add four atom/component-mean L2 terms to an existing E/F loss.

    L = L_base + sum_t w_t mean_atoms,components[(G * delta rho)_t²].
    Each scalar is averaged over atoms; each vector over atoms and its three
    components, exactly the force-MSE convention. No extra division by N.
    Targets are intrinsic (q,mu_y,mu_z,mu_x) coefficients, not weighted labels.
    """

    def __init__(self, base_loss, tau=1.5, sigma=6.0, weight=100.0,
                 charge_weight=None, dipole_weight=None, spin_weight=None,
                 spin_dipole_weight=None):
        super().__init__()
        self.base_loss = base_loss
        self.gaussian_window = GaussianWeightedMultipoles(tau=tau, sigma=sigma)
        for name, value in [('charge', charge_weight), ('dipole', dipole_weight),
                            ('spin', spin_weight), ('spin_dipole', spin_dipole_weight)]:
            value = float(weight if value is None else value)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Gaussian loss weights must be finite and nonnegative')
            self.register_buffer('gaussian_' + name + '_weight', torch.tensor(value, dtype=torch.get_default_dtype()))

    @property
    def gaussian_weights(self):
        return {key: getattr(self, 'gaussian_' + key + '_weight')
                for key in ['charge', 'dipole', 'spin', 'spin_dipole']}

    def gaussian_residuals(self, ref, pred) -> Dict[str, torch.Tensor]:
        if getattr(ref, 'pbc', None) is not None and bool(ref.pbc.any()):
            raise ValueError('Gaussian-window supervision currently supports finite nonperiodic systems only')
        charge = getattr(ref, 'mbis_charge_coefficients', None)
        spin = getattr(ref, 'mbis_spin_coefficients', None)
        if charge is None or spin is None:
            raise ValueError('Gaussian supervision requires intrinsic MBIS charge and spin coefficient arrays')
        if pred.get('density_coefficients') is None or pred.get('spin_density') is None:
            raise ValueError('Gaussian supervision requires MACE-POLAR charge and spin density coefficients')
        n = len(ref.positions)
        for value in [charge, spin, pred['density_coefficients'], pred['spin_density']]:
            if value.shape != (n, 4):
                raise ValueError('Gaussian supervision requires l=0,1 coefficients of shape (N,4), order q,y,z,x')
        return self.gaussian_window.residual(ref.positions, pred['density_coefficients'], charge,
                                             pred['spin_density'], spin, ref.batch)

    def weighted_components(self, ref, pred, ddp=None):
        parts = dict(self.base_loss.weighted_components(ref, pred, ddp=ddp))
        residuals = self.gaussian_residuals(ref, pred)
        weights = ref.weight[ref.batch]
        for name, delta in residuals.items():
            w = weights.view((-1,) + (1,) * (delta.ndim - 1))
            parts['gaussian_' + name] = self.gaussian_weights[name] * reduce_loss(w * delta.square(), ddp)
        return parts

    def forward(self, ref, pred, ddp=None):
        return sum(self.weighted_components(ref, pred, ddp=ddp).values())

    def extra_repr(self):
        weights = ', '.join(f'{k}={float(v):g}' for k, v in self.gaussian_weights.items())
        return f'tau={self.gaussian_window.tau:g} A, sigma={self.gaussian_window.sigma:g} A, atom/component mean L2, {weights}'
