"""Gaussian multipole features of erfc(alpha*r)/r, in graph_longrange units.

For moment-normalized source GTOs, convolution with erf(alpha*r)/r is
exactly Coulomb convolution after broadening the source variance by
1/(2*alpha**2). Receiver widths and normalizations stay unchanged.
"""

import math

import torch
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.gto_utils import GTOSelfInteractionBlock


class _DifferenceSelfInteraction(torch.nn.Module):
    def __init__(self, full, long_range):
        super().__init__()
        self.full = full
        self.long_range = long_range

    def forward(self, source_feats):
        return self.full(source_feats) - self.long_range(source_feats)


class _DifferenceRealSpaceFeatures(torch.nn.Module):
    def __init__(self, full, long_range):
        super().__init__()
        self.full = full
        self.long_range = long_range

    def forward(self, source_feats, node_positions, batch):
        full, full_self, _ = self.full(source_feats, node_positions, batch)
        long_range, long_self, _ = self.long_range(source_feats, node_positions, batch)
        return full - long_range, full_self - long_self, None


class ScreenedGTOElectrostaticFeatures(GTOElectrostaticFeatures):
    """Fixed-range spin convolution; supports open molecules and 3D PBC.

    Unlike Coulomb, the periodic kernel has a finite, nonzero zero mode.
    Coulomb slab/molecular-box corrections do not apply to this operator.
    Partial PBC and mixed open/periodic batches are rejected explicitly.
    """

    def __init__(self, screening_alpha=0.5, **kwargs):
        if not math.isfinite(screening_alpha) or screening_alpha <= 0:
            raise ValueError("spin_field_alpha must be finite and positive")
        self.screening_alpha = float(screening_alpha)
        kwargs.setdefault("pbc_handling", "auto")
        super().__init__(**kwargs)
        self.self_interaction_terms = _DifferenceSelfInteraction(
            self.self_interaction_terms,
            GTOSelfInteractionBlock(
                l_source=self.density_basis.max_l,
                sigma_source=self._broadened_source_width(),
                l_receive=self.feature_basis.max_l,
                sigmas_receive=self.feature_basis.sigmas,
                normalize_source="multipoles",
                normalize_receive=self.feature_basis.normalize,
            ),
        )

    def _broadened_source_width(self):
        return math.sqrt(
            self.density_basis.sigmas[0] ** 2 + 1.0 / (2.0 * self.screening_alpha**2)
        )

    def _build_realspace_features(self):
        full = super()._build_realspace_features()
        long_range = type(full)(
            density_max_l=self.density_basis.max_l,
            density_smearing_width=self._broadened_source_width(),
            projection_max_l=self.feature_basis.max_l,
            projection_smearing_widths=self.feature_basis.sigmas,
            include_self_interaction=self.include_self_interaction,
            integral_normalization=self.feature_basis.normalize,
        )
        return _DifferenceRealSpaceFeatures(full, long_range)

    def _select_precompute_geometry_impl(self):
        if self.pbc_handling == "realspace":
            return self._precompute_geometry_realspace
        if self.pbc_handling == "pbc":
            return self._precompute_geometry_pbc
        if self.pbc_handling == "auto":
            return self._precompute_geometry_screened_auto
        raise ValueError(
            "Screened spin fields support pbc_handling=auto, realspace or pbc"
        )

    def _select_forward_dynamic_impl(self):
        if self.pbc_handling == "realspace":
            return self._forward_dynamic_realspace
        if self.pbc_handling == "pbc":
            return self._forward_dynamic_pbc
        if self.pbc_handling == "auto":
            return self._forward_dynamic_screened_auto
        raise ValueError("Unsupported screened-spin boundary mode")

    def _precompute_geometry_screened_auto(self, **kwargs):
        pbc = kwargs["pbc"].bool()
        if not bool(pbc.any()):
            return self._precompute_geometry_realspace(**kwargs)
        if bool(pbc.all()):
            return self._precompute_geometry_pbc(**kwargs)
        raise ValueError(
            "Screened spin fields require all-open or all-3D-periodic batches"
        )

    def _precompute_geometry_periodic_common(self, **kwargs):
        cache = super()._precompute_geometry_periodic_common(**kwargs)
        k2 = kwargs["k_norm2"]
        zero = kwargs["k0_mask"] > 0
        safe_k2 = torch.where(zero, torch.ones_like(k2), k2)
        # FIELD_CONSTANT already contains the 4*pi Coulomb unit factor.
        scale = 1.0 / (4.0 * self.screening_alpha**2)
        cache["k_factor_coulomb"] = torch.where(
            zero, torch.full_like(k2, scale), -torch.expm1(-k2 * scale) / safe_k2
        )
        return cache

    def _forward_dynamic_screened_auto(self, source_feats, cache):
        if cache["mode"] == "realspace":
            return self._forward_dynamic_realspace(source_feats, cache)
        return self._forward_dynamic_pbc(source_feats, cache)
