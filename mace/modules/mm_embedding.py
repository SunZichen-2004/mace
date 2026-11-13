"""External point-charge potentials in PolarMACE's existing Gaussian bases."""

import math

import torch


def MM_potential_features(
    model,
    positions,
    batch,
    MM_coordinate,
    MM_charges,
    MM_batch,
    cell,
    pbc,
    k_vectors,
    k_norm2,
    k_vector_batch,
    k0_mask,
    method="direct",
    MM_charge_width=0.0,
):
    """Return descriptor features and the potential conjugate to ML multipoles.

    Both outputs use the same MM sources. The second contracts with the final
    density to give the cross energy, without MM self energy or a factor 1/2.
    Coordinates are Angstrom, charges elementary charges, energies eV.
    """
    from graph_longrange.gto_utils import get_Cl_sigma
    from graph_longrange.realspace_electrostatics import smeared_coulomb_kernels
    from graph_longrange.utils import FIELD_CONSTANT

    if method not in ("direct", "PME"):
        raise ValueError("MM_method must be 'direct' or 'PME'.")
    if not math.isfinite(MM_charge_width) or MM_charge_width < 0:
        raise ValueError("MM_charge_width must be finite and nonnegative.")
    descriptor = model.electric_potential_descriptor
    feature_basis = descriptor.feature_basis
    density_basis = model.coulomb_energy.density_basis
    if max(feature_basis.max_l, density_basis.max_l) > 1:
        raise NotImplementedError(
            "MM embedding currently supports charges and dipoles."
        )

    if not torch.any(pbc):

        # Build only within-graph pairs, including for ragged batches.
        num_graphs = len(cell)
        counts = torch.bincount(MM_batch, minlength=num_graphs)
        offsets = counts.cumsum(0) - counts
        counts = counts[batch]
        starts = counts.cumsum(0) - counts
        receiver = torch.repeat_interleave(
            torch.arange(len(positions), device=positions.device), counts
        )
        sender = torch.argsort(MM_batch, stable=True)[
            offsets[batch[receiver]]
            + torch.arange(len(receiver), device=positions.device)
            - starts[receiver]
        ]
        sigmas = sorted(set(feature_basis.sigmas) | set(density_basis.sigmas))
        widths = positions.new_tensor(sigmas)
        combined = torch.sqrt(widths.square() + MM_charge_width**2)
        scalar = positions.new_zeros((len(positions), len(widths)))
        gradient = positions.new_zeros((len(positions), len(widths), 3))
        max_l = max(feature_basis.max_l, density_basis.max_l)
        for start in range(0, len(receiver), 2**20):
            target = receiver[start : start + 2**20]
            source = sender[start : start + 2**20]
            separation = positions[target] - MM_coordinate[source]
            kernels = smeared_coulomb_kernels(
                separation.square().sum(-1, keepdim=True), combined, max_l
            )
            charges = MM_charges[source, None]
            scalar = scalar.index_add(0, target, charges * kernels.b0)
            if max_l:
                gradient = gradient.index_add(
                    0,
                    target,
                    -charges[..., None]
                    * separation[:, None, :]
                    * kernels.b1[..., None],
                )

        def project(basis):
            indices = [sigmas.index(w) for w in basis.sigmas]
            prefactor = FIELD_CONSTANT / (4 * math.pi)
            factors = positions.new_tensor(
                [
                    get_Cl_sigma(0, w, basis.normalize)
                    / get_Cl_sigma(0, w, "multipoles")
                    for w in basis.sigmas
                ]
            )
            parts = [prefactor * scalar[:, indices] * factors]
            if basis.max_l:
                factors = positions.new_tensor(
                    [
                        get_Cl_sigma(1, w, basis.normalize)
                        / get_Cl_sigma(1, w, "multipoles")
                        for w in basis.sigmas
                    ]
                )
                parts.append(
                    (
                        prefactor
                        * gradient[:, indices][..., [1, 2, 0]]
                        * factors[None, :, None]
                    ).reshape(len(positions), -1)
                )
            return torch.cat(parts, -1)

        return project(feature_basis), project(density_basis)

    if not torch.all(pbc):
        raise NotImplementedError(
            "MM embedding supports nonperiodic or fully periodic cells."
        )
    from graph_longrange.features import (
        apply_coulomb_kernel_batch,
        compute_coulomb_factor,
        project_to_features_batch,
    )

    if method == "PME":
        from .mm_pme import (
            precompute_pme_source_geometry,
            pme_structure_factor,
        )

        geometry = precompute_pme_source_geometry(
            k_vectors,
            k_vector_batch,
            MM_coordinate,
            MM_batch,
            cell,
            len(cell),
            5,
            6.0,
        )
        cosine, sine = pme_structure_factor(
            MM_charges[:, None], geometry, len(k_vectors)
        )
    else:
        phase = k_vectors @ MM_coordinate.t()
        mask = (k_vector_batch[:, None] == MM_batch[None, :]).to(phase.dtype)
        cosine = (phase.cos() * mask) @ MM_charges[:, None]
        sine = (phase.sin() * mask) @ MM_charges[:, None]
    volume = torch.linalg.det(cell).abs()[k_vector_batch]
    density = (2 * math.pi) ** 3 * torch.cat((cosine, -sine), -1) / volume[:, None]
    density = density * torch.exp(-0.5 * MM_charge_width**2 * k_norm2)[:, None]
    potential = apply_coulomb_kernel_batch(
        density, compute_coulomb_factor(k_norm2, k0_mask)
    )
    phase = k_vectors @ positions.t()
    mask = (k_vector_batch[:, None] == batch[None, :]).to(phase.dtype)
    weights = torch.where(k0_mask > 0, 0.5, 1.0)

    def project(basis):
        return project_to_features_batch(
            potential,
            basis(k_vectors, k_norm2, k0_mask),
            phase.cos() * mask,
            phase.sin() * mask,
            weights,
        ).reshape(len(positions), -1)

    return descriptor._permute_output_channels(project(feature_basis)), project(
        density_basis
    )
