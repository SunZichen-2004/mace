from __future__ import annotations

from enum import Enum


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    QUADRUPOLE = "quadrupole"
    SPIN_DIPOLE = "spin_dipole"
    SPIN_QUADRUPOLE = "spin_quadrupole"
    POLARIZABILITY = "polarizability"
    HEAD = "head"
    CHARGES = "REF_charges"
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    MBIS_CHARGE_COEFFICIENTS = "mbis_charge_coefficients"
    MBIS_SPIN_COEFFICIENTS = "mbis_spin_coefficients"
    ELEC_TEMP = "elec_temp"
    MAGMOM = "REF_magmom"
    MAGFORCES = "REF_magforces"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict
