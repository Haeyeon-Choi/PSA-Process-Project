"""Isotherm equations used in PSA modeling.

The direct optimizer still evaluates full cycle performance through Julia
`PSASimulator`, but this module now provides explicit adsorption equations that
match common PSA formulations and can be reused for future pure-Python models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualSiteLangmuirParameters:
    """Dual-site Langmuir parameters for one component.

    q = qsat1*b1*p/(1+b1*p) + qsat2*b2*p/(1+b2*p)
    """

    qsat1: float
    b1: float
    qsat2: float
    b2: float


def dual_site_langmuir_loading(partial_pressure: float, params: DualSiteLangmuirParameters) -> float:
    """Return equilibrium loading for a component at a given partial pressure."""
    term1 = params.qsat1 * params.b1 * partial_pressure / (1.0 + params.b1 * partial_pressure)
    term2 = params.qsat2 * params.b2 * partial_pressure / (1.0 + params.b2 * partial_pressure)
    return term1 + term2


@dataclass(frozen=True)
class IsothermParameters:
    """Compatibility container for high-level configuration."""

    material_index: int
