"""Adsorption kinetics equations for PSA modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LDFParameters:
    """Linear driving force (LDF) kinetic constant."""

    k_ldf: float


def ldf_uptake_rate(q: float, q_star: float, params: LDFParameters) -> float:
    """Return dq/dt from LDF kinetics: dq/dt = k_ldf * (q* - q)."""
    return params.k_ldf * (q_star - q)


@dataclass(frozen=True)
class KineticsParameters:
    """Compatibility container for high-level configuration."""

    n_grid: int