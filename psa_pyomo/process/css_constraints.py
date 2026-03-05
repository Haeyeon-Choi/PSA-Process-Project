"""Constraint residual calculations for PSA direct optimization."""

from __future__ import annotations

from typing import Dict, Tuple

from psa_pyomo.process.cycle_model import CycleEvaluation


def css_constraint_residuals(
    evaluation: CycleEvaluation,
    purity_min: float,
    recovery_min: float,
) -> Tuple[float, float]:
    """Return inequality residuals g(x) <= 0 for purity/recovery constraints."""
    return purity_min - evaluation.purity, recovery_min - evaluation.recovery


def pressure_ordering_residuals(point: Dict[str, float]) -> Tuple[float, float]:
    """Return residuals for process-pressure ordering constraints.

    Constraints are written as g(x) <= 0:
    - PI <= P0  ->  PI - P0 <= 0
    - Pl <= PI  ->  Pl - PI <= 0
    """

    return point["PI"] - point["P0"], point["Pl"] - point["PI"]
