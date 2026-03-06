"""Constraint residual calculations for PSA direct optimization."""

from __future__ import annotations

from typing import Dict, Tuple

from psa_pyomo.process.cycle_model import CycleEvaluation


def css_constraint_residuals(
    evaluation: CycleEvaluation,
    purity_min: float,
    recovery_min: float,
) -> Tuple[float, float, float]:
    """Return inequality residuals g(x) <= 0 for purity/recovery/CSS constraints."""
    return (
        purity_min - evaluation.purity,
        recovery_min - evaluation.recovery,
        evaluation.css_error,
    )


def pressure_ordering_residuals(point: Dict[str, float]) -> Tuple[float, float]:
    """Return residuals for process-pressure ordering constraints."""
    return point["PI"] - point["P0"], point["Pl"] - point["PI"]
