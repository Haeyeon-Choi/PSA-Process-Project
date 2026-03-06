"""Recovery metric utilities."""

from __future__ import annotations

from psa_pyomo.process.cycle_model import CycleEvaluation


def recovery_value(evaluation: CycleEvaluation) -> float:
    return evaluation.recovery
