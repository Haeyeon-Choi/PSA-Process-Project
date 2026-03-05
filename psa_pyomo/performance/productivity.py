"""Productivity metric utilities."""

from __future__ import annotations

from psa_pyomo.process.cycle_model import CycleEvaluation


def productivity_value(evaluation: CycleEvaluation) -> float:
    return evaluation.productivity
