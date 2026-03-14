"""Column variable definitions and bounds for optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

VARIABLE_ORDER = ["L", "P0", "ndot", "tads", "alpha", "beta", "PI", "Pl"]


@dataclass
class ColumnDecisionSpace:
    """Decision-variable bounds and initialization."""

    bounds: Dict[str, Tuple[float, float]]
    initial_point: Dict[str, float]

    def validate(self) -> None:
        for name in VARIABLE_ORDER:
            lb, ub = self.bounds[name]
            x0 = self.initial_point[name]
            if lb > ub:
                raise ValueError(f"Invalid bounds for {name}: lb={lb}, ub={ub}")
            if not (lb <= x0 <= ub):
                raise ValueError(f"Initial value for {name} is outside bounds: {x0} not in [{lb}, {ub}]")

        if self.initial_point["PI"] > self.initial_point["P0"]:
            raise ValueError("Initial point violates pressure ordering: require PI <= P0")
        if self.initial_point["Pl"] > self.initial_point["PI"]:
            raise ValueError("Initial point violates pressure ordering: require Pl <= PI")