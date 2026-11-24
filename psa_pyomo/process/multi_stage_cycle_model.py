"""Multi-stage PSA cycle wrapper.

Chains N single-stage PSA simulations sequentially. Stage k's output purity becomes
stage k+1's feed composition (y0). The 8 operating variables are shared across all
stages (stage-shared design).

Aggregation into a CycleEvaluation-compatible result:
    productivity = last stage productivity
    energy       = sum of per-stage energy
    purity       = last stage purity (final system output, concentration-replacement)
    recovery     = product of per-stage recoveries (cumulative, mass-balance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from psa_pyomo.process.cycle_model import CycleEvaluation, CycleSimulator


@dataclass
class MultiStageResult:
    aggregated: CycleEvaluation
    stages: List[CycleEvaluation]
    stage_y0: List[float] = field(default_factory=list)


class MultiStageCycleSimulator:
    """Chains N single-stage PSA evaluations; purity of stage k → y0 of stage k+1.

    Exposes the same `.evaluate(point) -> CycleEvaluation` interface as
    `CycleSimulator`, so it is a drop-in replacement for `run_optimization`.
    """

    def __init__(
        self,
        base: CycleSimulator,
        num_stages: int = 3,
        initial_y0: float = 0.15,
    ):
        self.base = base
        self.num_stages = num_stages
        self.initial_y0 = initial_y0

    def evaluate(self, point: Dict[str, float]) -> CycleEvaluation:
        return self.evaluate_with_stages(point).aggregated

    def evaluate_with_stages(self, point: Dict[str, float]) -> MultiStageResult:
        y0 = self.initial_y0
        stages: List[CycleEvaluation] = []
        y0_list: List[float] = []

        for _ in range(self.num_stages):
            ev = self.base.evaluate(point, feed_y0=y0)
            stages.append(ev)
            y0_list.append(y0)
            y0 = ev.purity  # chain to next stage

        aggregated = CycleEvaluation(
            productivity=stages[-1].productivity,
            energy=sum(s.energy for s in stages),
            purity=stages[-1].purity,
            recovery=math.prod(s.recovery for s in stages),
            css_error=max((s.css_error for s in stages), default=0.0),
        )
        return MultiStageResult(aggregated=aggregated, stages=stages, stage_y0=y0_list)
