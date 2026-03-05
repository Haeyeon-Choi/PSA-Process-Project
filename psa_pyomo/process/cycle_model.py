"""PSA cycle evaluation against Julia `PSASimulator`."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from psa_pyomo.model.column_model import VARIABLE_ORDER


@dataclass
class CycleEvaluation:
    productivity: float
    energy: float
    purity: float
    recovery: float


@dataclass(frozen=True)
class CycleConfig:
    material_index: int
    n_grid: int


class CycleSimulator:
    """Evaluates one operating point by calling the Julia bridge script."""

    def __init__(self, project_dir: Path, config: CycleConfig):
        self.project_dir = project_dir
        self.config = config
        self._cache: Dict[Tuple[float, ...], CycleEvaluation] = {}

    def evaluate(self, point: Dict[str, float]) -> CycleEvaluation:
        key = tuple(round(point[name], 10) for name in VARIABLE_ORDER)
        if key in self._cache:
            return self._cache[key]

        cmd = [
            "julia",
            "--project=.",
            "scripts/evaluate_psa_point.jl",
            str(self.config.material_index),
            str(self.config.n_grid),
            *(str(point[name]) for name in VARIABLE_ORDER),
            "false",
        ]
        proc = subprocess.run(cmd, cwd=self.project_dir, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "PSA simulation failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {proc.stdout}\n"
                f"stderr: {proc.stderr}"
            )

        line = proc.stdout.strip().splitlines()[-1]
        productivity, energy, purity, recovery = [float(x) for x in line.split(",")]
        evaluation = CycleEvaluation(productivity, energy, purity, recovery)
        self._cache[key] = evaluation
        return evaluation
