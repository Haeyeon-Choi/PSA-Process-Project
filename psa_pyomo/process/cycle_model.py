"""PSA cycle model with Julia PSASimulator bridge and persistent cache."""

from __future__ import annotations

import json
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
    css_error: float = 0.0


@dataclass(frozen=True)
class CycleConfig:
    material_index: int
    n_grid: int
    css_tol: float = 1e-4
    cache_path: str = ".psa_pyomo_cache.jsonl"
    feed_y0: float = 0.15


class CycleSimulator:
    """Evaluate one operating point using either Python cycle physics or Julia bridge."""

    def __init__(self, project_dir: Path, config: CycleConfig):
        self.project_dir = project_dir
        self.config = config
        self._cache: Dict[Tuple[float, ...], CycleEvaluation] = {}
        self._cache_file = (self.project_dir / self.config.cache_path).resolve()
        self._load_persistent_cache()

    def evaluate(self, point: Dict[str, float]) -> CycleEvaluation:
        key = tuple(round(point[name], 10) for name in VARIABLE_ORDER)
        if key in self._cache:
            return self._cache[key]

        evaluation = self._evaluate_with_julia(point)

        self._cache[key] = evaluation
        self._append_cache_entry(key, evaluation)
        return evaluation

    def _load_persistent_cache(self) -> None:
        if not self._cache_file.exists():
            return
        try:
            for line in self._cache_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = tuple(rec["key"])
                self._cache[key] = CycleEvaluation(
                    productivity=float(rec["productivity"]),
                    energy=float(rec["energy"]),
                    purity=float(rec["purity"]),
                    recovery=float(rec["recovery"]),
                    css_error=float(rec.get("css_error", 0.0)),
                )
        except Exception:
            # Ignore malformed cache and continue with in-memory cache only.
            pass

    def _append_cache_entry(self, key: Tuple[float, ...], evaluation: CycleEvaluation) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": list(key),
            "productivity": evaluation.productivity,
            "energy": evaluation.energy,
            "purity": evaluation.purity,
            "recovery": evaluation.recovery,
            "css_error": evaluation.css_error,
        }
        with self._cache_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")

    def _evaluate_with_julia(self, point: Dict[str, float]) -> CycleEvaluation:
        cmd = [
            "julia",
            "--project=.",
            "scripts/evaluate_psa_point.jl",
            str(self.config.material_index),
            str(self.config.n_grid),
            *(str(point[name]) for name in VARIABLE_ORDER),
            "false",
            str(self.config.feed_y0),
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
        return CycleEvaluation(productivity, energy, purity, recovery, css_error=0.0)
