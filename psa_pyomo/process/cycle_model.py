"""PSA cycle model with Python physics backend and Julia bridge option.

Python backend additions in this module:
- Dual-site Langmuir equilibrium loading
- LDF kinetics for CO2/N2
- Binary gas system tracking (CO2/N2)
- Spatial column model on z-grid
- Ergun pressure-drop approximation
- Persistent on-disk simulation cache
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from psa_pyomo.model.column_model import VARIABLE_ORDER
from psa_pyomo.model.isotherm import DualSiteLangmuirParameters, dual_site_langmuir_loading
from psa_pyomo.model.kinetics import LDFParameters, ldf_uptake_rate


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
    backend: str = "python"  # "python" or "julia"
    css_max_iter: int = 30
    css_tol: float = 1e-4
    cache_path: str = ".psa_pyomo_cache.jsonl"


@dataclass
class ColumnState:
    pressure_profile: List[float]
    y_n2_profile: List[float]
    q_n2_profile: List[float]
    q_co2_profile: List[float]


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

        if self.config.backend == "julia":
            evaluation = self._evaluate_with_julia(point)
        elif self.config.backend == "python":
            evaluation = self._evaluate_with_python_cycle(point)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

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

    def _evaluate_with_python_cycle(self, point: Dict[str, float]) -> CycleEvaluation:
        n = max(2, int(self.config.n_grid))
        p_init = point["Pl"]
        y_init = 0.79
        state = ColumnState(
            pressure_profile=[p_init for _ in range(n)],
            y_n2_profile=[y_init for _ in range(n)],
            q_n2_profile=[0.1 for _ in range(n)],
            q_co2_profile=[0.4 for _ in range(n)],
        )
        last_state = self._copy_state(state)

        for _ in range(self.config.css_max_iter):
            last_state = self._copy_state(state)
            state = self._adsorption_step(state, point)
            state = self._blowdown_step(state, point)
            state = self._purge_step(state, point)
            state = self._pressurization_step(state, point)

            css_error = self._state_distance(state, last_state)
            if css_error <= self.config.css_tol:
                break

        purity = max(1e-6, min(0.999999, sum(state.y_n2_profile[-3:]) / min(3, n)))
        feed_n2 = max(1e-8, point["ndot"] * point["tads"] * 0.79)
        product_n2 = max(1e-8, point["ndot"] * point["tads"] * purity * point["alpha"])
        recovery = max(1e-6, min(0.999999, product_n2 / feed_n2))

        cycle_time = point["tads"] * (1.0 + point["alpha"] + point["beta"])
        bed_volume = max(1e-8, math.pi * (0.5**2) * point["L"])
        productivity = product_n2 / (bed_volume * cycle_time)

        avg_dp = max(state.pressure_profile) - min(state.pressure_profile)
        dp_comp = max(0.0, point["P0"] - point["PI"]) + avg_dp
        dp_vac = max(0.0, point["PI"] - point["Pl"]) + 0.2 * avg_dp
        energy = (dp_comp * point["ndot"] * point["alpha"] + dp_vac * point["ndot"] * point["beta"]) / 1e5

        return CycleEvaluation(
            productivity=productivity,
            energy=energy,
            purity=purity,
            recovery=recovery,
            css_error=self._state_distance(state, last_state),
        )

    def _adsorption_step(self, state: ColumnState, point: Dict[str, float]) -> ColumnState:
        return self._step_update(state, point, p_in=point["P0"], y_n2_in=0.79, dt_factor=1.0)

    def _blowdown_step(self, state: ColumnState, point: Dict[str, float]) -> ColumnState:
        return self._step_update(state, point, p_in=point["PI"], y_n2_in=0.65, dt_factor=0.6)

    def _purge_step(self, state: ColumnState, point: Dict[str, float]) -> ColumnState:
        return self._step_update(state, point, p_in=point["Pl"], y_n2_in=0.55, dt_factor=point["beta"])

    def _pressurization_step(self, state: ColumnState, point: Dict[str, float]) -> ColumnState:
        return self._step_update(state, point, p_in=point["P0"], y_n2_in=0.79, dt_factor=point["alpha"])

    def _step_update(self, state: ColumnState, point: Dict[str, float], p_in: float, y_n2_in: float, dt_factor: float) -> ColumnState:
        n = len(state.pressure_profile)
        dz = max(1e-6, point["L"] / n)
        epsilon = 0.37
        dp = 2.0e-3
        mu = 1.8e-5
        rho = 1.2
        area = math.pi * (0.5**2)
        flow = max(1e-8, point["ndot"])
        velocity = flow / max(1e-8, area)

        # Material-style parameters (simple map by material index parity).
        if self.config.material_index % 2 == 0:
            n2_iso = DualSiteLangmuirParameters(1.6, 3e-6, 0.8, 8e-7)
            co2_iso = DualSiteLangmuirParameters(3.2, 8e-6, 1.5, 2e-6)
            kin_n2 = LDFParameters(0.18)
            kin_co2 = LDFParameters(0.10)
        else:
            n2_iso = DualSiteLangmuirParameters(1.4, 2.5e-6, 0.7, 6e-7)
            co2_iso = DualSiteLangmuirParameters(2.8, 7e-6, 1.3, 1.8e-6)
            kin_n2 = LDFParameters(0.16)
            kin_co2 = LDFParameters(0.09)

        dt = max(0.1, point["tads"] / 30.0) * max(0.1, dt_factor)

        p_prof = list(state.pressure_profile)
        y_prof = list(state.y_n2_profile)
        qn2_prof = list(state.q_n2_profile)
        qco2_prof = list(state.q_co2_profile)

        # Inlet boundary.
        p_prof[0] = p_in
        y_prof[0] = y_n2_in

        # Axial march with Ergun pressure-drop and adsorption updates.
        for i in range(1, n):
            dPdz = 150.0 * mu * ((1 - epsilon) ** 2) / (epsilon**3 * dp**2) * velocity
            dPdz += 1.75 * rho * (1 - epsilon) / (epsilon**3 * dp) * velocity**2
            p_prof[i] = max(5e3, p_prof[i - 1] - dPdz * dz)

            # Axial mixing and convective smoothing of composition.
            y_mix = 0.7 * y_prof[i] + 0.3 * y_prof[i - 1]
            y_prof[i] = max(1e-6, min(0.999999, y_mix))

            p_n2 = y_prof[i] * p_prof[i]
            p_co2 = (1.0 - y_prof[i]) * p_prof[i]
            qn2_star = dual_site_langmuir_loading(p_n2, n2_iso)
            qco2_star = dual_site_langmuir_loading(p_co2, co2_iso)

            qn2_prof[i] = max(0.0, qn2_prof[i] + dt * ldf_uptake_rate(qn2_prof[i], qn2_star, kin_n2))
            qco2_prof[i] = max(0.0, qco2_prof[i] + dt * ldf_uptake_rate(qco2_prof[i], qco2_star, kin_co2))

            # Gas composition feedback from relative uptake tendency.
            uptake_bias = qco2_prof[i] / max(1e-8, qco2_prof[i] + qn2_prof[i])
            y_prof[i] = max(1e-6, min(0.999999, y_prof[i] + 0.08 * (0.5 - uptake_bias)))

        return ColumnState(p_prof, y_prof, qn2_prof, qco2_prof)

    @staticmethod
    def _copy_state(state: ColumnState) -> ColumnState:
        return ColumnState(
            pressure_profile=list(state.pressure_profile),
            y_n2_profile=list(state.y_n2_profile),
            q_n2_profile=list(state.q_n2_profile),
            q_co2_profile=list(state.q_co2_profile),
        )

    @staticmethod
    def _state_distance(a: ColumnState, b: ColumnState) -> float:
        max_err = 0.0
        for aa, bb in zip(a.pressure_profile, b.pressure_profile):
            max_err = max(max_err, abs(aa - bb) / max(1.0, abs(aa)))
        for aa, bb in zip(a.y_n2_profile, b.y_n2_profile):
            max_err = max(max_err, abs(aa - bb))
        for aa, bb in zip(a.q_n2_profile, b.q_n2_profile):
            max_err = max(max_err, abs(aa - bb))
        for aa, bb in zip(a.q_co2_profile, b.q_co2_profile):
            max_err = max(max_err, abs(aa - bb))
        return max_err
