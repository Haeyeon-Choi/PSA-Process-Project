"""Direct PSA optimization using Pyomo trust-region LP subproblems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pyomo.environ as pyo

from psa_pyomo.model.column_model import ColumnDecisionSpace, VARIABLE_ORDER
from psa_pyomo.performance.productivity import productivity_value
from psa_pyomo.process.css_constraints import css_constraint_residuals, pressure_ordering_residuals
from psa_pyomo.process.cycle_model import CycleEvaluation, CycleSimulator


@dataclass(frozen=True)
class OptimizationConfig:
    purity_min: float
    recovery_min: float
    css_tol: float
    energy_weight: float
    fd_rel_step: float
    delta0: float
    max_iter: int
    solver_name: str
    iter_log_path: str | None = None


def objective_value(evaluation: CycleEvaluation, energy_weight: float) -> float:
    """Primary objective: maximize productivity (optional energy penalty)."""
    return productivity_value(evaluation) - energy_weight * evaluation.energy



def _log_iteration(config: OptimizationConfig, iteration: int, delta: float, evaluation: CycleEvaluation, accepted: bool) -> None:
    if not config.iter_log_path:
        return
    path = Path(config.iter_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("iter,accepted,delta,objective,purity,recovery,css_error,productivity,energy\n", encoding="utf-8")
    line = (
        f"{iteration},{int(accepted)},{delta:.6g},{objective_value(evaluation, config.energy_weight):.8g},"
        f"{evaluation.purity:.8g},{evaluation.recovery:.8g},{evaluation.css_error:.8g},"
        f"{evaluation.productivity:.8g},{evaluation.energy:.8g}\n"
    )
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line)



def objective_value(evaluation: CycleEvaluation, energy_weight: float) -> float:
    return productivity_value(evaluation) - energy_weight * evaluation.energy


def finite_difference_gradients(
    simulator: CycleSimulator,
    decision_space: ColumnDecisionSpace,
    current_point: Dict[str, float],
    base_eval: CycleEvaluation,
    config: OptimizationConfig,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    base_obj = objective_value(base_eval, config.energy_weight)
    base_g1, base_g2, base_g3 = css_constraint_residuals(base_eval, config.purity_min, config.recovery_min)

    grad_obj: Dict[str, float] = {}
    grad_cons = {"g1": {}, "g2": {}, "g3": {}}

    for name in VARIABLE_ORDER:
        lb, ub = decision_space.bounds[name]
        step = max(config.fd_rel_step * max(abs(current_point[name]), 1.0), 1e-8)
        forward_ok = current_point[name] + step <= ub
        backward_ok = current_point[name] - step >= lb

        x_fwd = dict(current_point)
        x_bwd = dict(current_point)

        if forward_ok and backward_ok:
            x_fwd[name] = current_point[name] + step
            x_bwd[name] = current_point[name] - step
            eval_fwd = simulator.evaluate(x_fwd)
            eval_bwd = simulator.evaluate(x_bwd)

            obj_grad = (objective_value(eval_fwd, config.energy_weight) - objective_value(eval_bwd, config.energy_weight)) / (2.0 * step)
            g1_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[0] - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[0]) / (2.0 * step)
            g2_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[1] - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[1]) / (2.0 * step)
            g3_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[2] - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[2]) / (2.0 * step)
        elif forward_ok:
            x_fwd[name] = current_point[name] + step
            eval_fwd = simulator.evaluate(x_fwd)
            obj_grad = (objective_value(eval_fwd, config.energy_weight) - base_obj) / step
            g1_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[0] - base_g1) / step
            g2_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[1] - base_g2) / step
            g3_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[2] - base_g3) / step
        elif backward_ok:
            x_bwd[name] = current_point[name] - step
            eval_bwd = simulator.evaluate(x_bwd)
            obj_grad = (base_obj - objective_value(eval_bwd, config.energy_weight)) / step
            g1_grad = (base_g1 - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[0]) / step
            g2_grad = (base_g2 - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[1]) / step
            g3_grad = (base_g3 - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[2]) / step
        else:
            obj_grad = 0.0
            g1_grad = 0.0
            g2_grad = 0.0
            g3_grad = 0.0

        grad_obj[name] = obj_grad
        grad_cons["g1"][name] = g1_grad
        grad_cons["g2"][name] = g2_grad
        grad_cons["g3"][name] = g3_grad

    return grad_obj, grad_cons


def solve_trust_region_lp(
    decision_space: ColumnDecisionSpace,
    current_point: Dict[str, float],
    base_eval: CycleEvaluation,
    grad_obj: Dict[str, float],
    grad_cons: Dict[str, Dict[str, float]],
    config: OptimizationConfig,
    delta: float,
) -> Dict[str, float]:
    base_obj = objective_value(base_eval, config.energy_weight)
    base_g1, base_g2, base_g3 = css_constraint_residuals(base_eval, config.purity_min, config.recovery_min)

    m = pyo.ConcreteModel(name="psa_trust_region_lp")
    m.V = pyo.Set(initialize=VARIABLE_ORDER)
    m.x = pyo.Var(m.V)
    m.t = pyo.Var(m.V, domain=pyo.NonNegativeReals)

    for var in VARIABLE_ORDER:
        lb, ub = decision_space.bounds[var]
        m.x[var].setlb(lb)
        m.x[var].setub(ub)

    m.step_pos = pyo.Constraint(m.V, rule=lambda model, v: model.x[v] - current_point[v] <= model.t[v])
    m.step_neg = pyo.Constraint(m.V, rule=lambda model, v: current_point[v] - model.x[v] <= model.t[v])
    m.trust_region = pyo.Constraint(expr=sum(m.t[v] for v in m.V) <= delta)

    m.lin_g1 = pyo.Constraint(expr=base_g1 + sum(grad_cons["g1"][v] * (m.x[v] - current_point[v]) for v in m.V) <= 0.0)
    m.lin_g2 = pyo.Constraint(expr=base_g2 + sum(grad_cons["g2"][v] * (m.x[v] - current_point[v]) for v in m.V) <= 0.0)
    m.lin_css = pyo.Constraint(expr=base_g3 + sum(grad_cons["g3"][v] * (m.x[v] - current_point[v]) for v in m.V) <= config.css_tol)

    m.pressure_pi_p0 = pyo.Constraint(expr=m.x["PI"] - m.x["P0"] <= 0.0)
    m.pressure_pl_pi = pyo.Constraint(expr=m.x["Pl"] - m.x["PI"] <= 0.0)

    m.obj = pyo.Objective(expr=base_obj + sum(grad_obj[v] * (m.x[v] - current_point[v]) for v in m.V), sense=pyo.maximize)

    solver = pyo.SolverFactory(config.solver_name)
    if solver is None or not solver.available(False):
        raise RuntimeError(f"Solver '{config.solver_name}' is not available.")

    result = solver.solve(m, tee=False)
    if result.solver.termination_condition not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        raise RuntimeError(f"LP subproblem failed: {result.solver.termination_condition}")

    return {v: float(pyo.value(m.x[v])) for v in VARIABLE_ORDER}


def run_optimization(
    simulator: CycleSimulator,
    decision_space: ColumnDecisionSpace,
    config: OptimizationConfig,
) -> Tuple[Dict[str, float], CycleEvaluation]:
    current_point = dict(decision_space.initial_point)
    current_eval = simulator.evaluate(current_point)
    best_point, best_eval = dict(current_point), current_eval

    delta = config.delta0
    for iteration in range(1, config.max_iter + 1):
        grad_obj, grad_cons = finite_difference_gradients(simulator, decision_space, current_point, current_eval, config)
        candidate = solve_trust_region_lp(decision_space, current_point, current_eval, grad_obj, grad_cons, config, delta)
        candidate_eval = simulator.evaluate(candidate)

        candidate_obj = objective_value(candidate_eval, config.energy_weight)
        current_obj = objective_value(current_eval, config.energy_weight)
        g1, g2, g3 = css_constraint_residuals(candidate_eval, config.purity_min, config.recovery_min)
        g_pi_p0, g_pl_pi = pressure_ordering_residuals(candidate)

        improved = (
            (g1 <= 0.0)
            and (g2 <= 0.0)
            and (g3 <= config.css_tol)
            and (g_pi_p0 <= 0.0)
            and (g_pl_pi <= 0.0)
            and (candidate_obj >= current_obj)
        )

        if improved:
            current_point, current_eval = candidate, candidate_eval
            delta = min(delta * 1.4, 2.0)
            if candidate_obj > objective_value(best_eval, config.energy_weight):
                best_point, best_eval = dict(candidate), candidate_eval
        else:
            delta = max(delta * 0.5, 1e-3)

        _log_iteration(config, iteration, delta, current_eval, improved)

        print(
            f"iter={iteration:02d} delta={delta:.4f} "
            f"objective={objective_value(current_eval, config.energy_weight):.6f} "
            f"purity={current_eval.purity:.6f} recovery={current_eval.recovery:.6f} css={current_eval.css_error:.2e}"
        )
        if delta <= 1e-3:
            break

    return best_point, best_eval
