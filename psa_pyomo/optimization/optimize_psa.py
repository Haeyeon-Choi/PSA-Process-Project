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
from psa_pyomo.run import SCALE

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
    infeasibility_penalty: float = 1e3


def objective_value(evaluation: CycleEvaluation, energy_weight: float) -> float:
    """Primary objective: maximize productivity (optional energy penalty)."""
    return productivity_value(evaluation) - energy_weight * evaluation.energy


def _residuals(
    evaluation: CycleEvaluation,
    point: Dict[str, float],
    config: OptimizationConfig,
) -> Dict[str, float]:
    g1, g2, g3 = css_constraint_residuals(evaluation, config.purity_min, config.recovery_min)
    g_pi_p0, g_pl_pi = pressure_ordering_residuals(point)
    return {
        "purity": g1,
        "recovery": g2,
        "css": g3 - config.css_tol,
        "pi_le_p0": g_pi_p0,
        "pl_le_pi": g_pl_pi,
    }


def _is_feasible(res: Dict[str, float]) -> bool:
    return all(v <= 0.0 for v in res.values())



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

def unscale_point(point):
    return {k: point[k] * SCALE[k] for k in point}

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
        step = max(config.fd_rel_step * max(abs(current_point[name]), 0.1), 1e-6)
        forward_ok = current_point[name] + step <= ub
        backward_ok = current_point[name] - step >= lb

        x_fwd = dict(current_point)
        x_bwd = dict(current_point)

        if forward_ok and backward_ok:
            x_fwd[name] = current_point[name] + step
            x_bwd[name] = current_point[name] - step
            eval_fwd = simulator.evaluate(unscale_point(x_fwd))
            eval_bwd = simulator.evaluate(unscale_point(x_bwd))

            obj_grad = (objective_value(eval_fwd, config.energy_weight) - objective_value(eval_bwd, config.energy_weight)) / (2.0 * step)
            g1_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[0] - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[0]) / (2.0 * step)
            g2_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[1] - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[1]) / (2.0 * step)
            g3_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[2] - css_constraint_residuals(eval_bwd, config.purity_min, config.recovery_min)[2]) / (2.0 * step)
        elif forward_ok:
            x_fwd[name] = current_point[name] + step
            eval_fwd = simulator.evaluate(unscale_point(x_fwd))
            obj_grad = (objective_value(eval_fwd, config.energy_weight) - base_obj) / step
            g1_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[0] - base_g1) / step
            g2_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[1] - base_g2) / step
            g3_grad = (css_constraint_residuals(eval_fwd, config.purity_min, config.recovery_min)[2] - base_g3) / step
        elif backward_ok:
            x_bwd[name] = current_point[name] - step
            eval_bwd = simulator.evaluate(unscale_point(x_bwd))
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
    base_g1, base_g2, base_g3 = css_constraint_residuals(
        base_eval, config.purity_min, config.recovery_min
    )

    m = pyo.ConcreteModel(name="psa_trust_region_lp")

    m.V = pyo.Set(initialize=VARIABLE_ORDER)

    m.x = pyo.Var(m.V)
    m.t = pyo.Var(m.V, domain=pyo.NonNegativeReals)

    # bounds
    for v in VARIABLE_ORDER:
        lb, ub = decision_space.bounds[v]
        m.x[v].setlb(lb)
        m.x[v].setub(ub)

    # step definition
    m.step_pos = pyo.Constraint(
        m.V, rule=lambda model, v: model.x[v] - current_point[v] <= model.t[v]
    )
    m.step_neg = pyo.Constraint(
        m.V, rule=lambda model, v: current_point[v] - model.x[v] <= model.t[v]
    )

    # trust region (L∞ version → more stable)
    m.trust_region = pyo.Constraint(
        m.V, rule=lambda model, v: model.t[v] <= delta
    )

    # slack variables
    m.s_purity = pyo.Var(domain=pyo.NonNegativeReals)
    m.s_recovery = pyo.Var(domain=pyo.NonNegativeReals)
    m.s_css = pyo.Var(domain=pyo.NonNegativeReals)
    m.s_pi_p0 = pyo.Var(domain=pyo.NonNegativeReals)
    m.s_pl_pi = pyo.Var(domain=pyo.NonNegativeReals)

    # linearized constraints
    m.lin_g1 = pyo.Constraint(
        expr=base_g1
        + sum(grad_cons["g1"][v] * (m.x[v] - current_point[v]) for v in m.V)
        <= m.s_purity
    )

    m.lin_g2 = pyo.Constraint(
        expr=base_g2
        + sum(grad_cons["g2"][v] * (m.x[v] - current_point[v]) for v in m.V)
        <= m.s_recovery
    )

    m.lin_css = pyo.Constraint(
        expr=base_g3
        + sum(grad_cons["g3"][v] * (m.x[v] - current_point[v]) for v in m.V)
        <= config.css_tol + m.s_css
    )

    m.pressure_pi_p0 = pyo.Constraint(expr=m.x["PI"] - m.x["P0"] <= m.s_pi_p0)
    m.pressure_pl_pi = pyo.Constraint(expr=m.x["Pl"] - m.x["PI"] <= m.s_pl_pi)

    # linear objective
    linear_obj = base_obj + sum(
        grad_obj[v] * (m.x[v] - current_point[v]) for v in m.V
    )

    # slack penalty
    slack_penalty = config.infeasibility_penalty * (
        m.s_purity
        + m.s_recovery
        + m.s_css
        + m.s_pi_p0
        + m.s_pl_pi
    )

    m.obj = pyo.Objective(expr=linear_obj - slack_penalty, sense=pyo.maximize)

    solver = pyo.SolverFactory(config.solver_name)

    if solver is None or not solver.available(False):
        raise RuntimeError(f"Solver '{config.solver_name}' is not available.")

    result = solver.solve(m, tee=False)

    term = result.solver.termination_condition

    acceptable = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.locallyOptimal,
    }

    # remain current point if solver fails
    if term not in acceptable:
        return dict(current_point)

    return {v: float(pyo.value(m.x[v])) for v in VARIABLE_ORDER}

def run_optimization(
    simulator: CycleSimulator,
    decision_space: ColumnDecisionSpace,
    config: OptimizationConfig,
) -> Tuple[Dict[str, float], CycleEvaluation]:

    current_point = dict(decision_space.initial_point)
    def unscale_point(point):
        return {k: point[k] * SCALE[k] for k in point}

    current_eval = simulator.evaluate(unscale_point(current_point))

    best_point = dict(current_point)
    best_eval = current_eval

    delta = config.delta0

    for iteration in range(1, config.max_iter + 1):

        # compute gradients
        grad_obj, grad_cons = finite_difference_gradients(
            simulator,
            decision_space,
            current_point,
            current_eval,
            config,
        )

        # solve trust-region LP
        candidate = solve_trust_region_lp(
            decision_space,
            current_point,
            current_eval,
            grad_obj,
            grad_cons,
            config,
            delta,
        )

        # evaluate candidate
        try:
            candidate_eval = simulator.evaluate(unscale_point(candidate))
            # candidate_eval = simulator.evaluate(candidate)
        except Exception:
            delta = max(delta * 0.5, 1e-3)
            continue
                
        # objective values
        candidate_obj = objective_value(candidate_eval, config.energy_weight)
        current_obj = objective_value(current_eval, config.energy_weight)

        # constraint residuals
        g1, g2, g3 = css_constraint_residuals(
            candidate_eval,
            config.purity_min,
            config.recovery_min,
        )

        current_g1, current_g2, current_g3 = css_constraint_residuals(
            current_eval,
            config.purity_min,
            config.recovery_min,
        )

        g_pi_p0, g_pl_pi = pressure_ordering_residuals(candidate)

        # violation metrics
        candidate_violation = (
            max(0, g1) +
            max(0, g2) +
            max(0, g3 - config.css_tol)
        )

        current_violation = (
            max(0, current_g1) +
            max(0, current_g2) +
            max(0, current_g3 - config.css_tol)
        )

        # accept rule
        improved = (
            (g_pi_p0 <= 0.0)
            and (g_pl_pi <= 0.0)
            and (
                (candidate_violation < current_violation)
                or
                (candidate_obj > current_obj + 1e-6)
            )
        )

        # trust-region update
        if improved:

            current_point = candidate
            current_eval = candidate_eval

            delta = min(delta * 1.4, 2.0)

            if candidate_obj > objective_value(best_eval, config.energy_weight):
                best_point = dict(candidate)
                best_eval = candidate_eval

        else:

            delta = max(delta * 0.5, 1e-3)

        # logging
        _log_iteration(config, iteration, delta, current_eval, improved)

        print(
            f"iter={iteration:02d} "
            f"delta={delta:.4f} "
            f"objective={objective_value(current_eval, config.energy_weight):.6f} "
            f"purity={current_eval.purity:.6f} "
            f"recovery={current_eval.recovery:.6f} "
            f"css={current_eval.css_error:.2e}"
        )

        # stopping condition
        if delta <= 5e-4:
            break

    # final feasibility check
    g1, g2, g3 = css_constraint_residuals(best_eval, config.purity_min, config.recovery_min)
    g_pi_p0, g_pl_pi = pressure_ordering_residuals(best_point)

    if not (
        (g1 <= 0.0)
        and (g2 <= 0.0)
        and (g3 <= config.css_tol)
        and (g_pi_p0 <= 0.0)
        and (g_pl_pi <= 0.0)
    ):
        raise RuntimeError(
            "No feasible point found for the requested constraints. "
            f"Final residuals: purity={g1:.3e}, recovery={g2:.3e}, css={g3:.3e}, "
            f"pi_le_p0={g_pi_p0:.3e}, pl_le_pi={g_pl_pi:.3e}. "
            "Try relaxing purity/recovery/CSS targets or changing initial point and bounds."
        )

    return best_point, best_eval
