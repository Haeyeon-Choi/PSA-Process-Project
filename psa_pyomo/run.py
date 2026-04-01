"""Command-line entry point for direct PSA optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from psa_pyomo.model.column_model import ColumnDecisionSpace, VARIABLE_ORDER
from psa_pyomo.process.cycle_model import CycleConfig, CycleSimulator
import random

SCALE = {
    "L": 1.0,
    "ndot": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "tads": 100.0,
    "P0": 1e5,
    "PI": 1e5,
    "Pl": 1e5,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct PSA optimization with Pyomo trust-region LP steps.")
    parser.add_argument("--solver", default="glpk")
    parser.add_argument("--mat-index", type=int, default=16)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--purity-min", type=float, default=0.90)
    parser.add_argument("--recovery-min", type=float, default=0.75)
    parser.add_argument("--css-tol", type=float, default=1e-4)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--fd-rel-step", type=float, default=1e-3)
    parser.add_argument("--delta0", type=float, default=0.25)
    parser.add_argument("--iter-log-path", default="logs/optimization_iterations.csv")
    parser.add_argument("--cache-path", default=".psa_pyomo_cache.jsonl")
    parser.add_argument("--infeasibility-penalty", type=float, default=1e6)

    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--P0", type=float, default=3.5e5)
    parser.add_argument("--ndot", type=float, default=1.0)
    parser.add_argument("--tads", type=float, default=300.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--PI", type=float, default=3.0e5)
    parser.add_argument("--Pl", type=float, default=1.0e4)

    parser.add_argument("--L-lb", type=float, default=1.0)
    parser.add_argument("--L-ub", type=float, default=1.0)
    parser.add_argument("--P0-lb", type=float, default=2.0e5)
    parser.add_argument("--P0-ub", type=float, default=1.0e8)
    parser.add_argument("--ndot-lb", type=float, default=0.1)
    parser.add_argument("--ndot-ub", type=float, default=5.0)
    parser.add_argument("--tads-lb", type=float, default=50.0)
    parser.add_argument("--tads-ub", type=float, default=800.0)
    parser.add_argument("--alpha-lb", type=float, default=0.15)
    parser.add_argument("--alpha-ub", type=float, default=0.40)
    parser.add_argument("--beta-lb", type=float, default=0.10)
    parser.add_argument("--beta-ub", type=float, default=0.40)
    parser.add_argument("--PI-lb", type=float, default=1.0e4)
    parser.add_argument("--PI-ub", type=float, default=1.0e6)
    parser.add_argument("--Pl-lb", type=float, default=1.0e3)
    parser.add_argument("--Pl-ub", type=float, default=1.0e5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from psa_pyomo.optimization.optimize_psa import OptimizationConfig, objective_value, run_optimization

    bounds = {
    k: (args.__dict__[f"{k}_lb"] / SCALE[k],
        args.__dict__[f"{k}_ub"] / SCALE[k])
    for k in VARIABLE_ORDER
}
    
    initial_point = {
    k: args.__dict__[k] / SCALE[k]
    for k in VARIABLE_ORDER
}

    decision_space = ColumnDecisionSpace(bounds=bounds, initial_point=initial_point)
    decision_space.validate()

    optimization_config = OptimizationConfig(
        purity_min=args.purity_min,
        recovery_min=args.recovery_min,
        css_tol=args.css_tol,
        energy_weight=args.energy_weight,
        fd_rel_step=args.fd_rel_step,
        delta0=args.delta0,
        max_iter=args.max_iter,
        solver_name=args.solver,
        iter_log_path=args.iter_log_path,
        infeasibility_penalty=args.infeasibility_penalty,
    )

    cycle_config = CycleConfig(
        material_index=args.mat_index,
        n_grid=args.N,
        css_tol=args.css_tol,
        cache_path=args.cache_path,
    )
    simulator = CycleSimulator(Path(__file__).resolve().parents[1], cycle_config)

    # best_point, best_eval = run_optimization(simulator, decision_space, optimization_config)
    best_point = None
    best_eval = None

    N_START = 20   # 시작점 개수

    for i in range(N_START):

        print(f"\n=== Optimization start {i+1} ===")

        # random initial point 생성
        decision_space.initial_point = {
            "L": random.uniform(bounds["L"][0], bounds["L"][1]),
            "P0": random.uniform(bounds["P0"][0], bounds["P0"][1]),
            "ndot": random.uniform(bounds["ndot"][0], bounds["ndot"][1]),
            "tads": random.uniform(bounds["tads"][0], bounds["tads"][1]),
            "alpha": random.uniform(bounds["alpha"][0], bounds["alpha"][1]),
            "beta": random.uniform(bounds["beta"][0], bounds["beta"][1]),
            "PI": random.uniform(bounds["PI"][0], bounds["PI"][1]),
            "Pl": random.uniform(bounds["Pl"][0], bounds["Pl"][1]),
        }

        # pressure ordering 유지
        if decision_space.initial_point["PI"] > decision_space.initial_point["P0"]:
            decision_space.initial_point["PI"] = decision_space.initial_point["P0"] * 0.5

        if decision_space.initial_point["Pl"] > decision_space.initial_point["PI"]:
            decision_space.initial_point["Pl"] = decision_space.initial_point["PI"] * 0.5

        try:

            point, evaluation = run_optimization(
                simulator,
                decision_space,
                optimization_config,
            )

            if best_eval is None or evaluation.productivity > best_eval.productivity:
                best_point = point
                best_eval = evaluation

        except RuntimeError:
            print("No feasible solution from this start.")

    if best_point is None:
            print("\nNo feasible solution found.")
            return

    print("\n=== Best global solution ===")

    for name in VARIABLE_ORDER:
        print(f"{name:>8} = {best_point[name]:.6g}")
    
    print(f"productivity = {best_eval.productivity:.6g}")
    print(f"purity       = {best_eval.purity:.6g}")
    print(f"recovery     = {best_eval.recovery:.6g}")

    # print("\n=== Best feasible point found ===")
    # for name in VARIABLE_ORDER:
    #     print(f"{name:>8} = {best_point[name] * SCALE[name]:.6g}")
    #     # print(f"{name:>8} = {best_point[name]:.6g}")
    # print(f"productivity = {best_eval.productivity:.6g}")
    # print(f"energy       = {best_eval.energy:.6g}")
    # print(f"purity       = {best_eval.purity:.6g}")
    # print(f"recovery     = {best_eval.recovery:.6g}")
    # print(f"css_error    = {best_eval.css_error:.3e}")
    # print(f"objective    = {objective_value(best_eval, optimization_config.energy_weight):.6g}")


if __name__ == "__main__":
    main()
