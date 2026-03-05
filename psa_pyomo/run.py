"""Command-line entry point for direct PSA optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from psa_pyomo.model.column_model import ColumnDecisionSpace, VARIABLE_ORDER
from psa_pyomo.model.isotherm import IsothermParameters
from psa_pyomo.model.kinetics import KineticsParameters
from psa_pyomo.optimization.optimize_psa import OptimizationConfig, objective_value, run_optimization
from psa_pyomo.process.cycle_model import CycleConfig, CycleSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct PSA optimization with Pyomo trust-region LP steps.")
    parser.add_argument("--solver", default="glpk")
    parser.add_argument("--mat-index", type=int, default=16)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--purity-min", type=float, default=0.90)
    parser.add_argument("--recovery-min", type=float, default=0.75)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--fd-rel-step", type=float, default=1e-3)
    parser.add_argument("--delta0", type=float, default=0.25)

    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--P0", type=float, default=3.5e5)
    parser.add_argument("--ndot", type=float, default=1.0)
    parser.add_argument("--tads", type=float, default=300.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--PI", type=float, default=1.0e5)
    parser.add_argument("--Pl", type=float, default=1.0e4)

    parser.add_argument("--L-lb", type=float, default=1.0)
    parser.add_argument("--L-ub", type=float, default=1.0)
    parser.add_argument("--P0-lb", type=float, default=2.0e5)
    parser.add_argument("--P0-ub", type=float, default=6.0e5)
    parser.add_argument("--ndot-lb", type=float, default=0.5)
    parser.add_argument("--ndot-ub", type=float, default=2.0)
    parser.add_argument("--tads-lb", type=float, default=150.0)
    parser.add_argument("--tads-ub", type=float, default=500.0)
    parser.add_argument("--alpha-lb", type=float, default=0.15)
    parser.add_argument("--alpha-ub", type=float, default=0.40)
    parser.add_argument("--beta-lb", type=float, default=0.10)
    parser.add_argument("--beta-ub", type=float, default=0.40)
    parser.add_argument("--PI-lb", type=float, default=1.0e4)
    parser.add_argument("--PI-ub", type=float, default=2.0e5)
    parser.add_argument("--Pl-lb", type=float, default=1.0e4)
    parser.add_argument("--Pl-ub", type=float, default=1.0e5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _ = IsothermParameters(material_index=args.mat_index)
    _ = KineticsParameters(n_grid=args.N)

    bounds = {
        "L": (args.L_lb, args.L_ub),
        "P0": (args.P0_lb, args.P0_ub),
        "ndot": (args.ndot_lb, args.ndot_ub),
        "tads": (args.tads_lb, args.tads_ub),
        "alpha": (args.alpha_lb, args.alpha_ub),
        "beta": (args.beta_lb, args.beta_ub),
        "PI": (args.PI_lb, args.PI_ub),
        "Pl": (args.Pl_lb, args.Pl_ub),
    }
    initial_point = {
        "L": args.L,
        "P0": args.P0,
        "ndot": args.ndot,
        "tads": args.tads,
        "alpha": args.alpha,
        "beta": args.beta,
        "PI": args.PI,
        "Pl": args.Pl,
    }

    decision_space = ColumnDecisionSpace(bounds=bounds, initial_point=initial_point)
    decision_space.validate()

    optimization_config = OptimizationConfig(
        purity_min=args.purity_min,
        recovery_min=args.recovery_min,
        energy_weight=args.energy_weight,
        fd_rel_step=args.fd_rel_step,
        delta0=args.delta0,
        max_iter=args.max_iter,
        solver_name=args.solver,
    )

    simulator = CycleSimulator(Path(__file__).resolve().parents[1], CycleConfig(args.mat_index, args.N))

    best_point, best_eval = run_optimization(simulator, decision_space, optimization_config)

    print("\n=== Best feasible point found ===")
    for name in VARIABLE_ORDER:
        print(f"{name:>8} = {best_point[name]:.6g}")
    print(f"productivity = {best_eval.productivity:.6g}")
    print(f"energy       = {best_eval.energy:.6g}")
    print(f"purity       = {best_eval.purity:.6g}")
    print(f"recovery     = {best_eval.recovery:.6g}")
    print(f"objective    = {objective_value(best_eval, optimization_config.energy_weight):.6g}")


if __name__ == "__main__":
    main()
