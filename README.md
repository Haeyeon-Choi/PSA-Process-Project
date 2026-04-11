# PSA Process Optimization

Pressure Swing Adsorption (PSA) optimization framework for CO2/N2 separation. Combines Julia-based PSA simulation ([PSASimulator.jl](https://github.com/xyin-anl/PSASimulator.jl)) with Python-based optimization (Pyomo + GLPK).

Developed as part of CHBE 6746 (Data-Driven PSE).

---

## Project Structure

```
PSA_project/
├── psa_pyomo/                     # Python optimization framework
│   ├── run.py                     # CLI entry point (multi-start optimization)
│   ├── model/
│   │   └── column_model.py        # Decision variables & bounds
│   ├── process/
│   │   ├── cycle_model.py         # Julia subprocess bridge + JSONL cache
│   │   └── css_constraints.py     # Constraint residuals
│   └── optimization/
│       └── optimize_psa.py        # Trust-region SLP optimizer
│
├── scripts/
│   ├── evaluate_psa_point.jl      # Single-point PSA simulation (Julia)
│   └── multistage_datagen.jl      # Multi-stage dataset generation (Sobol)
│
├── PSASimulator_local/            # Local fork of PSASimulator.jl (y0 configurable)
│
├── data/                          # Generated datasets (CSV)
│   ├── dataset_material{8,13,16}.csv           # Single-stage (2000 samples each)
│   ├── dataset_material{8,13,16}_2stage.csv    # 2-stage
│   └── dataset_material{8,13,16}_3stage.csv    # 3-stage
│
├── psa_pyomo/psa_run.ipynb        # Results notebook
└── logs/                          # Optimization iteration logs
```

## How It Works

```
Python (Pyomo + GLPK)              Julia (PSASimulator)
┌──────────────────────┐           ┌──────────────────────┐
│ Trust-region SLP      │──subprocess──│ 6-step PSA cycle     │
│ optimizer             │←──stdout────│ QNDF stiff ODE solver│
│                       │           │ Dual-site Langmuir    │
│ 1. FD gradients       │           │                       │
│ 2. LP subproblem      │           │ Returns:              │
│ 3. Accept/reject      │           │ productivity, energy, │
│ 4. Update trust-region│           │ purity, recovery      │
└──────────────────────┘           └──────────────────────┘
       + JSONL cache (avoids redundant simulations)
```

## Setup

### Julia
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Python
```bash
conda activate pyomo_env
pip install pyomo
# GLPK solver required
```

## Usage

### Run Optimization
```bash
python -m psa_pyomo.run \
  --mat-index 13 \
  --N 10 \
  --purity-min 0.25 \
  --recovery-min 0.50 \
  --max-iter 20 \
  --solver glpk
```

### Generate Dataset
```bash
# Single-stage, material 13
julia --project=. scripts/multistage_datagen.jl 13 1

# 3-stage, material 8
julia --project=. scripts/multistage_datagen.jl 8 3
```

### Single-Point Simulation
```bash
julia --project=. scripts/evaluate_psa_point.jl \
  13 10 1.0 200000 1.0 200 0.25 0.25 50000 5000 false 0.15
# Output: productivity,energy,purity,recovery
```

## Decision Variables

| Variable | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Column length | L | m | Fixed at 1.0 |
| Feed pressure | P0 | Pa | Pressure during adsorption |
| Feed flow rate | ndot | mol/(m2 s) | Molar flux |
| Adsorption time | tads | s | Duration of adsorption step |
| Pressure equalization | alpha | - | Depressurization fraction |
| Purge parameter | beta | - | Light product purge fraction |
| Intermediate pressure | PI | Pa | Pressure after equalization |
| Low pressure | Pl | Pa | Vacuum pressure during blowdown |

## Optimization Formulation

Objective (maximize productivity):

$$\max\ J(x) = \text{productivity}(x) - w \cdot \text{energy}(x)$$

Constraints:

$$\text{purity}(x) \geq \text{purity}_{\min}, \quad \text{recovery}(x) \geq \text{recovery}_{\min}$$

$$P_I \leq P_0, \quad P_l \leq P_I$$

Trust-region LP subproblem at each iteration:

$$\max\ \hat{J}(x) = J(x_k) + \nabla J(x_k)^T(x - x_k) \quad \text{s.t.} \quad \|x - x_k\|_\infty \leq \Delta$$

## Datasets

9 datasets generated with 2000 Sobol quasi-random samples each:

| Material | 1-stage | 2-stage | 3-stage | Best 3-stage purity |
|----------|---------|---------|---------|---------------------|
| 8 | 2000 | 2000 | 2000 | 96.7% |
| 13 | 2000 | 2000 | 2000 | 97.7% |
| 16 | 2000 | 2000 | 2000 | 62.6% |

Multi-stage PSA cascades each stage's heavy product as the next stage's feed (y0).

### CSV Columns (3-stage example)
```
Inputs:    P0, ndot, tads, alpha, beta, PI, Pl
Stage 1:   purity_s1, recovery_s1, productivity_s1, energy_s1
Stage 2:   purity_s2, recovery_s2, productivity_s2, energy_s2
Stage 3:   purity_s3, recovery_s3, productivity_s3, energy_s3
Summary:   final_purity, final_recovery, overall_recovery
```

## Acknowledgment

Based on the [PSASimulator.jl](https://github.com/xyin-anl/PSASimulator.jl) framework. The simulation core is adapted from the original implementation. Optimization and data generation modules were developed for CHBE 6746 (Data-Driven PSE).
