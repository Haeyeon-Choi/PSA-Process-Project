### Acknowledgment

This project is based on the original PSASimulator framework.

[Original repository](https://github.com/xyin-anl/PSASimulator.jl)

The simulation core is adapted from the original implementation.
Additional modules for optimization and surrogate modeling were developed as part of CHBE 6746 (Data Driven PSE).

---

### PSA Pyomo package layout

```text
psa_pyomo
│
├── model
│   ├── column_model.py
│   ├── isotherm.py
│   ├── kinetics.py
│
├── process
│   ├── cycle_model.py
│   ├── css_constraints.py
│
├── performance
│   ├── purity.py
│   ├── recovery.py
│   ├── productivity.py
│
├── optimization
│   ├── optimize_psa.py
│
└── run.py
```

A Julia bridge script is used for single-point simulator evaluation:
- `scripts/evaluate_psa_point.jl`

Compatibility entry-point:
- `pyomo_psa_optimization.py` (calls `psa_pyomo.run:main`)

---

### Equations currently implemented

#### Optimization objective
\[
\max\; J(x) = productivity(x) - w\cdot energy(x)
\]

#### Performance constraints
\[
purity(x) \ge purity_{min},\quad recovery(x) \ge recovery_{min}
\]

Residual form used in code:
\[
g_1(x)=purity_{min}-purity(x)\le 0,\quad g_2(x)=recovery_{min}-recovery(x)\le 0
\]

#### Pressure-ordering constraints
\[
PI \le P0,\quad Pl \le PI
\]
Residual form:
\[
g_3(x)=PI-P0\le0,\quad g_4(x)=Pl-PI\le0
\]

#### Trust-region LP subproblem
At each iteration around current point \(x_k\):
\[
\max\; \hat{J}(x)=J(x_k)+\nabla J(x_k)^T(x-x_k)
\]
subject to
\[
\hat{g}_i(x)=g_i(x_k)+\nabla g_i(x_k)^T(x-x_k)\le0\quad(i=1,2)
\]
\[
\|x-x_k\|_1\le\Delta
\]
and direct linear pressure-ordering constraints for `PI`, `P0`, and `Pl`.

#### Kinetics/isotherm formulas exposed in Python modules
- Dual-site Langmuir (helper in `model/isotherm.py`):
\[
q^*=\frac{q_{sat,1}b_1p}{1+b_1p}+\frac{q_{sat,2}b_2p}{1+b_2p}
\]
- LDF kinetics (helper in `model/kinetics.py`):
\[
\frac{dq}{dt}=k_{ldf}(q^*-q)
\]

> Full cycle physics (mass/momentum/energy dynamics and step transitions) are still evaluated by `PSASimulator` through the Julia bridge, while Pyomo handles the optimization subproblems.

---

### Requirements

#### Python
```bash
pip install pyomo
```
(Use any LP solver supported by Pyomo, for example `glpk`.)

#### Julia
You need Julia with this project environment and `PSASimulator` available.

---

### Run example

```bash
python -m psa_pyomo.run \
  --solver glpk \
  --mat-index 16 \
  --N 5 \
  --purity-min 0.90 \
  --recovery-min 0.75 \
  --energy-weight 0.01 \
  --P0 3.5e5 --ndot 1.0 --tads 300 --alpha 0.25 --beta 0.25 --PI 1e5 --Pl 1e4
```

Compatibility entry-point:

```bash
python pyomo_psa_optimization.py --solver glpk
```
