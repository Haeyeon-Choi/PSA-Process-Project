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

### Added physical/optimization features

- **Real adsorption physics helpers**
  - Dual-site Langmuir (`model/isotherm.py`)
  - LDF kinetics (`model/kinetics.py`)
- **Binary gas system** in Python backend (`CO2/N2` loading + gas composition updates)
- **Spatial column model** using **z-grid** (`n_grid`)
- **Pressure drop (Ergun)** approximation in each step update
- **Cycle step implementation**: adsorption, blowdown, purge, pressurization
- **CSS constraint** using cycle-to-cycle state distance (`css_error`)
- **Persistent simulation cache** on disk (`.jsonl`)
- **Iteration logging** during optimization (`.csv`)

---

### Core equations

Objective (default behavior: maximize productivity):
\[
\max\; J(x)=productivity(x)-w\cdot energy(x)
\]
(`w=0` by default.)

Constraints:
\[
purity\ge purity_{min},\quad recovery\ge recovery_{min},\quad css\_error\le css_{tol}
\]
\[
PI\le P0,\quad Pl\le PI
\]

Dual-site Langmuir helper:
\[
q^*=\frac{q_{sat,1}b_1p}{1+b_1p}+\frac{q_{sat,2}b_2p}{1+b_2p}
\]

LDF kinetics helper:
\[
\frac{dq}{dt}=k_{ldf}(q^*-q)
\]

---

### Backends

- `--backend python` (default): run implemented Python column/cycle physics.
- `--backend julia`: evaluate points through `scripts/evaluate_psa_point.jl` and `PSASimulator`.

---

### Requirements

```bash
pip install pyomo
```
Install a Pyomo-supported LP solver (for example `glpk`).

If you use `--backend julia`, install Julia and project dependencies as well.

---

### Run example

```bash
python -m psa_pyomo.run \
  --backend python \
  --solver glpk \
  --purity-min 0.90 \
  --recovery-min 0.75 \
  --css-tol 1e-4 \
  --energy-weight 0.0 \
  --cache-path .psa_pyomo_cache.jsonl \
  --iter-log-path logs/optimization_iterations.csv
```
