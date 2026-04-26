# Optimization for Data Science - The ELM Problem

This folder contains the code for Project 8, "Three Algorithms to solve the ELM Problem".

The code trains a regularized Extreme Learning Machine by solving the quadratic problem

```text
min_W 1/(2N) ||W H - Y||_F^2 + lambda/2 ||W||_F^2
```

where the hidden layer is generated once and then kept fixed.

## Algorithms

- `LDLT`: direct square-root-free factorization.
- `Heavy Ball`: momentum method for strongly convex quadratic functions.
- `Nesterov`: accelerated gradient with the gradient evaluated at the look-ahead point.

The three project algorithms are written directly in Python/NumPy. The code does not use
`np.linalg.cholesky`, `np.linalg.solve`, `scipy.optimize`, or other built-in solvers for the actual project
methods. A NumPy solve is included only as a separate correctness check.

## Files

- `elm_optimization/elm.py`: creates the synthetic data, hidden layer, augmented matrix `H`, and matrices `Q` and `C`.
- `elm_optimization/algorithms.py`: contains LDLT, triangular solves, Power Method, Heavy Ball, and Nesterov.
- `elm_optimization/metrics.py`: objective value, gradient norm, relative error, MSE, and accuracy.
- `run_experiments.py`: runs the numerical experiments and writes CSV files and plots.
- `tests/smoke_test.py`: small check that the algorithms still work.

## Run a Small Check

Install the required packages if needed:

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 tests/smoke_test.py
```

## Run Experiments

Quick version:

```bash
python3 run_experiments.py --suite quick
```

Larger version:

```bash
python3 run_experiments.py --suite full --tol 1e-6
```

The output is written to `results/`:

- `summary.csv`: one row per algorithm and instance.
- `convergence_history.csv`: gradient norms and gaps during the iterative methods.
- `figures/convergence_validation.png`: convergence curves.
- `figures/conditioning_iterations.png`: effect of changing `lambda`.
- `figures/scaling_runtime.png`: runtime as the hidden width changes.
