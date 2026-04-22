# ELM Optimization Project Code

This repository contains executable Python code for Project 8, "Three Algorithms to solve the ELM Problem".
It implements the regularized Extreme Learning Machine training problem from the theoretical report and runs
numerical experiments that compare the required optimization algorithms.

## Implemented Algorithms

- `LDLT`: square-root-free direct factorization of the symmetric positive definite matrix `Q`.
- `Heavy Ball`: Polyak momentum method with the optimal strongly convex quadratic parameters.
- `Nesterov`: strongly-convex accelerated gradient with constant step size and look-ahead gradient.

The algorithms are implemented directly in `elm_optimization/algorithms.py`. NumPy is used for array arithmetic,
but the project algorithms do not call `np.linalg.solve`, Cholesky, SciPy optimizers, or any built-in optimization
solver. The only built-in linear solve appears as an explicitly labeled correctness reference in the experiment runner
and smoke test.

## Project Structure

- `elm_optimization/elm.py`: data generation, hidden-layer construction, augmented ELM formulation, and `Q, C` construction.
- `elm_optimization/algorithms.py`: scratch LDLT, triangular solves, power method, Heavy Ball, and Nesterov.
- `elm_optimization/metrics.py`: objective value, gradient, relative error, MSE, and accuracy.
- `run_experiments.py`: reproducible experiment runner that writes CSV files and plots.
- `tests/smoke_test.py`: small executable validation test.
- `requirements.txt`: minimal Python dependencies.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m pip install -r requirements.txt
```

The current machine already has the required packages installed, so the command may not be necessary.

## Quick Validation

```bash
python3 tests/smoke_test.py
```

This verifies that the scratch LDLT solver matches NumPy's linear solve on a small instance, then checks that
Heavy Ball and Nesterov converge to the LDLT solution.

## Run Experiments

Quick suite:

```bash
python3 run_experiments.py --suite quick
```

Full suite for report-quality tables:

```bash
python3 run_experiments.py --suite full --tol 1e-6
```

Outputs are written to `results/`:

- `summary.csv`: one row per algorithm and instance, including time, iterations, gradient norm, objective gap,
  relative error to LDLT, train/test MSE, and train/test accuracy.
- `convergence_history.csv`: per-iteration history for Heavy Ball and Nesterov.
- `figures/convergence_validation.png`: log-scale gradient norm and objective gap curves.
- `figures/conditioning_iterations.png`: iteration counts as `lambda` changes.
- `figures/scaling_runtime.png`: runtime as hidden width grows.

## Experiment Design

The experiments follow the professor's guidelines:

- every algorithm is implemented in source files rather than a single notebook or one-line library call;
- all algorithms solve the exact same fixed ELM problem within each instance;
- synthetic data are generated on the fly with controlled size and reproducible seeds;
- the report-relevant metrics focus on optimization behavior: objective gap, gradient norm, time, iterations,
  relative error to the direct solution, and scaling with problem size;
- different regularization values are reported as different mathematical problems, not mixed as if they were the same
  optimization instance.
