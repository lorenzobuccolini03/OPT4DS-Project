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
methods. NumPy built-in routines are included only in the experiment script as separate benchmark references.

## Experiment Scenarios

The testing section uses several ELM training problems instead of only one simple Gaussian dataset:

- `synthetic_well_conditioned`: easy synthetic problems with correlation from `0.1` to `0.9`.
- `synthetic_ill_conditioned`: harder synthetic problems with correlation from `0.1` to `0.9`, a wider hidden layer, and smaller regularization.
- `synthetic_sparse`: partially sparse synthetic problems with zero probability from `0.0` to `0.9`.
- `wine`: a real multiclass classification dataset loaded from scikit-learn.
- `digits`: a real handwritten-digit dataset loaded from scikit-learn.

For every scenario the code builds the same ELM objects: hidden layer, augmented matrix `H`, normal matrix `Q`, matrix `C`, and the regularized output-weight problem.

## Files

- `elm_optimization/elm.py`: creates synthetic/real-data ELM instances, hidden layer, augmented matrix `H`, and matrices `Q` and `C`.
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

- `summary.csv`: one row per method and scenario, including scratch methods and built-in benchmarks.
- `conditioning_summary.csv`: dimensions, lambda, Power Method estimate of `lambda_max(Q)`, `L`, `mu`, and condition number for each `Q`.
- `builtin_benchmark.csv`: NumPy reference rows, kept separate from the project algorithms.
- `library_optimizer_benchmark.csv`: PyTorch momentum and PyTorch Nesterov reference rows.
- `convergence_history.csv`: gradient norms, objective gaps, and relative errors for the iterative methods.
- `figures/*_convergence.png`: main convergence comparison for Heavy Ball, Nesterov, and the PyTorch reference optimizers.
- `figures/*_nesterov_beta_comparison.png`: focused comparison between fixed-beta Nesterov and variable-beta Nesterov.
- `figures/conditioning_overview.png`: comparison of estimated condition numbers.
- `figures/convergence_time_all_cases.png`: bar plot of elapsed time for every method and every tested case.
- `figures/convergence_time_*.png`: per-dataset bar plots of elapsed time.
