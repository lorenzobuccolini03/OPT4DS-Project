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

The detailed analysis section is also executed by default. In the `full` suite,
the synthetic epsilon/correlation/sparsity studies use the reference size
requested for the report: 1000 training samples and about 10000 hidden-layer
weights. The dimensionality-scaling studies use a stronger sweep of 10000,
50000, and 100000 hidden-layer weights. The `quick` suite uses a smaller copy of
the same experiments, so it is useful for checking the code without waiting too
long.

You can choose the epsilon values for the detailed sweeps:

```bash
python3 run_experiments.py --suite full --epsilons 1e-3,1e-5,1e-7
```

If you only want the base experiment tables, without the additional analysis
plots, run:

```bash
python3 run_experiments.py --suite quick --skip-detailed-analysis
```

The output is written to `results/`:

- `summary.csv`: one row per method and scenario, including scratch methods and built-in benchmarks.
- `conditioning_summary.csv`: dimensions, lambda, Power Method estimate of `lambda_max(Q)`, `L`, `mu`, and condition number for each `Q`.
- `builtin_benchmark.csv`: NumPy reference rows, kept separate from the project algorithms.
- `library_optimizer_benchmark.csv`: PyTorch momentum and PyTorch Nesterov reference rows.
- `convergence_history.csv`: gradient norms, objective gaps, and relative errors for the iterative methods.
- `detailed_analysis_summary.csv`: extra study table for dimensionality, epsilon, correlation, sparsity, real datasets, built-ins, and beta comparisons.
- `detailed_analysis_history.csv`: convergence histories used by the detailed Nesterov fixed/variable beta plots.
- `figures/*_convergence.png`: main convergence comparison for Heavy Ball, Nesterov, and the PyTorch reference optimizers.
- `figures/*_nesterov_beta_comparison.png`: focused comparison between fixed-beta Nesterov and variable-beta Nesterov.
- `figures/conditioning_overview.png`: comparison of estimated condition numbers.
- `figures/convergence_time_all_cases.png`: bar plot of elapsed time for every method and every tested case.
- `figures/convergence_time_*.png`: per-dataset bar plots of elapsed time.
- `figures/analysis_dimension_scaling_times.png`: runtime comparison between LDLT, Heavy Ball, and Nesterov as the synthetic problem size grows.
- `figures/analysis_synthetic_*_parameter_epsilon_sweep.png`: final gradient norm, objective gap, and relative error while correlation/sparsity and epsilon vary.
- `figures/analysis_synthetic_ldlt_times.png`: LDLT runtime along the synthetic correlation and sparsity sweeps.
- `figures/analysis_builtin_fixed_*_performance.png`: scratch methods against PyTorch and NumPy references for fixed cases.
- `figures/analysis_builtin_fixed_*_times.png`: runtime comparison for the same fixed built-in benchmark cases.
- `figures/analysis_wine_epsilon_sweep.png` and `figures/analysis_digits_epsilon_sweep.png`: Wine/Digits iterative metrics while epsilon varies.
- `figures/analysis_real_ldlt_times.png`: LDLT runtime on Wine and Digits.
- `figures/analysis_beta_fixed_variable_*.png`: fixed-beta Nesterov against variable-beta Nesterov.

The script also writes a second group of files with the prefix `requested_`.
These files follow the extra testing checklist more directly:

- `requested_dimension_scaling_times.csv`: time of our LDLT, Heavy Ball, and Nesterov as hidden-layer weights increase from 10000 to 50000 and 100000 in the `full` suite.
- `requested_dimension_rho_times.csv`: time of our three algorithms for three dimensions and three values of `rho`.
- `requested_dimension_sparsity_times.csv`: time of our three algorithms for three dimensions and three sparseness percentages.
- `requested_library_rho_times.csv`: time comparison between our iterative methods and PyTorch references while dimension and `rho` vary.
- `requested_library_sparsity_times.csv`: same comparison while dimension and sparseness vary.
- `requested_ldlt_builtin_times.csv`: our LDLT against NumPy solve and NumPy Cholesky.
- `requested_real_dataset_times.csv`: time of our LDLT, Heavy Ball, and Nesterov on Wine and Digits.
- `figures/requested_conditioning_rho_*.png`: one two-panel convergence plot for each `rho` from `0.1` to `0.9`.
- `figures/requested_sparsity_*.png`: one two-panel convergence plot for each sparseness value from `0.1` to `0.9`.
- `figures/requested_rho_time_comparison.png`: runtime of our LDLT, Heavy Ball, and Nesterov as `rho` changes.
- `figures/requested_library_*.png`: PyTorch-vs-scratch iterative comparisons.
- `figures/requested_beta_*.png`: fixed-beta Nesterov against variable-beta Nesterov.
- `figures/requested_real_*_scratch_iterative.png`: Wine/Digits convergence plots for our Heavy Ball and Nesterov.
