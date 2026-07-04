"""Run the ELM optimization experiments.

The goal of this file is to test the algorithms from the project on several
ELM training problems:

1. well-conditioned synthetic problems with different correlations;
2. ill-conditioned synthetic problems with different correlations;
3. partially sparse synthetic problems with different percentages of zeros;
4. real classification datasets from scikit-learn.

The project algorithms are still the hand-written LDLT factorization, Heavy
Ball, and Nesterov. Built-in routines such as ``np.linalg.solve`` are used only
in the benchmark part of this script, so that we can check the numerical
correctness of our own implementations. PyTorch optimizers are also used as
external library references for momentum and Nesterov.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    from scipy.linalg import ldl as scipy_ldl
    from scipy.linalg import solve_triangular as scipy_solve_triangular
except ImportError:
    scipy_ldl = None
    scipy_solve_triangular = None

# Matplotlib and Fontconfig create cache files. This keeps them outside the
# project folder and avoids warnings on machines where the home cache is locked.
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(os.getenv("TMPDIR", "/tmp")) / "elm_optimization_cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.getenv("TMPDIR", "/tmp")) / "elm_optimization_matplotlib_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from sklearn.datasets import load_digits, load_wine
    from sklearn.model_selection import train_test_split
except ImportError:
    load_digits = None
    load_wine = None
    train_test_split = None

try:
    import torch
except ImportError:
    torch = None

from elm_optimization.algorithms import (
    OptimizationResult,
    estimate_spectral_bounds,
    heavy_ball,
    ldlt_factorize,
    ldlt_solve_weights,
    nesterov_accelerated_gradient,
    solve_with_ldlt,
)
from elm_optimization.elm import (
    ELMInstance,
    apply_sparse_feature_mask,
    augment_hidden_matrix,
    build_hidden_matrix,
    create_elm_instance_from_arrays,
    generate_correlated_classification_data,
    one_hot,
    standardize_train_test,
)
from elm_optimization.metrics import (
    classification_accuracy,
    gradient,
    mean_squared_error,
    objective_value,
    relative_error,
)


SUMMARY_FIELDS = [
    "dataset_name",
    "scenario_type",
    "correlation_strength",
    "zero_probability",
    "method",
    "method_type",
    "beta_rule",
    "condition_number",
    "converged",
    "iterations",
    "time_seconds",
    "initial_gradient_norm",
    "final_gradient_norm",
    "gradient_reduction_factor",
    "objective_value",
    "train_accuracy",
    "test_accuracy",
    "train_mse",
    "test_mse",
    "relative_error_to_reference",
    "relative_error_to_ldlt",
    "q_dimension",
    "lambda_reg",
    "estimated_L",
    "power_method_lambda_max_estimate",
    "mu",
    "activation",
    "hidden_width",
    "total_output_weights",
    "alpha",
    "beta",
]

CONDITIONING_FIELDS = [
    "dataset_name",
    "scenario_type",
    "correlation_strength",
    "zero_probability",
    "scenario_description",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "hidden_width",
    "total_output_weights",
    "q_dimension",
    "lambda_reg",
    "power_method_lambda_max_estimate",
    "estimated_L",
    "mu",
    "estimated_condition_number",
    "power_iterations",
    "exact_lambda_min_diagnostic",
    "exact_lambda_max_diagnostic",
    "exact_condition_number_diagnostic",
]

HISTORY_FIELDS = [
    "dataset_name",
    "scenario_type",
    "correlation_strength",
    "zero_probability",
    "method",
    "iteration",
    "grad_norm",
    "objective_value",
    "objective_gap_to_reference",
    "relative_error_to_reference",
]

DETAILED_ANALYSIS_FIELDS = [
    "analysis_group",
    "dataset_name",
    "scenario_type",
    "sweep_parameter",
    "sweep_value",
    "epsilon",
    "method",
    "method_type",
    "beta_rule",
    "converged",
    "iterations",
    "time_seconds",
    "final_gradient_norm",
    "objective_gap_to_reference",
    "relative_error_to_reference",
    "relative_error_to_ldlt",
    "train_accuracy",
    "test_accuracy",
    "q_dimension",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "hidden_width",
    "total_output_weights",
    "lambda_reg",
    "estimated_L",
    "mu",
    "condition_number",
]

DETAILED_HISTORY_FIELDS = [
    "analysis_group",
    "dataset_name",
    "scenario_type",
    "sweep_parameter",
    "sweep_value",
    "epsilon",
    "method",
    "iteration",
    "grad_norm",
    "objective_gap_to_reference",
    "relative_error_to_reference",
]

SYNTHETIC_PARAMETER_SWEEP_FIELDS = [
    "analysis_type",
    "parameter_name",
    "parameter_value",
    "algorithm",
    "epsilon",
    "output_weight_count",
    "hidden_width",
    "q_dimension",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "lambda_reg",
    "estimated_L",
    "mu",
    "condition_number",
    "converged",
    "iterations",
    "time_seconds",
    "final_gradient_norm",
    "objective_value",
    "objective_gap_to_ldlt",
    "relative_error_to_ldlt",
    "train_accuracy",
    "test_accuracy",
    "alpha",
    "beta",
]

SYNTHETIC_PARAMETER_HISTORY_FIELDS = [
    "analysis_type",
    "parameter_name",
    "parameter_value",
    "method",
    "epsilon",
    "output_weight_count",
    "condition_number",
    "iteration",
    "relative_gap",
    "gradient_norm",
    "objective_gap_to_ldlt",
]

SYNTHETIC_PARAMETER_AGGREGATE_FIELDS = [
    "analysis_type",
    "parameter_name",
    "parameter_value",
    "algorithm",
    "epsilon",
    "number_of_trials",
    "output_weight_count",
    "hidden_width",
    "q_dimension",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "lambda_reg",
    "estimated_L",
    "mu",
    "condition_number",
    "convergence_rate",
    "mean_computational_time_seconds",
    "std_computational_time_seconds",
    "mean_iterations",
    "std_iterations",
    "mean_initial_gradient_norm",
    "std_initial_gradient_norm",
    "mean_final_gradient_norm",
    "std_final_gradient_norm",
    "mean_objective_value",
    "mean_objective_gap_to_ldlt",
    "mean_relative_error_to_ldlt",
    "mean_train_accuracy",
    "mean_test_accuracy",
    "alpha",
    "beta",
]

SYNTHETIC_PARAMETER_TRIAL_FIELDS = [
    "analysis_type",
    "parameter_name",
    "parameter_value",
    "initialization_index",
    "initialization_seed",
    "algorithm",
    "epsilon",
    "output_weight_count",
    "hidden_width",
    "q_dimension",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "lambda_reg",
    "estimated_L",
    "mu",
    "condition_number",
    "converged",
    "iterations",
    "computational_time_seconds",
    "initial_gradient_norm",
    "final_gradient_norm",
    "objective_value",
    "objective_gap_to_ldlt",
    "relative_error_to_ldlt",
    "train_accuracy",
    "test_accuracy",
    "alpha",
    "beta",
]

REQUESTED_DIMENSION_TIME_FIELDS = [
    "algorithm",
    "dimensionality",
    "computational_time",
]

HANDWRITTEN_DIMENSION_SCALING_FIELDS = [
    "algorithm",
    "output_weight_count",
    "hidden_width",
    "q_dimension",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "epsilon",
    "converged",
    "iterations",
    "computational_time_seconds",
    "final_gradient_norm",
    "alpha",
    "beta",
]

REQUESTED_RHO_TIME_FIELDS = [
    "algorithm",
    "dimensionality",
    "rho",
    "computational_time",
]

REQUESTED_SPARSITY_TIME_FIELDS = [
    "algorithm",
    "dimensionality",
    "sparseness_percentage",
    "computational_time",
]

REQUESTED_LDLT_BUILTIN_FIELDS = [
    "algorithm",
    "dimensionality",
    "computational_time",
    "relative_error_to_our_ldlt",
]

REQUESTED_REAL_TIME_FIELDS = [
    "dataset",
    "algorithm",
    "computational_time",
]

REQUESTED_ITERATIVE_HISTORY_FIELDS = [
    "method",
    "iteration",
    "gradient_norm",
    "relative_gap",
    "parameter_name",
    "parameter_value",
    "dimensionality",
    "epsilon",
]

REAL_PYTORCH_COMPARISON_FIELDS = [
    "algorithm",
    "dataset",
    "epsilon",
    "initial_gradient_norm",
    "final_gradient_norm",
    "absolute_gap",
    "relative_gap",
    "computational_time_seconds",
    "number_of_iterations",
    "hidden_layer_size",
    "alpha",
    "beta",
]

LDLT_SCIPY_COMPARISON_FIELDS = [
    "dataset",
    "case_type",
    "algorithm",
    "hidden_layer_size",
    "output_weight_count",
    "q_dimension",
    "minimum_value",
    "computational_time_seconds",
    "residual_norm",
    "relative_residual_norm",
    "absolute_error_to_known_optimum",
    "relative_error_to_known_optimum",
    "relative_difference_to_our_ldlt",
]

RHO_CONDITIONING_SWEEP_FIELDS = [
    "rho",
    "method",
    "epsilon",
    "output_weight_count",
    "hidden_width",
    "q_dimension",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "lambda_reg",
    "estimated_L",
    "mu",
    "condition_number",
    "alpha",
    "beta",
    "converged",
    "iterations",
    "time_seconds",
    "initial_gradient_norm",
    "final_gradient_norm",
    "objective_value",
    "objective_gap_to_reference",
    "relative_error_to_reference",
    "train_accuracy",
    "test_accuracy",
]

RHO_CONDITIONING_HISTORY_FIELDS = [
    "rho",
    "method",
    "epsilon",
    "output_weight_count",
    "condition_number",
    "iteration",
    "relative_gap",
    "grad_norm",
    "objective_value",
    "objective_gap_to_reference",
    "relative_error_to_reference",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="quick runs fast; full uses larger instances.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=50)
    parser.add_argument(
        "--epsilons",
        default="",
        help=(
            "Comma-separated tolerances for the detailed epsilon sweeps. "
            "If empty, the script chooses simple default values."
        ),
    )
    parser.add_argument(
        "--skip-detailed-analysis",
        action="store_true",
        help="Run only the base experiment tables, without the extra study plots.",
    )
    parser.add_argument(
        "--only-real-pytorch-comparison",
        action="store_true",
        help=(
            "Run only the Wine/Digits comparison between our HB/Nesterov "
            "and PyTorch HB/Nesterov using the same random initial weights."
        ),
    )
    parser.add_argument(
        "--only-ldlt-scipy-comparison",
        "--only-real-ldlt-cholesky-comparison",
        dest="only_ldlt_scipy_comparison",
        action="store_true",
        help=(
            "Run only the LDLT comparison between our implementation and "
            "SciPy LDLT. The old Cholesky option name is still accepted for "
            "compatibility."
        ),
    )
    parser.add_argument(
        "--only-rho-conditioning-sweep",
        action="store_true",
        help=(
            "Run only the rho conditioning experiment with hidden_width=10000 "
            "for our Heavy Ball and Nesterov methods."
        ),
    )
    parser.add_argument(
        "--only-handwritten-dimension-scaling",
        action="store_true",
        help=(
            "Run only the dimensionality comparison between our LDLT, Heavy "
            "Ball, and Nesterov implementations."
        ),
    )
    parser.add_argument(
        "--only-synthetic-parameter-sweeps",
        action="store_true",
        help=(
            "Run only the new synthetic sparsity and rho sweeps with our "
            "hand-written LDLT, Heavy Ball, and Nesterov methods."
        ),
    )
    parser.add_argument(
        "--only-synthetic-rho-converged-plots",
        action="store_true",
        help=(
            "Regenerate only the synthetic rho convergence plots, selecting "
            "a plot run where both Heavy Ball and Nesterov converge."
        ),
    )
    parser.add_argument(
        "--only-nesterov-beta-size-comparison",
        action="store_true",
        help=(
            "Run only the fixed-beta versus variable-beta Nesterov comparison "
            "for three larger hidden layer sizes."
        ),
    )
    parser.add_argument(
        "--only-nesterov-beta-focused-plots",
        action="store_true",
        help=(
            "Run only the 2x2 fixed-beta versus variable-beta Nesterov plots "
            "for hidden widths 100 and 500 at epsilon 1e-3 and 1e-6."
        ),
    )
    parser.add_argument(
        "--synthetic-init-trials",
        type=int,
        default=100,
        help=(
            "Number of random initializations for HB/Nesterov in the "
            "synthetic sparsity/rho sweep."
        ),
    )
    parser.add_argument(
        "--real-comparison-output-weights",
        "--real-comparison-hidden-weights",
        dest="real_comparison_output_weights",
        default="10000,50000,100000",
        help=(
            "Comma-separated target counts for the trainable output weights "
            "in the real-data comparisons. The old option name "
            "--real-comparison-hidden-weights is still accepted for "
            "compatibility."
        ),
    )
    parser.add_argument(
        "--real-comparison-epsilons",
        default="1e-3,1e-5,1e-7",
        help="Comma-separated epsilon values for the real PyTorch comparison.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.only_real_pytorch_comparison:
        rows = run_real_pytorch_same_initialization_comparison(
            output_weight_values=parse_int_list(
                args.real_comparison_output_weights,
            ),
            epsilon_values=parse_float_list(
                args.real_comparison_epsilons,
            ),
            max_iter=choose_max_iter(args.suite, args.max_iter),
            record_every=args.record_every,
            seed=args.seed,
        )
        output_path = output_dir / "real_pytorch_same_initialization.csv"
        write_csv(output_path, REAL_PYTORCH_COMPARISON_FIELDS, rows)
        plot_real_pytorch_time_vs_hidden_size(
            rows,
            figures_dir,
            epsilon=1e-3,
        )
        plot_real_pytorch_time_vs_epsilon_extreme_hidden_sizes(
            rows,
            figures_dir,
        )
        print("Wrote " + str(output_path))
        return

    if args.only_ldlt_scipy_comparison:
        rows = run_ldlt_scipy_comparison(
            output_weight_values=parse_int_list(
                args.real_comparison_output_weights,
            ),
            seed=args.seed,
        )
        output_path = output_dir / "ldlt_scipy_comparison.csv"
        write_csv(output_path, LDLT_SCIPY_COMPARISON_FIELDS, rows)
        remove_old_ldlt_scipy_synthetic_outputs(figures_dir)
        plot_ldlt_scipy_time_bars(rows, figures_dir)
        print("Wrote " + str(output_path))
        return

    if args.only_rho_conditioning_sweep:
        rho_max_iter = choose_rho_sweep_max_iter(args.max_iter)
        summary_rows, history_rows = run_rho_conditioning_sweep(
            seed=args.seed,
            epsilon=args.tol,
            max_iter=rho_max_iter,
            record_every=args.record_every,
        )

        summary_path = output_dir / "rho_conditioning_sweep_summary.csv"
        history_path = output_dir / "rho_conditioning_sweep_history.csv"

        write_csv(
            summary_path,
            RHO_CONDITIONING_SWEEP_FIELDS,
            summary_rows,
        )
        write_csv(
            history_path,
            RHO_CONDITIONING_HISTORY_FIELDS,
            history_rows,
        )

        remove_old_rho_conditioning_outputs(output_dir, figures_dir)
        plot_requested_conditioning_effect(history_rows, summary_rows, figures_dir)

        print("Wrote " + str(summary_path))
        print("Wrote " + str(history_path))
        return

    if args.only_handwritten_dimension_scaling:
        epsilon = 1e-3
        scaling_max_iter = choose_handwritten_scaling_max_iter(args.max_iter)
        rows, scratch_history_rows = run_handwritten_dimension_scaling(
            seed=args.seed,
            epsilon=epsilon,
            max_iter=scaling_max_iter,
            record_every=args.record_every,
        )

        remove_old_dimension_scaling_outputs(output_dir, figures_dir)

        output_path = output_dir / "handwritten_dimension_scaling_times.csv"
        scratch_history_path = output_dir / "requested_real_digits_scratch_iterative.csv"
        write_csv(output_path, HANDWRITTEN_DIMENSION_SCALING_FIELDS, rows)
        write_csv(
            scratch_history_path,
            REQUESTED_ITERATIVE_HISTORY_FIELDS,
            scratch_history_rows,
        )
        plot_handwritten_dimension_scaling_times(rows, figures_dir)
        plot_handwritten_dimension_scaling_histogram(rows, figures_dir)
        plot_requested_real_digits_scratch_iterative_scaling(
            scratch_history_rows,
            epsilon,
            figures_dir / "requested_real_digits_scratch_iterative.png",
        )

        print("Wrote " + str(output_path))
        print("Wrote " + str(scratch_history_path))
        return

    if args.only_synthetic_parameter_sweeps:
        epsilon = args.tol
        sweep_max_iter = choose_synthetic_parameter_sweep_max_iter(args.max_iter)
        initialization_trials = args.synthetic_init_trials
        shared_plot_seed = args.seed + 910000
        remove_old_synthetic_parameter_outputs(output_dir, figures_dir)

        sparsity_aggregate, sparsity_trials, sparsity_plot_history = run_synthetic_parameter_sweep(
            analysis_type="sparsity",
            seed=args.seed + 3000,
            epsilon=epsilon,
            max_iter=sweep_max_iter,
            record_every=args.record_every,
            initialization_trials=initialization_trials,
            shared_plot_seed=shared_plot_seed,
        )
        rho_aggregate, rho_trials, rho_plot_history = run_synthetic_parameter_sweep(
            analysis_type="rho",
            seed=args.seed + 4000,
            epsilon=epsilon,
            max_iter=sweep_max_iter,
            record_every=args.record_every,
            initialization_trials=initialization_trials,
            shared_plot_seed=shared_plot_seed,
        )

        aggregate_rows = []
        aggregate_rows.extend(sparsity_aggregate)
        aggregate_rows.extend(rho_aggregate)

        trial_rows = []
        trial_rows.extend(sparsity_trials)
        trial_rows.extend(rho_trials)

        aggregate_path = output_dir / "synthetic_parameter_sweep_aggregate.csv"
        trial_path = output_dir / "synthetic_parameter_sweep_trials.csv"

        write_csv(
            aggregate_path,
            SYNTHETIC_PARAMETER_AGGREGATE_FIELDS,
            aggregate_rows,
        )
        write_csv(trial_path, SYNTHETIC_PARAMETER_TRIAL_FIELDS, trial_rows)

        plot_synthetic_parameter_convergence_by_value(
            sparsity_plot_history,
            "sparsity",
            shared_plot_seed,
            figures_dir,
        )
        plot_synthetic_parameter_time_histogram(
            aggregate_rows,
            "sparsity",
            figures_dir / "synthetic_sparsity_sweep_time_histogram.png",
        )
        plot_synthetic_parameter_convergence_by_value(
            rho_plot_history,
            "rho",
            shared_plot_seed,
            figures_dir,
        )
        plot_synthetic_parameter_time_histogram(
            aggregate_rows,
            "rho",
            figures_dir / "synthetic_rho_sweep_time_histogram.png",
        )

        print("Wrote " + str(aggregate_path))
        print("Wrote " + str(trial_path))
        return

    if args.only_synthetic_rho_converged_plots:
        epsilon = args.tol
        sweep_max_iter = choose_synthetic_parameter_sweep_max_iter(args.max_iter)
        shared_plot_seed = args.seed + 910000
        rho_plot_history = run_synthetic_rho_converged_plot_history(
            seed=args.seed + 4000,
            epsilon=epsilon,
            max_iter=sweep_max_iter,
            record_every=args.record_every,
            shared_plot_seed=shared_plot_seed,
        )
        remove_paths_matching(
            figures_dir,
            ["synthetic_rho_convergence_*.png"],
        )
        plot_synthetic_parameter_convergence_by_value(
            rho_plot_history,
            "rho",
            shared_plot_seed,
            figures_dir,
        )
        print("Wrote synthetic rho convergence plots to " + str(figures_dir))
        return

    if args.only_nesterov_beta_size_comparison:
        max_iter = choose_max_iter(args.suite, args.max_iter)
        remove_old_nesterov_beta_size_outputs(output_dir, figures_dir)
        rows, history_rows = run_requested_beta_comparisons(
            args.suite,
            args.seed,
            args.tol,
            max_iter,
            args.record_every,
            figures_dir,
        )

        summary_path = output_dir / "nesterov_beta_size_comparison.csv"
        history_path = output_dir / "nesterov_beta_size_comparison_history.csv"
        write_csv(summary_path, DETAILED_ANALYSIS_FIELDS, rows)
        write_csv(history_path, DETAILED_HISTORY_FIELDS, history_rows)
        plot_beta_fixed_variable_analysis(history_rows, figures_dir)

        print("Wrote " + str(summary_path))
        print("Wrote " + str(history_path))
        return

    if args.only_nesterov_beta_focused_plots:
        max_iter = choose_nesterov_beta_focus_max_iter(args.max_iter)
        remove_old_nesterov_beta_focus_outputs(output_dir, figures_dir)
        rows, history_rows = run_nesterov_beta_focused_plots(
            args.suite,
            args.seed,
            max_iter,
            args.record_every,
            figures_dir,
        )

        summary_path = output_dir / "nesterov_beta_focused_comparison.csv"
        history_path = output_dir / "nesterov_beta_focused_comparison_history.csv"
        write_csv(summary_path, DETAILED_ANALYSIS_FIELDS, rows)
        write_csv(history_path, DETAILED_HISTORY_FIELDS, history_rows)

        print("Wrote " + str(summary_path))
        print("Wrote " + str(history_path))
        return

    max_iter = choose_max_iter(args.suite, args.max_iter)
    scenarios = build_scenarios(args.suite, args.seed)

    summary_rows = []
    conditioning_rows = []
    history_rows = []

    for scenario in scenarios:
        dataset_name = scenario["dataset_name"]
        scenario_type = scenario["scenario_type"]
        instance = scenario["instance"]

        print("")
        print("Running " + dataset_name + " / " + scenario_type)
        print("  " + scenario["description"])

        suite_summary, suite_conditioning, suite_history = run_one_scenario(
            scenario,
            args.tol,
            max_iter,
            args.record_every,
            args.seed,
        )

        summary_rows.extend(suite_summary)
        conditioning_rows.append(suite_conditioning)
        history_rows.extend(suite_history)

        print_short_scenario_report(instance, suite_conditioning, suite_summary)

    summary_path = output_dir / "summary.csv"
    conditioning_path = output_dir / "conditioning_summary.csv"
    history_path = output_dir / "convergence_history.csv"
    benchmark_path = output_dir / "builtin_benchmark.csv"
    library_benchmark_path = output_dir / "library_optimizer_benchmark.csv"

    benchmark_rows = []
    library_benchmark_rows = []
    for row in summary_rows:
        if row["method_type"] == "built-in benchmark":
            benchmark_rows.append(row)
        if row["method_type"] == "library optimizer":
            library_benchmark_rows.append(row)

    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(conditioning_path, CONDITIONING_FIELDS, conditioning_rows)
    write_csv(history_path, HISTORY_FIELDS, history_rows)
    write_csv(benchmark_path, SUMMARY_FIELDS, benchmark_rows)
    write_csv(library_benchmark_path, SUMMARY_FIELDS, library_benchmark_rows)

    plot_convergence_by_scenario(history_rows, figures_dir)
    plot_nesterov_beta_comparison(history_rows, figures_dir)
    plot_conditioning_overview(conditioning_rows, figures_dir)
    plot_convergence_time_bars(summary_rows, figures_dir)

    detailed_summary_rows = []
    detailed_history_rows = []
    if not args.skip_detailed_analysis:
        detailed_summary_rows, detailed_history_rows = run_detailed_analysis_suite(
            args.suite,
            args.seed + 1000,
            args.tol,
            args.epsilons,
            max_iter,
            args.record_every,
        )

        detailed_summary_path = output_dir / "detailed_analysis_summary.csv"
        detailed_history_path = output_dir / "detailed_analysis_history.csv"
        write_csv(
            detailed_summary_path,
            DETAILED_ANALYSIS_FIELDS,
            detailed_summary_rows,
        )
        write_csv(
            detailed_history_path,
            DETAILED_HISTORY_FIELDS,
            detailed_history_rows,
        )

        plot_dimension_scaling_times(detailed_summary_rows, figures_dir)
        plot_synthetic_parameter_epsilon_sweeps(
            detailed_summary_rows,
            figures_dir,
        )
        plot_synthetic_ldlt_times(detailed_summary_rows, figures_dir)
        plot_builtin_fixed_comparisons(detailed_summary_rows, figures_dir)
        plot_real_epsilon_sweeps(detailed_summary_rows, figures_dir)
        plot_beta_fixed_variable_analysis(detailed_history_rows, figures_dir)
        run_requested_test_analyses(
            output_dir,
            figures_dir,
            args.suite,
            args.seed + 2000,
            args.tol,
            max_iter,
            args.record_every,
        )

    metadata = {
        "suite": args.suite,
        "seed": args.seed,
        "tol": args.tol,
        "epsilon_sweep_values": parse_epsilon_values(args.suite, args.tol, args.epsilons),
        "max_iter": max_iter,
        "record_every": args.record_every,
        "notes": (
            "LDLT, Heavy Ball, and Nesterov are the hand-written project "
            "algorithms. Built-in routines are used only in the benchmark "
            "section to validate the numerical results. PyTorch SGD with "
            "momentum and PyTorch SGD with Nesterov are external library "
            "optimizer references. Dimension sweeps count the trainable "
            "output weights, computed as hidden_width * n_classes. The "
            "random hidden-layer weights are generated once for each ELM "
            "instance and then kept fixed."
        ),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    validate_results(summary_rows, history_rows)

    print("")
    print("Wrote " + str(summary_path))
    print("Wrote " + str(conditioning_path))
    print("Wrote " + str(history_path))
    print("Wrote " + str(benchmark_path))
    print("Wrote " + str(library_benchmark_path))
    if not args.skip_detailed_analysis:
        print("Wrote " + str(output_dir / "detailed_analysis_summary.csv"))
        print("Wrote " + str(output_dir / "detailed_analysis_history.csv"))
    print("Wrote plots to " + str(figures_dir))


def choose_max_iter(suite, max_iter_override):
    if max_iter_override is not None:
        return max_iter_override
    if suite == "full":
        return 8000
    return 1500


def choose_rho_sweep_max_iter(max_iter_override):
    """Default budget for the large hidden-width rho sweep.

    The hidden layer has 10000 neurons, so each gradient step is much more
    expensive than in the small smoke tests. The user can still override this
    with --max-iter when a longer run is needed.
    """

    if max_iter_override is not None:
        return max_iter_override
    return 500


def choose_handwritten_scaling_max_iter(max_iter_override):
    """Default iteration budget for the handwritten scaling experiment."""

    if max_iter_override is not None:
        return max_iter_override
    return 5000


def choose_synthetic_parameter_sweep_max_iter(max_iter_override):
    """Default budget for the new sparsity/rho synthetic sweeps."""

    if max_iter_override is not None:
        return max_iter_override
    return 5000


def choose_nesterov_beta_focus_max_iter(max_iter_override):
    """Iteration budget for the stricter 1e-6 beta-comparison plots."""

    if max_iter_override is not None:
        return max_iter_override
    return 20000


def build_scenarios(suite, seed):
    """Create all ELM test cases used in the numerical experiments."""

    scenarios = []

    add_well_conditioned_scenarios(scenarios, suite, seed)
    add_ill_conditioned_scenarios(scenarios, suite, seed + 100)
    add_sparse_scenarios(scenarios, suite, seed + 200)
    add_real_wine_scenario(scenarios, suite, seed + 300)
    add_real_digits_scenario(scenarios, suite, seed + 400)

    if torch is None:
        raise ImportError(
            "PyTorch is required for the library Heavy Ball and Nesterov "
            "benchmarks. Install torch or remove those benchmark methods."
        )

    return scenarios


def correlation_values():
    values = []
    for number in range(1, 10):
        values.append(number / 10.0)
    return values


def zero_probability_values():
    values = []
    for number in range(0, 10):
        values.append(number / 10.0)
    return values


def format_decimal_for_name(value):
    text = format(value, ".1f")
    return text.replace(".", "_")


def add_well_conditioned_scenarios(scenarios, suite, seed):
    """Easy cases: increasing correlation and enough regularization.

    This case checks that all algorithms behave correctly when Q is not close
    to singular. Using a correlation sweep makes the baseline more informative.
    """

    if suite == "full":
        n_train = 1000
        n_test = 350
        n_features = 24
        hidden_width = 70
    else:
        n_train = 500
        n_test = 180
        n_features = 18
        hidden_width = 45

    scales = np.ones(n_features)
    values = correlation_values()

    for index in range(len(values)):
        correlation = values[index]
        data = generate_correlated_classification_data(
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            n_classes=3,
            class_sep=2.5,
            noise=0.7,
            correlation_strength=correlation,
            feature_scales=scales,
            seed=seed + index,
        )
        x_train, train_labels, x_test, test_labels = data

        instance = create_elm_instance_from_arrays(
            x_train,
            train_labels,
            x_test,
            test_labels,
            hidden_width=hidden_width,
            lambda_reg=5e-2,
            activation="tanh",
            hidden_scale=0.7,
            seed=seed + index,
            standardize_data=False,
        )

        scenario = {
            "dataset_name": "synthetic_well_conditioned",
            "scenario_type": (
                "well_conditioned_corr_"
                + format_decimal_for_name(correlation)
            ),
            "correlation_strength": correlation,
            "zero_probability": "",
            "description": (
                "Well-conditioned ELM with correlation "
                + format(correlation, ".1f")
                + " and moderate regularization."
            ),
            "instance": instance,
        }
        scenarios.append(scenario)


def add_ill_conditioned_scenarios(scenarios, suite, seed):
    """Difficult cases: correlation sweep, wide hidden layer, small lambda.

    The resulting Q usually has a much larger condition number. This is useful
    for observing the difference between the direct LDLT method and accelerated
    gradient methods on elongated quadratic level sets.
    """

    if suite == "full":
        n_train = 1200
        n_test = 400
        n_features = 34
        hidden_width = 130
    else:
        n_train = 650
        n_test = 220
        n_features = 28
        hidden_width = 90

    scales = np.logspace(0.0, -3.0, n_features)
    values = correlation_values()

    for index in range(len(values)):
        correlation = values[index]
        data = generate_correlated_classification_data(
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            n_classes=3,
            class_sep=1.4,
            noise=0.9,
            correlation_strength=correlation,
            feature_scales=scales,
            seed=seed + index,
        )
        x_train, train_labels, x_test, test_labels = data

        instance = create_elm_instance_from_arrays(
            x_train,
            train_labels,
            x_test,
            test_labels,
            hidden_width=hidden_width,
            lambda_reg=1e-3,
            activation="sigmoid",
            hidden_scale=2.0,
            seed=seed + index,
            standardize_data=False,
        )

        scenario = {
            "dataset_name": "synthetic_ill_conditioned",
            "scenario_type": (
                "ill_conditioned_corr_"
                + format_decimal_for_name(correlation)
            ),
            "correlation_strength": correlation,
            "zero_probability": "",
            "description": (
                "Ill-conditioned ELM with correlation "
                + format(correlation, ".1f")
                + ", small regularization, and a wider hidden layer."
            ),
            "instance": instance,
        }
        scenarios.append(scenario)


def add_sparse_scenarios(scenarios, suite, seed):
    """Partially sparse cases: increasing percentage of zero entries.

    We still use dense NumPy arrays. The point is to test a less standard data
    pattern without changing the ELM formulation or implementing sparse linear
    algebra.
    """

    if suite == "full":
        n_train = 1000
        n_test = 350
        n_features = 48
        hidden_width = 100
    else:
        n_train = 560
        n_test = 200
        n_features = 36
        hidden_width = 70

    scales = np.linspace(1.0, 0.3, n_features)
    values = zero_probability_values()

    for index in range(len(values)):
        zero_probability = values[index]
        data = generate_correlated_classification_data(
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            n_classes=4,
            class_sep=1.8,
            noise=1.0,
            correlation_strength=0.25,
            feature_scales=scales,
            seed=seed + index,
        )
        x_train, train_labels, x_test, test_labels = data
        x_train, x_test = apply_sparse_feature_mask(
            x_train,
            x_test,
            zero_probability=zero_probability,
            seed=seed + 100 + index,
        )

        instance = create_elm_instance_from_arrays(
            x_train,
            train_labels,
            x_test,
            test_labels,
            hidden_width=hidden_width,
            lambda_reg=5e-3,
            activation="relu",
            hidden_scale=0.9,
            seed=seed + index,
            standardize_data=False,
        )

        scenario = {
            "dataset_name": "synthetic_sparse",
            "scenario_type": (
                "partially_sparse_zero_"
                + format_decimal_for_name(zero_probability)
            ),
            "correlation_strength": 0.25,
            "zero_probability": zero_probability,
            "description": (
                "Partially sparse ELM with zero probability "
                + format(zero_probability, ".1f")
                + " before the hidden layer is built."
            ),
            "instance": instance,
        }
        scenarios.append(scenario)


def add_real_wine_scenario(scenarios, suite, seed, hidden_width_override=None):
    """Real dataset case based on the Wine classification dataset.

    A real dataset is included to check that the code is not tuned only to
    artificial Gaussian data. Preprocessing is intentionally simple:
    train/test split followed by standardization inside the ELM builder.
    """

    check_sklearn_available()

    data = load_wine()
    x = data.data.astype(float)
    labels = data.target.astype(int)

    x_train, x_test, train_labels, test_labels = train_test_split(
        x,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    if hidden_width_override is not None:
        hidden_width = hidden_width_override
    elif suite == "full":
        hidden_width = 80
    else:
        hidden_width = 50

    instance = create_elm_instance_from_arrays(
        x_train.T,
        train_labels,
        x_test.T,
        test_labels,
        hidden_width=hidden_width,
        lambda_reg=1e-2,
        activation="tanh",
        hidden_scale=1.0,
        seed=seed,
        standardize_data=True,
    )

    scenario = {
        "dataset_name": "wine",
        "scenario_type": "real_dataset",
        "correlation_strength": "",
        "zero_probability": "",
        "description": (
            "Real multiclass classification data from scikit-learn."
        ),
        "instance": instance,
    }
    scenarios.append(scenario)


def add_real_digits_scenario(scenarios, suite, seed, hidden_width_override=None):
    """Real handwritten-digit dataset.

    Digits is now included in both the quick and the full suite, so all
    algorithms are tested on two real datasets.
    """

    check_sklearn_available()

    data = load_digits()
    x = data.data.astype(float)
    labels = data.target.astype(int)

    x_train, x_test, train_labels, test_labels = train_test_split(
        x,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )

    if hidden_width_override is not None:
        hidden_width = hidden_width_override
    elif suite == "full":
        hidden_width = 120
    else:
        hidden_width = 80

    instance = create_elm_instance_from_arrays(
        x_train.T,
        train_labels,
        x_test.T,
        test_labels,
        hidden_width=hidden_width,
        lambda_reg=1e-2,
        activation="tanh",
        hidden_scale=0.8,
        seed=seed,
        standardize_data=True,
    )

    scenario = {
        "dataset_name": "digits",
        "scenario_type": "real_dataset",
        "correlation_strength": "",
        "zero_probability": "",
        "description": (
            "Real handwritten-digit classification data from scikit-learn."
        ),
        "instance": instance,
    }
    scenarios.append(scenario)


def check_sklearn_available():
    if load_wine is None or load_digits is None or train_test_split is None:
        raise ImportError(
            "scikit-learn is required for the real dataset experiments. "
            "Install it with: python3 -m pip install -r requirements.txt"
        )


def check_scipy_available():
    if scipy_ldl is None or scipy_solve_triangular is None:
        raise ImportError(
            "SciPy is required for the LDLT reference benchmark. "
            "Install it with: python3 -m pip install -r requirements.txt"
        )


def run_one_scenario(scenario, tol, max_iter, record_every, seed):
    instance = scenario["instance"]
    q = instance.q
    c = instance.c

    spectral = estimate_spectral_bounds(
        q,
        instance.lambda_reg,
        seed=seed,
        l_safety_factor=1.01,
    )

    conditioning_row = make_conditioning_row(scenario, instance, spectral)

    def objective_fn(weights):
        return objective_value(
            weights,
            instance.h_train_aug,
            instance.y_train,
            instance.lambda_reg,
        )

    # Built-in NumPy solve is used first only to create a trusted reference for
    # the experiment tables. This does not enter the project algorithms.
    numpy_reference = numpy_solve_reference(q, c)
    reference_weights = numpy_reference.weights
    reference_objective = objective_fn(reference_weights)

    ldlt_result = ldlt_solve_weights(q, c)
    ldlt_weights = ldlt_result.weights

    w0 = np.zeros_like(c)

    hb_result = heavy_ball(
        q,
        c,
        w0,
        mu=spectral.mu,
        l_smooth=spectral.l_smooth,
        tol=tol,
        max_iter=max_iter,
        objective_fn=objective_fn,
        reference_weights=reference_weights,
        record_every=record_every,
    )

    nag_result = nesterov_accelerated_gradient(
        q,
        c,
        w0,
        mu=spectral.mu,
        l_smooth=spectral.l_smooth,
        tol=tol,
        max_iter=max_iter,
        objective_fn=objective_fn,
        reference_weights=reference_weights,
        record_every=record_every,
    )

    nag_variable_result = nesterov_variable_beta(
        q,
        c,
        w0,
        l_smooth=spectral.l_smooth,
        tol=tol,
        max_iter=max_iter,
        objective_fn=objective_fn,
        reference_weights=reference_weights,
        record_every=record_every,
    )

    pytorch_hb_result = pytorch_sgd_reference(
        q,
        c,
        w0,
        method_name="PyTorch SGD momentum",
        alpha=hb_result.alpha,
        beta=hb_result.beta,
        use_nesterov=False,
        tol=tol,
        max_iter=max_iter,
        objective_fn=objective_fn,
        reference_weights=reference_weights,
        record_every=record_every,
    )

    pytorch_nesterov_result = pytorch_sgd_reference(
        q,
        c,
        w0,
        method_name="PyTorch SGD Nesterov",
        alpha=nag_result.alpha,
        beta=nag_result.beta,
        use_nesterov=True,
        tol=tol,
        max_iter=max_iter,
        objective_fn=objective_fn,
        reference_weights=reference_weights,
        record_every=record_every,
    )

    scipy_ldlt = scipy_ldlt_reference(q, c)

    results = [
        ldlt_result,
        hb_result,
        nag_result,
        nag_variable_result,
        pytorch_hb_result,
        pytorch_nesterov_result,
        numpy_reference,
        scipy_ldlt,
    ]

    summary = []
    for result in results:
        row = summarize_result(
            scenario,
            instance,
            result,
            spectral,
            reference_weights,
            ldlt_weights,
            objective_fn,
        )
        summary.append(row)

    history = []
    iterative_results = [
        hb_result,
        nag_result,
        nag_variable_result,
        pytorch_hb_result,
        pytorch_nesterov_result,
    ]
    for result in iterative_results:
        rows = history_rows_for_result(
            scenario,
            result,
            reference_objective,
        )
        history.extend(rows)

    return summary, conditioning_row, history


def parse_epsilon_values(suite, main_tol, text):
    """Return the epsilon values used in the detailed tolerance sweeps."""

    values = []

    if text != "":
        parts = text.split(",")
        for part in parts:
            stripped = part.strip()
            if stripped != "":
                values.append(float(stripped))
    else:
        if suite == "full":
            values = [1e-3, 1e-5, 1e-7]
        else:
            values = [1e-3, 1e-5]

    already_present = False
    for value in values:
        if abs(value - main_tol) <= 1e-15:
            already_present = True

    if not already_present:
        values.append(float(main_tol))

    positive_values = []
    for value in values:
        if value <= 0.0:
            raise ValueError("All epsilon values must be positive.")
        if value not in positive_values:
            positive_values.append(value)

    positive_values.sort(reverse=True)
    return positive_values


def run_detailed_analysis_suite(
    suite,
    seed,
    main_tol,
    epsilon_text,
    max_iter,
    record_every,
):
    """Run the extra analyses requested for the experimental section."""

    epsilons = parse_epsilon_values(suite, main_tol, epsilon_text)

    print("")
    print("Running detailed analysis suite")
    print("  epsilon values: " + ", ".join(format(v, ".0e") for v in epsilons))

    summary_rows = []
    history_rows = []

    rows, histories = run_dimension_scaling_analysis(
        suite,
        seed + 10,
        main_tol,
        max_iter,
        record_every,
    )
    summary_rows.extend(rows)
    history_rows.extend(histories)

    rows, histories = run_synthetic_parameter_epsilon_analysis(
        suite,
        seed + 100,
        epsilons,
        max_iter,
        record_every,
    )
    summary_rows.extend(rows)
    history_rows.extend(histories)

    rows, histories = run_synthetic_builtin_fixed_analysis(
        suite,
        seed + 300,
        main_tol,
        max_iter,
        record_every,
    )
    summary_rows.extend(rows)
    history_rows.extend(histories)

    rows, histories = run_real_dataset_detailed_analysis(
        suite,
        seed + 500,
        epsilons,
        main_tol,
        max_iter,
        record_every,
    )
    summary_rows.extend(rows)
    history_rows.extend(histories)

    rows, histories = run_beta_fixed_variable_analysis(
        suite,
        seed + 700,
        main_tol,
        max_iter,
        record_every,
    )
    summary_rows.extend(rows)
    history_rows.extend(histories)

    return summary_rows, history_rows


def detailed_correlation_values(suite):
    if suite == "full":
        return correlation_values()
    return [0.1, 0.5, 0.9]


def detailed_zero_probability_values(suite):
    if suite == "full":
        return zero_probability_values()
    return [0.0, 0.5, 0.9]


def fixed_dimension_config(suite):
    """Dimension used in the correlation/sparsity and epsilon studies.

    In the full suite we use the requested reference size: 1000 training
    samples and about 10000 output weights. The quick suite uses a
    smaller version so that the code can be checked quickly during development.
    """

    if suite == "full":
        return {
            "n_train": 1000,
            "n_test": 350,
            "n_features": 20,
            "n_classes": 20,
            "target_output_weights": 10000,
        }

    return {
        "n_train": 260,
        "n_test": 100,
        "n_features": 10,
        "n_classes": 6,
        "target_output_weights": 600,
    }


def dimension_scaling_configs(suite):
    """Return dimensions for the scaling test.

    The full suite keeps the input size at 1000 training samples and uses a
    visible output-weight sweep: 10000, 50000, and 100000. The random hidden
    layer is still fixed after it is generated; the algorithms optimize only
    the output matrix.
    """

    configs = []

    if suite == "full":
        levels = [
            (1000, 350, 100, 100, 10000),
            (1000, 350, 100, 100, 50000),
            (1000, 350, 100, 100, 100000),
        ]
    else:
        levels = [
            (180, 70, 10, 6, 300),
            (240, 90, 10, 6, 600),
            (300, 110, 10, 6, 900),
        ]

    for n_train, n_test, n_features, n_classes, total_weights in levels:
        configs.append(
            {
                "n_train": n_train,
                "n_test": n_test,
                "n_features": n_features,
                "n_classes": n_classes,
                "target_output_weights": total_weights,
            }
        )

    return configs


def hidden_width_from_output_weights(target_output_weights, n_classes):
    """Convert a target number of output weights into hidden neurons."""

    width = int(round(float(target_output_weights) / float(n_classes)))
    return max(2, width)


def output_weight_count(instance):
    """Number of trainable weights in the hidden-to-output matrix.

    The hidden-layer weights are random and fixed in an ELM. The optimization
    algorithms update the output matrix, whose main size is
    hidden_width * n_classes. The augmented bias row is still represented by
    q_dimension = hidden_width + 1 in the mathematical problem.
    """

    return instance.hidden_width * instance.n_classes


def create_synthetic_analysis_scenario(
    kind,
    sweep_value,
    dimension_config,
    seed,
):
    """Create one synthetic ELM instance for the detailed analysis."""

    n_train = dimension_config["n_train"]
    n_test = dimension_config["n_test"]
    n_features = dimension_config["n_features"]

    if kind == "well":
        dataset_name = "synthetic_well_conditioned"
        scenario_prefix = "well_corr_"
        correlation = sweep_value
        zero_probability = ""
        n_classes = 3
        class_sep = 2.5
        noise = 0.7
        lambda_reg = 5e-2
        activation = "tanh"
        hidden_scale = 0.7
        feature_scales = np.ones(n_features)
        description = "Detailed well-conditioned synthetic case."
    elif kind == "ill":
        dataset_name = "synthetic_ill_conditioned"
        scenario_prefix = "ill_corr_"
        correlation = sweep_value
        zero_probability = ""
        n_classes = 3
        class_sep = 1.4
        noise = 0.9
        lambda_reg = 1e-3
        activation = "sigmoid"
        hidden_scale = 2.0
        feature_scales = np.logspace(0.0, -3.0, n_features)
        description = "Detailed ill-conditioned synthetic case."
    elif kind == "sparse":
        dataset_name = "synthetic_sparse"
        scenario_prefix = "sparse_zero_"
        correlation = 0.25
        zero_probability = sweep_value
        n_classes = 4
        class_sep = 1.8
        noise = 1.0
        lambda_reg = 5e-3
        activation = "relu"
        hidden_scale = 0.9
        feature_scales = np.linspace(1.0, 0.3, n_features)
        description = "Detailed sparse synthetic case."
    else:
        raise ValueError("Unknown synthetic analysis kind: " + kind)

    if "n_classes" in dimension_config:
        n_classes = dimension_config["n_classes"]

    hidden_width = hidden_width_from_output_weights(
        dimension_config["target_output_weights"],
        n_classes,
    )

    data = generate_correlated_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_classes=n_classes,
        class_sep=class_sep,
        noise=noise,
        correlation_strength=correlation,
        feature_scales=feature_scales,
        seed=seed,
    )
    x_train, train_labels, x_test, test_labels = data
    train_labels, test_labels = force_label_dimension(
        train_labels,
        test_labels,
        n_classes,
    )

    if kind == "sparse":
        x_train, x_test = apply_sparse_feature_mask(
            x_train,
            x_test,
            zero_probability=zero_probability,
            seed=seed + 1000,
        )

    instance = create_elm_instance_from_arrays(
        x_train,
        train_labels,
        x_test,
        test_labels,
        hidden_width=hidden_width,
        lambda_reg=lambda_reg,
        activation=activation,
        hidden_scale=hidden_scale,
        seed=seed,
        standardize_data=False,
    )

    if kind == "sparse":
        parameter_name = "sparsity"
        parameter_value = zero_probability
    else:
        parameter_name = "correlation"
        parameter_value = correlation

    scenario = {
        "dataset_name": dataset_name,
        "scenario_type": scenario_prefix + format_decimal_for_name(sweep_value),
        "correlation_strength": correlation,
        "zero_probability": zero_probability,
        "description": description,
        "instance": instance,
        "sweep_parameter": parameter_name,
        "sweep_value": parameter_value,
    }
    return scenario


def conditioning_effect_rho_values():
    return [0.1, 0.5, 0.9]


def conditioning_effect_dimension_config():
    """Fixed size requested for the conditioning-number experiment."""

    return {
        "n_train": 1000,
        "n_test": 350,
        "n_features": 100,
        "n_classes": 100,
        "target_output_weights": 10000,
    }


def create_conditioning_effect_instance(rho, config, seed):
    """Create the fixed-size ELM problem for one rho value.

    The synthetic features are sampled from an equicorrelated covariance matrix.
    This makes rho directly control the feature correlation and gives a clearer
    conditioning effect in Q.
    """

    n_train = config["n_train"]
    n_test = config["n_test"]
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    hidden_width = hidden_width_from_output_weights(
        config["target_output_weights"],
        n_classes,
    )

    data = generate_equicorrelated_classification_data(
        n_train,
        n_test,
        n_features,
        n_classes,
        rho,
        seed,
    )
    x_train, train_labels, x_test, test_labels = data
    train_labels, test_labels = force_label_dimension(
        train_labels,
        test_labels,
        n_classes,
    )

    return create_elm_instance_from_arrays(
        x_train,
        train_labels,
        x_test,
        test_labels,
        hidden_width=hidden_width,
        lambda_reg=1e-3,
        activation="linear",
        hidden_scale=1.0,
        seed=seed,
        standardize_data=False,
    )


def generate_equicorrelated_classification_data(
    n_train,
    n_test,
    n_features,
    n_classes,
    rho,
    seed,
):
    """Generate synthetic data with approximately rho-correlated features."""

    rng = np.random.default_rng(seed)
    rho = min(max(rho, 0.0), 0.99)

    covariance = (1.0 - rho) * np.eye(n_features)
    covariance = covariance + rho * np.ones((n_features, n_features))
    covariance = covariance + 1e-12 * np.eye(n_features)
    cholesky = np.linalg.cholesky(covariance)

    class_sep = 0.4
    centers = class_sep * rng.normal(size=(n_features, n_classes))

    train_labels = rng.integers(0, n_classes, size=n_train)
    test_labels = rng.integers(0, n_classes, size=n_test)

    train_noise = cholesky @ rng.normal(size=(n_features, n_train))
    test_noise = cholesky @ rng.normal(size=(n_features, n_test))

    x_train = centers[:, train_labels] + train_noise
    x_test = centers[:, test_labels] + test_noise
    x_train, x_test = standardize_train_test(x_train, x_test)
    return x_train, train_labels, x_test, test_labels


def force_label_dimension(train_labels, test_labels, n_classes):
    """Ensure that one_hot creates exactly n_classes output rows."""

    fixed_train_labels = np.array(train_labels, dtype=int, copy=True)
    fixed_test_labels = np.array(test_labels, dtype=int, copy=True)

    if fixed_train_labels.size > 0:
        fixed_train_labels[0] = n_classes - 1
    if fixed_test_labels.size > 0:
        fixed_test_labels[0] = n_classes - 1

    return fixed_train_labels, fixed_test_labels


def run_rho_conditioning_sweep(seed, epsilon, max_iter, record_every):
    """Run our HB and Nesterov for three rho values at fixed size.

    The fixed size is 1000 training examples and 10000 hidden-to-output
    weights. The reference optimum used for the relative gap is computed with
    our hand-written LDLT solver.
    """

    summary_rows = []
    history_rows = []
    rho_values = conditioning_effect_rho_values()
    config = conditioning_effect_dimension_config()
    base_seed = seed + 1000

    for index in range(len(rho_values)):
        rho = rho_values[index]
        print(
            "Running conditioning effect: rho="
            + format(rho, ".1f")
            + ", output weights=10000"
        )

        instance = create_conditioning_effect_instance(
            rho,
            config,
            base_seed,
        )

        spectral = estimate_spectral_bounds(
            instance.q,
            instance.lambda_reg,
            seed=base_seed,
            l_safety_factor=1.01,
        )

        ldlt_result = ldlt_solve_weights(instance.q, instance.c)

        def objective_fn(weights):
            return objective_value(
                weights,
                instance.h_train_aug,
                instance.y_train,
                instance.lambda_reg,
            )

        reference_weights = ldlt_result.weights
        reference_objective = objective_fn(reference_weights)
        w0 = np.zeros_like(instance.c)

        hb_result = heavy_ball(
            instance.q,
            instance.c,
            w0,
            mu=spectral.mu,
            l_smooth=spectral.l_smooth,
            tol=epsilon,
            max_iter=max_iter,
            objective_fn=objective_fn,
            reference_weights=reference_weights,
            record_every=record_every,
        )
        nag_result = nesterov_accelerated_gradient(
            instance.q,
            instance.c,
            w0,
            mu=spectral.mu,
            l_smooth=spectral.l_smooth,
            tol=epsilon,
            max_iter=max_iter,
            objective_fn=objective_fn,
            reference_weights=reference_weights,
            record_every=record_every,
        )

        results = [hb_result, nag_result]
        for result in results:
            summary_rows.append(
                make_rho_conditioning_summary_row(
                    rho,
                    epsilon,
                    instance,
                    spectral,
                    result,
                    reference_weights,
                    reference_objective,
                    objective_fn,
                )
            )
            history_rows.extend(
                make_rho_conditioning_history_rows(
                    rho,
                    epsilon,
                    output_weight_count(instance),
                    spectral.condition_estimate,
                    result,
                    reference_objective,
                )
            )

    return summary_rows, history_rows


def create_rho_conditioning_instance(rho, seed):
    """Create one synthetic ELM instance for the rho sweep.

    The hidden layer is fixed after random generation, as in the usual ELM
    setting. The optimization algorithms update only the output weights.
    """

    n_train = 1000
    n_test = 300
    n_features = 30
    n_classes = 2
    hidden_width = 10000
    lambda_reg = 1e-3
    activation = "linear"
    hidden_scale = 1.0

    feature_scales = np.ones(n_features)
    data = generate_correlated_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_classes=n_classes,
        class_sep=1.6,
        noise=0.9,
        correlation_strength=rho,
        feature_scales=feature_scales,
        seed=seed,
    )
    x_train, train_labels, x_test, test_labels = data

    y_train = one_hot(train_labels, n_classes)
    y_test = one_hot(test_labels, n_classes)

    rng = np.random.default_rng(seed + 10000)
    hidden_weights = rng.normal(size=(hidden_width, n_features))
    hidden_weights = hidden_scale * hidden_weights / np.sqrt(float(n_features))
    # The rho sweep is about conditioning caused by the input correlation.
    # A very small hidden bias avoids hiding that effect behind a large
    # constant component in the hidden matrix.
    hidden_bias = rng.uniform(-0.01, 0.01, size=hidden_width)

    h_train = build_hidden_matrix(
        x_train,
        hidden_weights,
        hidden_bias,
        activation,
    )
    h_test = build_hidden_matrix(
        x_test,
        hidden_weights,
        hidden_bias,
        activation,
    )
    h_train_aug = augment_hidden_matrix(h_train)
    h_test_aug = augment_hidden_matrix(h_test)

    # For this large hidden layer experiment we avoid storing the full primal
    # matrix Q. The iterative algorithms below compute the same gradient from H.
    q = None
    c = (y_train @ h_train_aug.T) / float(n_train)

    return ELMInstance(
        x_train,
        y_train,
        train_labels,
        x_test,
        y_test,
        test_labels,
        hidden_weights,
        hidden_bias,
        h_train_aug,
        h_test_aug,
        q,
        c,
        lambda_reg,
        activation,
    )


def estimate_spectral_bounds_from_h_exact(instance):
    """Estimate L and mu from the smaller Gram matrix H^T H.

    The primal matrix would be Q = H H^T / n + lambda I. Since H has many more
    rows than columns in this experiment, the non-zero eigenvalues of H H^T and
    H^T H are the same. Therefore we can obtain L without forming Q.
    """

    h_aug = instance.h_train_aug
    gram = (h_aug.T @ h_aug) / float(instance.n_train)
    eigenvalues = np.linalg.eigvalsh(gram)
    largest_from_data = float(np.max(eigenvalues))

    mu = instance.lambda_reg
    raw_l = largest_from_data + instance.lambda_reg
    l_smooth = raw_l * 1.01
    condition_number = l_smooth / mu

    return {
        "mu": mu,
        "l_smooth": l_smooth,
        "condition_number": condition_number,
        "raw_largest_eigenvalue_estimate": raw_l,
    }


def reference_weights_from_dual_for_metrics(instance):
    """Reference solution used only to measure gaps and relative errors.

    The algorithms are still our hand-written Heavy Ball and Nesterov methods.
    This solve is not used by the algorithms; it is only a diagnostic reference.
    It uses the equivalent small system based on H^T H, because hidden_width is
    10000 and forming the full Q would be wasteful.
    """

    h_aug = instance.h_train_aug
    n_samples = instance.n_train
    dual_matrix = h_aug.T @ h_aug
    dual_matrix = dual_matrix + (
        float(n_samples) * instance.lambda_reg * np.eye(n_samples)
    )
    solved = np.linalg.solve(dual_matrix, h_aug.T)
    weights = instance.y_train @ solved
    return weights


def elm_gradient_and_objective_from_h(weights, instance):
    h_aug = instance.h_train_aug
    residual = weights @ h_aug - instance.y_train
    grad = (residual @ h_aug.T) / float(instance.n_train)
    grad = grad + instance.lambda_reg * weights

    loss = 0.5 * np.sum(residual * residual) / float(instance.n_train)
    regularization = 0.5 * instance.lambda_reg * np.sum(weights * weights)
    objective = float(loss + regularization)
    return grad, objective


def new_rho_history():
    return {
        "iteration": [],
        "grad_norm": [],
        "objective_value": [],
        "objective_gap_to_reference": [],
        "relative_error_to_reference": [],
    }


def record_rho_history(
    history,
    iteration,
    weights,
    grad_norm,
    objective,
    reference_weights,
    reference_objective,
):
    history["iteration"].append(iteration)
    history["grad_norm"].append(grad_norm)
    history["objective_value"].append(objective)
    history["objective_gap_to_reference"].append(
        max(0.0, objective - reference_objective)
    )
    history["relative_error_to_reference"].append(
        relative_error(weights, reference_weights)
    )


def heavy_ball_from_h_with_history(
    instance,
    w0,
    mu,
    l_smooth,
    tol,
    max_iter,
    reference_weights,
    reference_objective,
    record_every,
):
    sqrt_l = np.sqrt(l_smooth)
    sqrt_mu = np.sqrt(mu)
    alpha = 4.0 / (sqrt_l + sqrt_mu) ** 2
    beta = ((sqrt_l - sqrt_mu) / (sqrt_l + sqrt_mu)) ** 2

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    history = new_rho_history()
    converged = False
    final_grad_norm = np.inf
    final_objective = np.inf
    iterations = 0

    start = perf_counter()
    for iteration in range(max_iter + 1):
        grad, objective = elm_gradient_and_objective_from_h(weights, instance)
        final_grad_norm = float(np.sqrt(np.sum(grad * grad)))
        final_objective = objective

        if iteration % record_every == 0:
            record_rho_history(
                history,
                iteration,
                weights,
                final_grad_norm,
                final_objective,
                reference_weights,
                reference_objective,
            )

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = weights - alpha * grad
        next_weights = next_weights + beta * (weights - previous_weights)
        previous_weights = weights
        weights = next_weights

    elapsed = perf_counter() - start

    if history["iteration"][-1] != iterations:
        record_rho_history(
            history,
            iterations,
            weights,
            final_grad_norm,
            final_objective,
            reference_weights,
            reference_objective,
        )

    return OptimizationResult(
        method="Heavy Ball",
        weights=weights,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
        history=history,
    )


def nesterov_from_h_with_history(
    instance,
    w0,
    mu,
    l_smooth,
    tol,
    max_iter,
    reference_weights,
    reference_objective,
    record_every,
):
    alpha = 1.0 / l_smooth
    beta = (np.sqrt(l_smooth) - np.sqrt(mu)) / (
        np.sqrt(l_smooth) + np.sqrt(mu)
    )

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    final_weights = weights.copy()
    history = new_rho_history()
    converged = False
    final_grad_norm = np.inf
    final_objective = np.inf
    iterations = 0

    start = perf_counter()
    for iteration in range(max_iter + 1):
        evaluation_point = weights + beta * (weights - previous_weights)
        grad, objective = elm_gradient_and_objective_from_h(
            evaluation_point,
            instance,
        )
        final_grad_norm = float(np.sqrt(np.sum(grad * grad)))
        final_objective = objective
        final_weights = evaluation_point

        if iteration % record_every == 0:
            record_rho_history(
                history,
                iteration,
                evaluation_point,
                final_grad_norm,
                final_objective,
                reference_weights,
                reference_objective,
            )

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = evaluation_point - alpha * grad
        previous_weights = weights
        weights = next_weights

    elapsed = perf_counter() - start

    if history["iteration"][-1] != iterations:
        record_rho_history(
            history,
            iterations,
            final_weights,
            final_grad_norm,
            final_objective,
            reference_weights,
            reference_objective,
        )

    return OptimizationResult(
        method="Nesterov",
        weights=final_weights,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
        history=history,
    )


def make_rho_conditioning_summary_row(
    rho,
    epsilon,
    instance,
    spectral,
    result,
    reference_weights,
    reference_objective,
    objective_fn,
):
    final_objective = objective_fn(result.weights)
    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug
    initial_gradient_norm = ""
    if "grad_norm" in result.history and len(result.history["grad_norm"]) > 0:
        initial_gradient_norm = result.history["grad_norm"][0]

    return {
        "rho": rho,
        "method": result.method,
        "epsilon": epsilon,
        "output_weight_count": output_weight_count(instance),
        "hidden_width": instance.hidden_width,
        "q_dimension": instance.q.shape[0],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
        "alpha": result.alpha,
        "beta": result.beta,
        "converged": result.converged,
        "iterations": result.iterations,
        "time_seconds": result.elapsed_seconds,
        "initial_gradient_norm": initial_gradient_norm,
        "final_gradient_norm": result.final_gradient_norm,
        "objective_value": final_objective,
        "objective_gap_to_reference": max(
            0.0,
            final_objective - reference_objective,
        ),
        "relative_error_to_reference": relative_error(
            result.weights,
            reference_weights,
        ),
        "train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
    }


def make_rho_conditioning_history_rows(
    rho,
    epsilon,
    output_weights,
    condition_number,
    result,
    reference_objective,
):
    rows = []
    history = result.history
    denominator = max(1.0, abs(reference_objective))

    for index in range(len(history["iteration"])):
        objective_gap = history["objective"][index] - reference_objective
        objective_gap = max(0.0, objective_gap)
        rows.append(
            {
                "rho": rho,
                "method": result.method,
                "epsilon": epsilon,
                "output_weight_count": output_weights,
                "condition_number": condition_number,
                "iteration": history["iteration"][index],
                "relative_gap": objective_gap / denominator,
                "grad_norm": history["grad_norm"][index],
                "objective_value": history["objective"][index],
                "objective_gap_to_reference": objective_gap,
                "relative_error_to_reference": history["relative_error"][index],
            }
        )

    return rows


def create_dimension_scaling_scenario(config, seed):
    scenario = create_synthetic_analysis_scenario(
        "well",
        0.3,
        config,
        seed,
    )
    total_weights = output_weight_count(scenario["instance"])
    scenario["dataset_name"] = "dimension_scaling"
    scenario["scenario_type"] = "output_weights_" + str(total_weights)
    scenario["description"] = (
        "Well-conditioned scaling case with "
        + str(config["n_train"])
        + " training samples and "
        + str(total_weights)
        + " trainable output weights."
    )
    scenario["sweep_parameter"] = "total_output_weights"
    scenario["sweep_value"] = total_weights
    return scenario


def prepare_analysis_context(scenario, seed):
    """Precompute spectral constants and reference solutions for one case."""

    instance = scenario["instance"]
    q = instance.q
    c = instance.c

    spectral = estimate_spectral_bounds(
        q,
        instance.lambda_reg,
        seed=seed,
        l_safety_factor=1.01,
    )

    def objective_fn(weights):
        return objective_value(
            weights,
            instance.h_train_aug,
            instance.y_train,
            instance.lambda_reg,
        )

    numpy_reference = numpy_solve_reference(q, c)
    reference_weights = numpy_reference.weights
    reference_objective = objective_fn(reference_weights)
    ldlt_result = ldlt_solve_weights(q, c)
    scipy_ldlt = scipy_ldlt_reference(q, c)

    context = {
        "scenario": scenario,
        "instance": instance,
        "spectral": spectral,
        "objective_fn": objective_fn,
        "numpy_reference": numpy_reference,
        "scipy_ldlt": scipy_ldlt,
        "reference_weights": reference_weights,
        "reference_objective": reference_objective,
        "ldlt_result": ldlt_result,
        "ldlt_weights": ldlt_result.weights,
    }
    return context


def run_scratch_iterative_methods(context, epsilon, max_iter, record_every):
    instance = context["instance"]
    spectral = context["spectral"]
    w0 = np.zeros_like(instance.c)

    hb_result = heavy_ball(
        instance.q,
        instance.c,
        w0,
        mu=spectral.mu,
        l_smooth=spectral.l_smooth,
        tol=epsilon,
        max_iter=max_iter,
        objective_fn=context["objective_fn"],
        reference_weights=context["reference_weights"],
        record_every=record_every,
    )

    nag_result = nesterov_accelerated_gradient(
        instance.q,
        instance.c,
        w0,
        mu=spectral.mu,
        l_smooth=spectral.l_smooth,
        tol=epsilon,
        max_iter=max_iter,
        objective_fn=context["objective_fn"],
        reference_weights=context["reference_weights"],
        record_every=record_every,
    )

    return hb_result, nag_result


def run_variable_nesterov_method(context, epsilon, max_iter, record_every):
    instance = context["instance"]
    spectral = context["spectral"]
    w0 = np.zeros_like(instance.c)

    return nesterov_variable_beta(
        instance.q,
        instance.c,
        w0,
        l_smooth=spectral.l_smooth,
        tol=epsilon,
        max_iter=max_iter,
        objective_fn=context["objective_fn"],
        reference_weights=context["reference_weights"],
        record_every=record_every,
    )


def run_library_reference_methods(
    context,
    hb_result,
    nag_result,
    epsilon,
    max_iter,
    record_every,
):
    instance = context["instance"]
    w0 = np.zeros_like(instance.c)

    pytorch_hb_result = pytorch_sgd_reference(
        instance.q,
        instance.c,
        w0,
        method_name="PyTorch SGD momentum",
        alpha=hb_result.alpha,
        beta=hb_result.beta,
        use_nesterov=False,
        tol=epsilon,
        max_iter=max_iter,
        objective_fn=context["objective_fn"],
        reference_weights=context["reference_weights"],
        record_every=record_every,
    )

    pytorch_nag_result = pytorch_sgd_reference(
        instance.q,
        instance.c,
        w0,
        method_name="PyTorch SGD Nesterov",
        alpha=nag_result.alpha,
        beta=nag_result.beta,
        use_nesterov=True,
        tol=epsilon,
        max_iter=max_iter,
        objective_fn=context["objective_fn"],
        reference_weights=context["reference_weights"],
        record_every=record_every,
    )

    return pytorch_hb_result, pytorch_nag_result


def run_dimension_scaling_analysis(
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    """Compare runtime of LDLT and iterative methods as size grows."""

    rows = []
    histories = []
    configs = dimension_scaling_configs(suite)

    for index in range(len(configs)):
        scenario = create_dimension_scaling_scenario(
            configs[index],
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        sweep_value = scenario["sweep_value"]

        append_analysis_result(
            rows,
            histories,
            "dimension_scaling",
            context,
            context["ldlt_result"],
            epsilon,
            "total_output_weights",
            sweep_value,
        )

        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )

        append_analysis_result(
            rows,
            histories,
            "dimension_scaling",
            context,
            hb_result,
            epsilon,
            "total_output_weights",
            sweep_value,
        )
        append_analysis_result(
            rows,
            histories,
            "dimension_scaling",
            context,
            nag_result,
            epsilon,
            "total_output_weights",
            sweep_value,
        )

    return rows, histories


def run_synthetic_parameter_epsilon_analysis(
    suite,
    seed,
    epsilons,
    max_iter,
    record_every,
):
    """Run HB/Nesterov/LDLT while varying correlation or sparsity and epsilon."""

    rows = []
    histories = []
    config = fixed_dimension_config(suite)

    synthetic_kinds = ["well", "ill", "sparse"]

    for kind_index in range(len(synthetic_kinds)):
        kind = synthetic_kinds[kind_index]

        if kind == "sparse":
            values = detailed_zero_probability_values(suite)
            parameter_name = "sparsity"
        else:
            values = detailed_correlation_values(suite)
            parameter_name = "correlation"

        for value_index in range(len(values)):
            value = values[value_index]
            scenario = create_synthetic_analysis_scenario(
                kind,
                value,
                config,
                seed + 100 * kind_index + value_index,
            )
            context = prepare_analysis_context(
                scenario,
                seed + 100 * kind_index + value_index,
            )

            for epsilon in epsilons:
                append_analysis_result(
                    rows,
                    histories,
                    "synthetic_parameter_epsilon",
                    context,
                    context["ldlt_result"],
                    epsilon,
                    parameter_name,
                    value,
                )

                hb_result, nag_result = run_scratch_iterative_methods(
                    context,
                    epsilon,
                    max_iter,
                    record_every,
                )
                append_analysis_result(
                    rows,
                    histories,
                    "synthetic_parameter_epsilon",
                    context,
                    hb_result,
                    epsilon,
                    parameter_name,
                    value,
                )
                append_analysis_result(
                    rows,
                    histories,
                    "synthetic_parameter_epsilon",
                    context,
                    nag_result,
                    epsilon,
                    parameter_name,
                    value,
                )

    return rows, histories


def run_synthetic_builtin_fixed_analysis(
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    """Compare our methods with built-ins on three fixed synthetic cases."""

    rows = []
    histories = []
    config = fixed_dimension_config(suite)
    cases = [
        ("well", 0.1),
        ("ill", 0.9),
        ("sparse", 0.7),
    ]

    for index in range(len(cases)):
        kind, value = cases[index]
        scenario = create_synthetic_analysis_scenario(
            kind,
            value,
            config,
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        append_builtin_comparison_results(
            rows,
            histories,
            "builtin_fixed_synthetic",
            context,
            epsilon,
            scenario["sweep_parameter"],
            scenario["sweep_value"],
            max_iter,
            record_every,
        )

    return rows, histories


def run_real_dataset_detailed_analysis(
    suite,
    seed,
    epsilons,
    main_tol,
    max_iter,
    record_every,
):
    """Run epsilon sweeps and built-in comparisons on Wine and Digits."""

    rows = []
    histories = []
    scenarios = []
    add_real_wine_scenario(scenarios, suite, seed)
    add_real_digits_scenario(scenarios, suite, seed + 1)

    for index in range(len(scenarios)):
        scenario = scenarios[index]
        scenario["sweep_parameter"] = "epsilon"
        scenario["sweep_value"] = ""
        context = prepare_analysis_context(scenario, seed + index)

        for epsilon in epsilons:
            append_analysis_result(
                rows,
                histories,
                "real_epsilon",
                context,
                context["ldlt_result"],
                epsilon,
                "epsilon",
                epsilon,
            )

            hb_result, nag_result = run_scratch_iterative_methods(
                context,
                epsilon,
                max_iter,
                record_every,
            )
            append_analysis_result(
                rows,
                histories,
                "real_epsilon",
                context,
                hb_result,
                epsilon,
                "epsilon",
                epsilon,
            )
            append_analysis_result(
                rows,
                histories,
                "real_epsilon",
                context,
                nag_result,
                epsilon,
                "epsilon",
                epsilon,
            )

        append_builtin_comparison_results(
            rows,
            histories,
            "builtin_fixed_real",
            context,
            main_tol,
            "dataset",
            scenario["dataset_name"],
            max_iter,
            record_every,
        )

    return rows, histories


def run_beta_fixed_variable_analysis(
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    """Compare fixed-beta and variable-beta Nesterov on selected cases."""

    rows = []
    histories = []
    scenarios = []
    hidden_sizes = nesterov_beta_hidden_sizes()
    scenario_seed = seed

    synthetic_cases = [
        ("well", 0.1),
        ("ill", 0.9),
        ("sparse", 0.7),
    ]

    for case_index in range(len(synthetic_cases)):
        kind, sweep_value = synthetic_cases[case_index]
        for hidden_width in hidden_sizes:
            config = nesterov_beta_synthetic_config(hidden_width)
            scenario = create_synthetic_analysis_scenario(
                kind,
                sweep_value,
                config,
                scenario_seed,
            )
            add_hidden_size_to_scenario_name(scenario, hidden_width)
            scenarios.append(scenario)
            scenario_seed += 1

    for hidden_width in hidden_sizes:
        add_real_wine_scenario(scenarios, suite, scenario_seed, hidden_width)
        add_hidden_size_to_scenario_name(scenarios[-1], hidden_width)
        scenario_seed += 1

    for hidden_width in hidden_sizes:
        add_real_digits_scenario(scenarios, suite, scenario_seed, hidden_width)
        add_hidden_size_to_scenario_name(scenarios[-1], hidden_width)
        scenario_seed += 1

    for index in range(len(scenarios)):
        scenario = scenarios[index]
        context = prepare_analysis_context(scenario, seed + index)

        unused_hb, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )
        variable_result = run_variable_nesterov_method(
            context,
            epsilon,
            max_iter,
            record_every,
        )

        append_analysis_result(
            rows,
            histories,
            "beta_fixed_vs_variable",
            context,
            nag_result,
            epsilon,
            "hidden_width",
            context["instance"].hidden_width,
        )
        append_analysis_result(
            rows,
            histories,
            "beta_fixed_vs_variable",
            context,
            variable_result,
            epsilon,
            "hidden_width",
            context["instance"].hidden_width,
        )

    return rows, histories


def append_builtin_comparison_results(
    rows,
    histories,
    analysis_group,
    context,
    epsilon,
    sweep_parameter,
    sweep_value,
    max_iter,
    record_every,
):
    """Append scratch, PyTorch, and NumPy results for one fixed case."""

    hb_result, nag_result = run_scratch_iterative_methods(
        context,
        epsilon,
        max_iter,
        record_every,
    )
    pytorch_hb_result, pytorch_nag_result = run_library_reference_methods(
        context,
        hb_result,
        nag_result,
        epsilon,
        max_iter,
        record_every,
    )

    results = [
        context["ldlt_result"],
        hb_result,
        nag_result,
        pytorch_hb_result,
        pytorch_nag_result,
        context["numpy_reference"],
        context["scipy_ldlt"],
    ]

    for result in results:
        append_analysis_result(
            rows,
            histories,
            analysis_group,
            context,
            result,
            epsilon,
            sweep_parameter,
            sweep_value,
        )


def append_analysis_result(
    rows,
    histories,
    analysis_group,
    context,
    result,
    epsilon,
    sweep_parameter,
    sweep_value,
):
    row = make_detailed_analysis_row(
        analysis_group,
        context,
        result,
        epsilon,
        sweep_parameter,
        sweep_value,
    )
    rows.append(row)

    if is_iterative_method(result.method):
        history = make_detailed_history_rows(
            analysis_group,
            context,
            result,
            epsilon,
            sweep_parameter,
            sweep_value,
        )
        histories.extend(history)


def make_detailed_analysis_row(
    analysis_group,
    context,
    result,
    epsilon,
    sweep_parameter,
    sweep_value,
):
    scenario = context["scenario"]
    instance = context["instance"]
    spectral = context["spectral"]
    objective_fn = context["objective_fn"]

    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug
    objective = objective_fn(result.weights)
    objective_gap = max(0.0, objective - context["reference_objective"])

    grad = gradient(result.weights, instance.q, instance.c)
    final_gradient_norm = float(np.sqrt(np.sum(grad * grad)))

    total_output_weights = output_weight_count(instance)

    row = {
        "analysis_group": analysis_group,
        "dataset_name": scenario["dataset_name"],
        "scenario_type": scenario["scenario_type"],
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
        "epsilon": epsilon,
        "method": result.method,
        "method_type": method_type_for_method(result.method),
        "beta_rule": beta_rule_for_method(result.method),
        "converged": result.converged,
        "iterations": result.iterations,
        "time_seconds": result.elapsed_seconds,
        "final_gradient_norm": final_gradient_norm,
        "objective_gap_to_reference": objective_gap,
        "relative_error_to_reference": relative_error(
            result.weights,
            context["reference_weights"],
        ),
        "relative_error_to_ldlt": relative_error(
            result.weights,
            context["ldlt_weights"],
        ),
        "train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
        "q_dimension": instance.q.shape[0],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "hidden_width": instance.hidden_width,
        "total_output_weights": total_output_weights,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
    }
    return row


def make_detailed_history_rows(
    analysis_group,
    context,
    result,
    epsilon,
    sweep_parameter,
    sweep_value,
):
    scenario = context["scenario"]
    rows = []
    iterations = result.history["iteration"]
    grad_norms = result.history["grad_norm"]
    objectives = result.history["objective"]
    relative_errors = result.history["relative_error"]

    for index in range(len(iterations)):
        gap = max(0.0, objectives[index] - context["reference_objective"])
        row = {
            "analysis_group": analysis_group,
            "dataset_name": scenario["dataset_name"],
            "scenario_type": scenario["scenario_type"],
            "sweep_parameter": sweep_parameter,
            "sweep_value": sweep_value,
            "epsilon": epsilon,
            "method": result.method,
            "iteration": int(iterations[index]),
            "grad_norm": grad_norms[index],
            "objective_gap_to_reference": gap,
            "relative_error_to_reference": relative_errors[index],
        }
        rows.append(row)

    return rows


def method_type_for_method(method):
    if method in ["LDLT", "Heavy Ball", "Nesterov", "Nesterov variable beta"]:
        return "scratch"
    if method in ["NumPy solve", "SciPy LDLT"]:
        return "built-in benchmark"
    if method in ["PyTorch SGD momentum", "PyTorch SGD Nesterov"]:
        return "library optimizer"
    return ""


def run_requested_test_analyses(
    output_dir,
    figures_dir,
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    """Run the exact analysis blocks requested in the attached test plan."""

    print("")
    print("Running requested test analyses")

    fixed_config = requested_fixed_dimension_config(suite)
    dimension_configs = requested_dimension_configs(suite)

    dimension_rows = run_requested_dimension_scaling_times(
        dimension_configs,
        seed + 10,
        epsilon,
        max_iter,
        record_every,
    )
    write_csv(
        output_dir / "requested_dimension_scaling_times.csv",
        REQUESTED_DIMENSION_TIME_FIELDS,
        dimension_rows,
    )
    plot_requested_dimension_scaling_times(
        dimension_rows,
        figures_dir,
    )

    conditioning_rows, conditioning_history = run_requested_conditioning_effect(
        fixed_config,
        seed + 100,
        epsilon,
        max_iter,
        record_every,
    )
    plot_requested_parameter_history(
        conditioning_history,
        "rho",
        "requested_conditioning_rho",
        "Conditioning Effect",
        figures_dir,
        epsilon,
    )
    plot_requested_rho_time_comparison(conditioning_rows, figures_dir)

    rho_dimension_rows = run_requested_dimension_parameter_times(
        "ill",
        "rho",
        requested_small_rho_values(),
        dimension_configs,
        seed + 200,
        epsilon,
        max_iter,
        record_every,
        False,
    )
    write_csv(
        output_dir / "requested_dimension_rho_times.csv",
        REQUESTED_RHO_TIME_FIELDS,
        rho_dimension_rows,
    )

    sparsity_rows, sparsity_history = run_requested_sparsity_effect(
        fixed_config,
        seed + 300,
        epsilon,
        max_iter,
        record_every,
    )
    plot_requested_parameter_history(
        sparsity_history,
        "sparseness_percentage",
        "requested_sparsity",
        "Sparseness Effect",
        figures_dir,
        epsilon,
    )

    sparsity_dimension_rows = run_requested_dimension_parameter_times(
        "sparse",
        "sparseness_percentage",
        requested_small_sparsity_values(),
        dimension_configs,
        seed + 400,
        epsilon,
        max_iter,
        record_every,
        False,
    )
    write_csv(
        output_dir / "requested_dimension_sparsity_times.csv",
        REQUESTED_SPARSITY_TIME_FIELDS,
        sparsity_dimension_rows,
    )

    run_requested_library_plots(
        fixed_config,
        seed + 500,
        epsilon,
        max_iter,
        record_every,
        figures_dir,
    )

    library_rho_rows = run_requested_dimension_parameter_times(
        "ill",
        "rho",
        requested_small_rho_values(),
        dimension_configs,
        seed + 600,
        epsilon,
        max_iter,
        record_every,
        True,
    )
    write_csv(
        output_dir / "requested_library_rho_times.csv",
        REQUESTED_RHO_TIME_FIELDS,
        library_rho_rows,
    )

    library_sparsity_rows = run_requested_dimension_parameter_times(
        "sparse",
        "sparseness_percentage",
        requested_small_sparsity_values(),
        dimension_configs,
        seed + 700,
        epsilon,
        max_iter,
        record_every,
        True,
    )
    write_csv(
        output_dir / "requested_library_sparsity_times.csv",
        REQUESTED_SPARSITY_TIME_FIELDS,
        library_sparsity_rows,
    )

    ldlt_builtin_rows = run_requested_ldlt_builtin_times(
        dimension_configs,
        seed + 800,
    )
    write_csv(
        output_dir / "requested_ldlt_builtin_times.csv",
        REQUESTED_LDLT_BUILTIN_FIELDS,
        ldlt_builtin_rows,
    )

    beta_rows, beta_history_rows = run_requested_beta_comparisons(
        suite,
        seed + 900,
        epsilon,
        max_iter,
        record_every,
        figures_dir,
    )
    write_csv(
        output_dir / "nesterov_beta_size_comparison.csv",
        DETAILED_ANALYSIS_FIELDS,
        beta_rows,
    )
    write_csv(
        output_dir / "nesterov_beta_size_comparison_history.csv",
        DETAILED_HISTORY_FIELDS,
        beta_history_rows,
    )

    real_time_rows = run_requested_real_dataset_analysis(
        suite,
        seed + 1000,
        epsilon,
        max_iter,
        record_every,
        figures_dir,
    )
    write_csv(
        output_dir / "requested_real_dataset_times.csv",
        REQUESTED_REAL_TIME_FIELDS,
        real_time_rows,
    )


def requested_fixed_dimension_config(suite):
    if suite == "full":
        return {
            "n_train": 1000,
            "n_test": 350,
            "n_features": 20,
            "n_classes": 20,
            "target_output_weights": 10000,
        }

    return {
        "n_train": 260,
        "n_test": 100,
        "n_features": 10,
        "n_classes": 6,
        "target_output_weights": 600,
    }


def requested_dimension_configs(suite):
    if suite == "full":
        total_weights = [10000, 50000, 100000]
        n_train = 1000
        n_test = 350
        n_features = 100
        n_classes = 100
    else:
        total_weights = [600, 900, 1200]
        n_train = 260
        n_test = 100
        n_features = 10
        n_classes = 6

    configs = []
    for value in total_weights:
        configs.append(
            {
                "n_train": n_train,
                "n_test": n_test,
                "n_features": n_features,
                "n_classes": n_classes,
                "target_output_weights": value,
            }
        )
    return configs


def nesterov_beta_hidden_sizes():
    """Hidden sizes used in the fixed-beta versus variable-beta comparison."""

    return [100, 250, 500]


def nesterov_beta_focused_hidden_sizes():
    """Hidden sizes used in the compact 2x2 beta plots."""

    return [100, 500]


def nesterov_beta_focused_epsilons():
    """Tolerance values requested for the compact beta plots."""

    return [1e-3, 1e-6]


def nesterov_beta_synthetic_config(hidden_width):
    """Dimension of the synthetic beta-comparison problems.

    The ELM hidden layer is random and fixed. The optimized variable is the
    output matrix, so with 100 classes these three hidden sizes correspond to
    10000, 25000, and 50000 hidden-to-output weights.
    """

    n_classes = 100
    return {
        "n_train": 1000,
        "n_test": 350,
        "n_features": 100,
        "n_classes": n_classes,
        "target_output_weights": hidden_width * n_classes,
    }


def add_hidden_size_to_scenario_name(scenario, hidden_width):
    """Make the hidden size visible in tables and plot file names."""

    scenario["scenario_type"] = (
        scenario["scenario_type"] + "_hidden_" + str(hidden_width)
    )
    scenario["description"] = (
        scenario["description"]
        + " Fixed/variable beta comparison with hidden width "
        + str(hidden_width)
        + "."
    )


def handwritten_dimension_scaling_configs():
    """Output-weight sizes for the Digits handwritten scaling comparison."""

    configs = []
    total_weights = [10000, 25000, 50000, 100000]

    for value in total_weights:
        configs.append(
            {
                "target_output_weights": value,
            }
        )

    return configs


def run_handwritten_dimension_scaling(seed, epsilon, max_iter, record_every):
    """Compare our handwritten methods on Digits as the ELM size increases."""

    check_sklearn_available()
    rows = []
    history_rows = []
    configs = handwritten_dimension_scaling_configs()

    for index in range(len(configs)):
        config = configs[index]
        target_weights = config["target_output_weights"]
        print(
            "Running handwritten dimension scaling on Digits: output weights="
            + str(target_weights)
        )

        instance = create_real_comparison_instance(
            "digits",
            target_weights,
            seed + index,
        )

        # This experiment intentionally uses the full primal matrix Q.
        # No dual formulation is used, because the goal is to show the
        # computational bottleneck of our handwritten LDLT factorization.
        q, c = build_primal_elm_system(instance)

        spectral = estimate_spectral_bounds(
            q,
            instance.lambda_reg,
            seed=seed + index,
            l_safety_factor=1.01,
        )

        reference_weights = reference_weights_for_real_gap(instance)
        reference_objective = objective_value_from_h(
            reference_weights,
            instance,
        )

        def objective_fn(weights):
            return objective_value_from_h(weights, instance)

        ldlt_result = ldlt_solve_weights(q, c)
        w0 = np.zeros_like(c)

        hb_result = heavy_ball(
            q,
            c,
            w0,
            mu=spectral.mu,
            l_smooth=spectral.l_smooth,
            tol=epsilon,
            max_iter=max_iter,
            objective_fn=objective_fn,
            reference_weights=reference_weights,
            record_every=record_every,
        )

        nag_result = nesterov_accelerated_gradient(
            q,
            c,
            w0,
            mu=spectral.mu,
            l_smooth=spectral.l_smooth,
            tol=epsilon,
            max_iter=max_iter,
            objective_fn=objective_fn,
            reference_weights=reference_weights,
            record_every=record_every,
        )

        results = [ldlt_result, hb_result, nag_result]
        for result in results:
            rows.append(
                make_handwritten_dimension_scaling_row(
                    result,
                    instance,
                    epsilon,
                )
            )

        context = {
            "reference_objective": reference_objective,
        }
        iterative_results = [hb_result, nag_result]
        for result in iterative_results:
            history_rows.extend(
                make_requested_history_rows(
                    context,
                    result,
                    "output_weight_count",
                    target_weights,
                    output_weight_count(instance),
                    epsilon,
                )
            )

    return rows, history_rows


def make_handwritten_dimension_scaling_row(result, instance, epsilon):
    if hasattr(instance, "q"):
        q_dimension = instance.q.shape[0]
    else:
        q_dimension = instance.hidden_width + 1

    return {
        "algorithm": result.method,
        "output_weight_count": output_weight_count(instance),
        "hidden_width": instance.hidden_width,
        "q_dimension": q_dimension,
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "epsilon": epsilon,
        "converged": result.converged,
        "iterations": result.iterations,
        "computational_time_seconds": result.elapsed_seconds,
        "final_gradient_norm": result.final_gradient_norm,
        "alpha": value_or_empty(result.alpha),
        "beta": value_or_empty(result.beta),
    }


def value_or_empty(value):
    if value is None:
        return ""
    return value


def synthetic_parameter_sweep_values():
    values = []
    for index in range(1, 10):
        values.append(index / 10.0)
    return values


def synthetic_parameter_histogram_values():
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def synthetic_parameter_sweep_config():
    """Fixed synthetic size for the new rho/sparsity experiments."""

    return {
        "n_train": 1000,
        "n_test": 350,
        "n_features": 100,
        "n_classes": 100,
        "target_output_weights": 50000,
    }


def run_synthetic_parameter_sweep(
    analysis_type,
    seed,
    epsilon,
    max_iter,
    record_every,
    initialization_trials,
    shared_plot_seed,
):
    """Run LDLT once and HB/Nesterov from many random initializations."""

    aggregate_rows = []
    trial_rows = []
    plot_history_rows = []
    config = synthetic_parameter_sweep_config()
    values = synthetic_parameter_sweep_values()

    if analysis_type == "sparsity":
        kind = "sparse"
        parameter_name = "sparseness_percentage"
    elif analysis_type == "rho":
        kind = "ill"
        parameter_name = "rho"
    else:
        raise ValueError("Unknown analysis_type: " + str(analysis_type))

    for index in range(len(values)):
        parameter_value = values[index]
        print(
            "Running synthetic "
            + analysis_type
            + " sweep: "
            + parameter_name
            + "="
            + format(parameter_value, ".1f")
        )

        if analysis_type == "rho":
            instance = create_conditioning_effect_instance(
                parameter_value,
                config,
                seed,
            )
        else:
            scenario = create_synthetic_analysis_scenario(
                kind,
                parameter_value,
                config,
                seed,
            )
            instance = scenario["instance"]
        q = instance.q
        c = instance.c

        spectral = estimate_spectral_bounds(
            q,
            instance.lambda_reg,
            seed=seed,
            l_safety_factor=1.01,
        )

        def objective_fn(weights):
            return objective_value(
                weights,
                instance.h_train_aug,
                instance.y_train,
                instance.lambda_reg,
            )

        ldlt_result = ldlt_solve_weights(q, c)
        reference_weights = ldlt_result.weights
        reference_objective = objective_fn(reference_weights)

        aggregate_rows.append(
            make_synthetic_parameter_ldlt_aggregate_row(
                analysis_type,
                parameter_name,
                parameter_value,
                ldlt_result,
                instance,
                spectral,
                epsilon,
                objective_fn,
                reference_weights,
                reference_objective,
            )
        )

        method_trial_rows = {
            "Heavy Ball": [],
            "Nesterov": [],
        }

        for trial_index in range(initialization_trials):
            initialization_seed = seed + 100000 + trial_index
            w0 = random_initial_weights(c.shape, initialization_seed)

            hb_result = heavy_ball(
                q,
                c,
                w0,
                mu=spectral.mu,
                l_smooth=spectral.l_smooth,
                tol=epsilon,
                max_iter=max_iter,
                objective_fn=None,
                reference_weights=None,
                record_every=max_iter + 1,
            )
            nag_result = nesterov_accelerated_gradient(
                q,
                c,
                w0,
                mu=spectral.mu,
                l_smooth=spectral.l_smooth,
                tol=epsilon,
                max_iter=max_iter,
                objective_fn=None,
                reference_weights=None,
                record_every=max_iter + 1,
            )

            iterative_results = [hb_result, nag_result]
            for result in iterative_results:
                row = make_synthetic_parameter_trial_row(
                    analysis_type,
                    parameter_name,
                    parameter_value,
                    trial_index,
                    initialization_seed,
                    result,
                    w0,
                    instance,
                    spectral,
                    epsilon,
                    objective_fn,
                    reference_weights,
                    reference_objective,
                )
                trial_rows.append(row)
                method_trial_rows[result.method].append(row)

        for method in ["Heavy Ball", "Nesterov"]:
            aggregate_rows.append(
                make_synthetic_parameter_iterative_aggregate_row(
                    analysis_type,
                    parameter_name,
                    parameter_value,
                    method,
                    method_trial_rows[method],
                    instance,
                    spectral,
                    epsilon,
                )
            )

        plot_seed, plot_max_iter, hb_plot_result, nag_plot_result = (
            find_convergent_synthetic_plot_run(
                analysis_type,
                parameter_value,
                q,
                c,
                spectral,
                epsilon,
                max_iter,
                record_every,
                objective_fn,
                reference_weights,
                shared_plot_seed,
            )
        )

        plot_results = [hb_plot_result, nag_plot_result]
        for result in plot_results:
            plot_history_rows.extend(
                make_synthetic_parameter_history_rows(
                    analysis_type,
                    parameter_name,
                    parameter_value,
                    result,
                    instance,
                    spectral,
                    epsilon,
                    reference_objective,
                    plot_seed,
                    plot_max_iter,
                )
            )

    return aggregate_rows, trial_rows, plot_history_rows


def run_synthetic_rho_converged_plot_history(
    seed,
    epsilon,
    max_iter,
    record_every,
    shared_plot_seed,
):
    """Regenerate only rho plot histories with converged plot runs."""

    plot_history_rows = []
    config = synthetic_parameter_sweep_config()
    values = synthetic_parameter_sweep_values()

    for index in range(len(values)):
        rho = values[index]
        print(
            "Regenerating synthetic rho convergence plot: rho="
            + format(rho, ".1f")
        )

        instance = create_conditioning_effect_instance(
            rho,
            config,
            seed,
        )
        q = instance.q
        c = instance.c
        spectral = estimate_spectral_bounds(
            q,
            instance.lambda_reg,
            seed=seed,
            l_safety_factor=1.01,
        )

        def objective_fn(weights):
            return objective_value(
                weights,
                instance.h_train_aug,
                instance.y_train,
                instance.lambda_reg,
            )

        ldlt_result = ldlt_solve_weights(q, c)
        reference_weights = ldlt_result.weights
        reference_objective = objective_fn(reference_weights)

        plot_seed, plot_max_iter, hb_result, nag_result = (
            find_convergent_synthetic_plot_run(
                "rho",
                rho,
                q,
                c,
                spectral,
                epsilon,
                max_iter,
                record_every,
                objective_fn,
                reference_weights,
                shared_plot_seed,
            )
        )

        for result in [hb_result, nag_result]:
            plot_history_rows.extend(
                make_synthetic_parameter_history_rows(
                    "rho",
                    "rho",
                    rho,
                    result,
                    instance,
                    spectral,
                    epsilon,
                    reference_objective,
                    plot_seed,
                    plot_max_iter,
                )
            )

    return plot_history_rows


def find_convergent_synthetic_plot_run(
    analysis_type,
    parameter_value,
    q,
    c,
    spectral,
    epsilon,
    max_iter,
    record_every,
    objective_fn,
    reference_weights,
    shared_plot_seed,
):
    """Find plot curves where both iterative algorithms converge.

    The 100 statistical trials keep their original max_iter. This helper is
    only for the visual convergence curves. For difficult rho values, Nesterov
    may need more iterations to reach 1e-6, so we increase only the plot budget.
    """

    max_iter_values = [max_iter]
    if analysis_type == "rho":
        max_iter_values.extend(
            [
                max_iter * 2,
                max_iter * 4,
                max_iter * 8,
            ]
        )

    last_results = None
    search_plan = []
    for plot_max_iter in max_iter_values:
        search_plan.append((shared_plot_seed, plot_max_iter))

    if analysis_type == "rho":
        largest_max_iter = max_iter_values[-1]
        for attempt in range(20):
            search_plan.append((shared_plot_seed + 1000 + attempt, largest_max_iter))

    for plot_seed, plot_max_iter in search_plan:
        w0 = random_initial_weights(c.shape, plot_seed)
        hb_result = heavy_ball(
            q,
            c,
            w0,
            mu=spectral.mu,
            l_smooth=spectral.l_smooth,
            tol=epsilon,
            max_iter=plot_max_iter,
            objective_fn=objective_fn,
            reference_weights=reference_weights,
            record_every=record_every,
        )
        nag_result = nesterov_accelerated_gradient(
            q,
            c,
            w0,
            mu=spectral.mu,
            l_smooth=spectral.l_smooth,
            tol=epsilon,
            max_iter=plot_max_iter,
            objective_fn=objective_fn,
            reference_weights=reference_weights,
            record_every=record_every,
        )
        last_results = (plot_seed, plot_max_iter, hb_result, nag_result)

        if hb_result.converged and nag_result.converged:
            if plot_seed != shared_plot_seed or plot_max_iter != max_iter:
                print(
                    "  Plot run selected for "
                    + analysis_type
                    + "="
                    + format(float(parameter_value), ".1f")
                    + ": seed="
                    + str(plot_seed)
                    + ", max_iter="
                    + str(plot_max_iter)
                )
            return plot_seed, plot_max_iter, hb_result, nag_result

    print(
        "  Warning: no fully converged plot run found for "
        + analysis_type
        + "="
        + format(float(parameter_value), ".1f")
        + ". Using the best attempted run."
    )
    return last_results


def random_initial_weights(shape, seed):
    """Create a reproducible random starting point for W."""

    rng = np.random.default_rng(seed)
    return 0.001 * rng.normal(size=shape)


def frobenius_norm_numpy(values):
    return float(np.sqrt(np.sum(values * values)))


def make_synthetic_parameter_trial_row(
    analysis_type,
    parameter_name,
    parameter_value,
    trial_index,
    initialization_seed,
    result,
    initial_weights,
    instance,
    spectral,
    epsilon,
    objective_fn,
    reference_weights,
    reference_objective,
):
    initial_gradient = initial_weights @ instance.q - instance.c
    initial_gradient_norm = frobenius_norm_numpy(initial_gradient)

    objective = objective_fn(result.weights)
    objective_gap = max(0.0, objective - reference_objective)
    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug

    return {
        "analysis_type": analysis_type,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "initialization_index": trial_index,
        "initialization_seed": initialization_seed,
        "algorithm": result.method,
        "epsilon": epsilon,
        "output_weight_count": output_weight_count(instance),
        "hidden_width": instance.hidden_width,
        "q_dimension": instance.q.shape[0],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
        "converged": result.converged,
        "iterations": result.iterations,
        "computational_time_seconds": result.elapsed_seconds,
        "initial_gradient_norm": initial_gradient_norm,
        "final_gradient_norm": result.final_gradient_norm,
        "objective_value": objective,
        "objective_gap_to_ldlt": objective_gap,
        "relative_error_to_ldlt": relative_error(result.weights, reference_weights),
        "train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
        "alpha": value_or_empty(result.alpha),
        "beta": value_or_empty(result.beta),
    }


def make_synthetic_parameter_iterative_aggregate_row(
    analysis_type,
    parameter_name,
    parameter_value,
    method,
    trial_rows,
    instance,
    spectral,
    epsilon,
):
    number_of_trials = len(trial_rows)
    converged_count = 0
    for row in trial_rows:
        if row["converged"] is True:
            converged_count += 1

    return {
        "analysis_type": analysis_type,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "algorithm": method,
        "epsilon": epsilon,
        "number_of_trials": number_of_trials,
        "output_weight_count": output_weight_count(instance),
        "hidden_width": instance.hidden_width,
        "q_dimension": instance.q.shape[0],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
        "convergence_rate": safe_fraction(converged_count, number_of_trials),
        "mean_computational_time_seconds": mean_from_rows(
            trial_rows,
            "computational_time_seconds",
        ),
        "std_computational_time_seconds": std_from_rows(
            trial_rows,
            "computational_time_seconds",
        ),
        "mean_iterations": mean_from_rows(trial_rows, "iterations"),
        "std_iterations": std_from_rows(trial_rows, "iterations"),
        "mean_initial_gradient_norm": mean_from_rows(
            trial_rows,
            "initial_gradient_norm",
        ),
        "std_initial_gradient_norm": std_from_rows(
            trial_rows,
            "initial_gradient_norm",
        ),
        "mean_final_gradient_norm": mean_from_rows(
            trial_rows,
            "final_gradient_norm",
        ),
        "std_final_gradient_norm": std_from_rows(
            trial_rows,
            "final_gradient_norm",
        ),
        "mean_objective_value": mean_from_rows(trial_rows, "objective_value"),
        "mean_objective_gap_to_ldlt": mean_from_rows(
            trial_rows,
            "objective_gap_to_ldlt",
        ),
        "mean_relative_error_to_ldlt": mean_from_rows(
            trial_rows,
            "relative_error_to_ldlt",
        ),
        "mean_train_accuracy": mean_from_rows(trial_rows, "train_accuracy"),
        "mean_test_accuracy": mean_from_rows(trial_rows, "test_accuracy"),
        "alpha": value_from_first_row(trial_rows, "alpha"),
        "beta": value_from_first_row(trial_rows, "beta"),
    }


def make_synthetic_parameter_ldlt_aggregate_row(
    analysis_type,
    parameter_name,
    parameter_value,
    result,
    instance,
    spectral,
    epsilon,
    objective_fn,
    reference_weights,
    reference_objective,
):
    objective = objective_fn(result.weights)
    objective_gap = max(0.0, objective - reference_objective)
    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug

    return {
        "analysis_type": analysis_type,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "algorithm": result.method,
        "epsilon": epsilon,
        "number_of_trials": 1,
        "output_weight_count": output_weight_count(instance),
        "hidden_width": instance.hidden_width,
        "q_dimension": instance.q.shape[0],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
        "convergence_rate": 1.0,
        "mean_computational_time_seconds": result.elapsed_seconds,
        "std_computational_time_seconds": 0.0,
        "mean_iterations": result.iterations,
        "std_iterations": 0.0,
        "mean_initial_gradient_norm": "",
        "std_initial_gradient_norm": "",
        "mean_final_gradient_norm": result.final_gradient_norm,
        "std_final_gradient_norm": 0.0,
        "mean_objective_value": objective,
        "mean_objective_gap_to_ldlt": objective_gap,
        "mean_relative_error_to_ldlt": relative_error(
            result.weights,
            reference_weights,
        ),
        "mean_train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "mean_test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
        "alpha": "",
        "beta": "",
    }


def mean_from_rows(rows, field):
    values = numeric_values_from_rows(rows, field)
    if len(values) == 0:
        return ""
    return float(np.mean(values))


def std_from_rows(rows, field):
    values = numeric_values_from_rows(rows, field)
    if len(values) == 0:
        return ""
    return float(np.std(values))


def numeric_values_from_rows(rows, field):
    values = []
    for row in rows:
        value = row[field]
        if value != "":
            values.append(float(value))
    return values


def value_from_first_row(rows, field):
    if len(rows) == 0:
        return ""
    return rows[0][field]


def safe_fraction(numerator, denominator):
    if denominator == 0:
        return ""
    return float(numerator) / float(denominator)


def make_synthetic_parameter_sweep_row(
    analysis_type,
    parameter_name,
    parameter_value,
    result,
    instance,
    spectral,
    epsilon,
    objective_fn,
    reference_weights,
    reference_objective,
):
    objective = objective_fn(result.weights)
    objective_gap = max(0.0, objective - reference_objective)
    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug

    return {
        "analysis_type": analysis_type,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "algorithm": result.method,
        "epsilon": epsilon,
        "output_weight_count": output_weight_count(instance),
        "hidden_width": instance.hidden_width,
        "q_dimension": instance.q.shape[0],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
        "converged": result.converged,
        "iterations": result.iterations,
        "time_seconds": result.elapsed_seconds,
        "final_gradient_norm": result.final_gradient_norm,
        "objective_value": objective,
        "objective_gap_to_ldlt": objective_gap,
        "relative_error_to_ldlt": relative_error(result.weights, reference_weights),
        "train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
        "alpha": value_or_empty(result.alpha),
        "beta": value_or_empty(result.beta),
    }


def make_synthetic_parameter_history_rows(
    analysis_type,
    parameter_name,
    parameter_value,
    result,
    instance,
    spectral,
    epsilon,
    reference_objective,
    plot_initialization_seed="",
    plot_max_iter="",
):
    rows = []
    history = result.history
    denominator = max(1.0, abs(reference_objective))

    iterations = history["iteration"]
    grad_norms = history["grad_norm"]
    objectives = history["objective"]

    for index in range(len(iterations)):
        objective_gap = max(0.0, objectives[index] - reference_objective)
        rows.append(
            {
                "analysis_type": analysis_type,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "method": result.method,
                "epsilon": epsilon,
                "output_weight_count": output_weight_count(instance),
                "condition_number": spectral.condition_estimate,
                "plot_initialization_seed": plot_initialization_seed,
                "plot_max_iter": plot_max_iter,
                "plot_converged": result.converged,
                "iteration": int(iterations[index]),
                "relative_gap": objective_gap / denominator,
                "gradient_norm": grad_norms[index],
                "objective_gap_to_ldlt": objective_gap,
            }
        )

    return rows


def requested_all_rho_values():
    return correlation_values()


def requested_small_rho_values():
    return [0.1, 0.5, 0.9]


def requested_all_sparsity_values():
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def requested_small_sparsity_values():
    return [0.1, 0.4, 0.7]


def run_requested_dimension_scaling_times(
    dimension_configs,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    rows = []

    for index in range(len(dimension_configs)):
        config = dimension_configs[index]
        scenario = create_synthetic_analysis_scenario(
            "well",
            0.1,
            config,
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        dimensionality = output_weight_count(context["instance"])
        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )
        results = [context["ldlt_result"], hb_result, nag_result]

        for result in results:
            rows.append(
                {
                    "algorithm": result.method,
                    "dimensionality": dimensionality,
                    "computational_time": result.elapsed_seconds,
                }
            )

    return rows


def run_requested_conditioning_effect(
    fixed_config,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    time_rows = []
    history_rows = []
    rho_values = requested_all_rho_values()

    for index in range(len(rho_values)):
        rho = rho_values[index]
        scenario = create_synthetic_analysis_scenario(
            "ill",
            rho,
            fixed_config,
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        dimensionality = output_weight_count(context["instance"])
        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )
        results = [context["ldlt_result"], hb_result, nag_result]

        for result in results:
            time_rows.append(
                {
                    "algorithm": result.method,
                    "rho": rho,
                    "dimensionality": dimensionality,
                    "computational_time": result.elapsed_seconds,
                }
            )

        history_rows.extend(
            make_requested_history_rows(
                context,
                hb_result,
                "rho",
                rho,
                dimensionality,
                epsilon,
            )
        )
        history_rows.extend(
            make_requested_history_rows(
                context,
                nag_result,
                "rho",
                rho,
                dimensionality,
                epsilon,
            )
        )

    return time_rows, history_rows


def run_requested_sparsity_effect(
    fixed_config,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    time_rows = []
    history_rows = []
    sparsity_values = requested_all_sparsity_values()

    for index in range(len(sparsity_values)):
        sparsity = sparsity_values[index]
        scenario = create_synthetic_analysis_scenario(
            "sparse",
            sparsity,
            fixed_config,
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        dimensionality = output_weight_count(context["instance"])
        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )
        results = [context["ldlt_result"], hb_result, nag_result]

        for result in results:
            time_rows.append(
                {
                    "algorithm": result.method,
                    "sparseness_percentage": sparsity,
                    "dimensionality": dimensionality,
                    "computational_time": result.elapsed_seconds,
                }
            )

        history_rows.extend(
            make_requested_history_rows(
                context,
                hb_result,
                "sparseness_percentage",
                sparsity,
                dimensionality,
                epsilon,
            )
        )
        history_rows.extend(
            make_requested_history_rows(
                context,
                nag_result,
                "sparseness_percentage",
                sparsity,
                dimensionality,
                epsilon,
            )
        )

    return time_rows, history_rows


def run_requested_dimension_parameter_times(
    kind,
    parameter_name,
    values,
    dimension_configs,
    seed,
    epsilon,
    max_iter,
    record_every,
    include_library_methods,
):
    rows = []

    for config_index in range(len(dimension_configs)):
        config = dimension_configs[config_index]

        for value_index in range(len(values)):
            value = values[value_index]
            scenario = create_synthetic_analysis_scenario(
                kind,
                value,
                config,
                seed + 100 * config_index + value_index,
            )
            context = prepare_analysis_context(
                scenario,
                seed + 100 * config_index + value_index,
            )
            dimensionality = output_weight_count(context["instance"])
            hb_result, nag_result = run_scratch_iterative_methods(
                context,
                epsilon,
                max_iter,
                record_every,
            )

            if include_library_methods:
                pytorch_hb, pytorch_nag = run_library_reference_methods(
                    context,
                    hb_result,
                    nag_result,
                    epsilon,
                    max_iter,
                    record_every,
                )
                results = [
                    hb_result,
                    nag_result,
                    pytorch_hb,
                    pytorch_nag,
                ]
            else:
                results = [
                    context["ldlt_result"],
                    hb_result,
                    nag_result,
                ]

            for result in results:
                row = {
                    "algorithm": result.method,
                    "dimensionality": dimensionality,
                    "computational_time": result.elapsed_seconds,
                }
                if parameter_name == "rho":
                    row["rho"] = value
                else:
                    row["sparseness_percentage"] = value
                rows.append(row)

    return rows


def run_requested_library_plots(
    fixed_config,
    seed,
    epsilon,
    max_iter,
    record_every,
    figures_dir,
):
    rho_values = requested_small_rho_values()
    sparsity_values = requested_small_sparsity_values()

    for index in range(len(rho_values)):
        rho = rho_values[index]
        scenario = create_synthetic_analysis_scenario(
            "ill",
            rho,
            fixed_config,
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        dimensionality = output_weight_count(context["instance"])
        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )
        pytorch_hb, pytorch_nag = run_library_reference_methods(
            context,
            hb_result,
            nag_result,
            epsilon,
            max_iter,
            record_every,
        )
        plot_requested_library_pair(
            context,
            hb_result,
            pytorch_hb,
            "rho",
            rho,
            dimensionality,
            epsilon,
            figures_dir / (
                "requested_library_rho_"
                + format_decimal_for_name(rho)
                + "_heavy_ball.png"
            ),
        )
        plot_requested_library_pair(
            context,
            nag_result,
            pytorch_nag,
            "rho",
            rho,
            dimensionality,
            epsilon,
            figures_dir / (
                "requested_library_rho_"
                + format_decimal_for_name(rho)
                + "_nesterov.png"
            ),
        )

    for index in range(len(sparsity_values)):
        sparsity = sparsity_values[index]
        scenario = create_synthetic_analysis_scenario(
            "sparse",
            sparsity,
            fixed_config,
            seed + 100 + index,
        )
        context = prepare_analysis_context(scenario, seed + 100 + index)
        dimensionality = output_weight_count(context["instance"])
        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )
        pytorch_hb, pytorch_nag = run_library_reference_methods(
            context,
            hb_result,
            nag_result,
            epsilon,
            max_iter,
            record_every,
        )
        plot_requested_library_pair(
            context,
            hb_result,
            pytorch_hb,
            "sparseness_percentage",
            sparsity,
            dimensionality,
            epsilon,
            figures_dir / (
                "requested_library_sparsity_"
                + format_decimal_for_name(sparsity)
                + "_heavy_ball.png"
            ),
        )
        plot_requested_library_pair(
            context,
            nag_result,
            pytorch_nag,
            "sparseness_percentage",
            sparsity,
            dimensionality,
            epsilon,
            figures_dir / (
                "requested_library_sparsity_"
                + format_decimal_for_name(sparsity)
                + "_nesterov.png"
            ),
        )


def run_requested_ldlt_builtin_times(dimension_configs, seed):
    rows = []

    for index in range(len(dimension_configs)):
        config = dimension_configs[index]
        scenario = create_synthetic_analysis_scenario(
            "well",
            0.1,
            config,
            seed + index,
        )
        context = prepare_analysis_context(scenario, seed + index)
        dimensionality = output_weight_count(context["instance"])
        results = [
            context["ldlt_result"],
            context["numpy_reference"],
            context["scipy_ldlt"],
        ]

        for result in results:
            rows.append(
                {
                    "algorithm": result.method,
                    "dimensionality": dimensionality,
                    "computational_time": result.elapsed_seconds,
                    "relative_error_to_our_ldlt": relative_error(
                        result.weights,
                        context["ldlt_weights"],
                    ),
                }
            )

    return rows


def run_requested_beta_comparisons(
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
    figures_dir,
):
    rows = []
    histories = []
    case_groups = build_nesterov_beta_case_groups(suite, seed)

    for group_index in range(len(case_groups)):
        case_title, filename, scenarios = case_groups[group_index]
        plot_items = []

        for scenario_index in range(len(scenarios)):
            scenario = scenarios[scenario_index]
            context, fixed_result, variable_result = run_nesterov_beta_pair(
                scenario,
                seed + group_index * 100 + scenario_index,
                epsilon,
                max_iter,
                record_every,
            )

            append_analysis_result(
                rows,
                histories,
                "beta_fixed_vs_variable",
                context,
                fixed_result,
                epsilon,
                "hidden_width",
                context["instance"].hidden_width,
            )
            append_analysis_result(
                rows,
                histories,
                "beta_fixed_vs_variable",
                context,
                variable_result,
                epsilon,
                "hidden_width",
                context["instance"].hidden_width,
            )

            plot_items.append(
                {
                    "context": context,
                    "results": [fixed_result, variable_result],
                }
            )

        plot_requested_beta_size_grid(
            plot_items,
            epsilon,
            figures_dir / filename,
            case_title,
        )

    return rows, histories


def build_nesterov_beta_case_groups(suite, seed, hidden_sizes=None):
    """Create the five beta-comparison groups at three hidden sizes."""

    if hidden_sizes is None:
        hidden_sizes = nesterov_beta_hidden_sizes()

    case_groups = []
    scenario_seed = seed

    synthetic_definitions = [
        (
            "Synthetic well-conditioned",
            "requested_beta_well_conditioned.png",
            "well",
            0.1,
        ),
        (
            "Synthetic ill-conditioned",
            "requested_beta_ill_conditioned.png",
            "ill",
            0.9,
        ),
        (
            "Synthetic sparse, zero probability 0.7",
            "requested_beta_sparse_70_percent.png",
            "sparse",
            0.7,
        ),
    ]

    for definition in synthetic_definitions:
        title, filename, kind, sweep_value = definition
        scenarios = []
        for hidden_width in hidden_sizes:
            config = nesterov_beta_synthetic_config(hidden_width)
            scenario = create_synthetic_analysis_scenario(
                kind,
                sweep_value,
                config,
                scenario_seed,
            )
            add_hidden_size_to_scenario_name(scenario, hidden_width)
            scenarios.append(scenario)
            scenario_seed += 1
        case_groups.append((title, filename, scenarios))

    wine_scenarios = []
    for hidden_width in hidden_sizes:
        temporary = []
        add_real_wine_scenario(temporary, suite, scenario_seed, hidden_width)
        add_hidden_size_to_scenario_name(temporary[0], hidden_width)
        wine_scenarios.append(temporary[0])
        scenario_seed += 1
    case_groups.append(("Wine real dataset", "requested_beta_wine.png", wine_scenarios))

    digits_scenarios = []
    for hidden_width in hidden_sizes:
        temporary = []
        add_real_digits_scenario(temporary, suite, scenario_seed, hidden_width)
        add_hidden_size_to_scenario_name(temporary[0], hidden_width)
        digits_scenarios.append(temporary[0])
        scenario_seed += 1
    case_groups.append(("Digits real dataset", "requested_beta_digits.png", digits_scenarios))

    return case_groups


def run_nesterov_beta_pair(
    scenario,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    """Run only the two Nesterov variants needed for this comparison."""

    context = prepare_analysis_context(scenario, seed)
    unused_hb, fixed_result = run_scratch_iterative_methods(
        context,
        epsilon,
        max_iter,
        record_every,
    )
    variable_result = run_variable_nesterov_method(
        context,
        epsilon,
        max_iter,
        record_every,
    )
    return context, fixed_result, variable_result


def plot_requested_beta_size_grid(plot_items, epsilon, path, title):
    """Plot fixed-beta and variable-beta Nesterov for several hidden sizes."""

    if len(plot_items) == 0:
        return

    metrics = [
        ("grad_norm", "Gradient Norm", "||grad f(W)||_F"),
        ("objective", "Objective Gap", "f(W) - f(W_ref)"),
        ("relative_error", "Relative Error", "||W - W_ref|| / ||W_ref||"),
    ]

    fig, axes = plt.subplots(
        len(plot_items),
        3,
        figsize=(15.5, 4.1 * len(plot_items)),
        squeeze=False,
    )

    for row_index in range(len(plot_items)):
        item = plot_items[row_index]
        context = item["context"]
        instance = context["instance"]
        reference_objective = context["reference_objective"]

        row_label = (
            "hidden="
            + str(instance.hidden_width)
            + ", output weights="
            + str(output_weight_count(instance))
            + ", Q dim="
            + str(instance.q.shape[0])
        )

        for metric_index in range(len(metrics)):
            field, metric_title, ylabel = metrics[metric_index]
            axis = axes[row_index][metric_index]

            for result in item["results"]:
                history = result.history
                iterations = history["iteration"]

                if field == "grad_norm":
                    values = history["grad_norm"]
                elif field == "objective":
                    values = []
                    for value in history["objective"]:
                        gap = max(0.0, value - reference_objective)
                        values.append(gap)
                else:
                    values = history["relative_error"]

                safe_values = []
                for value in values:
                    safe_values.append(max(float(value), metric_floor(field)))

                style = plot_style_for_method(result.method)
                axis.semilogy(
                    iterations,
                    safe_values,
                    label=result.method,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markevery=style["markevery"],
                    alpha=style["alpha"],
                )

            if row_index == 0:
                axis.set_title(metric_title)
            axis.set_xlabel("Iteration")
            if metric_index == 0:
                axis.set_ylabel(row_label + "\n" + ylabel)
            else:
                axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

    subtitle = (
        "Fixed beta vs variable beta, epsilon="
        + format_epsilon_label(epsilon)
        + ", hidden sizes="
        + ", ".join(str(value) for value in nesterov_beta_hidden_sizes())
    )
    fig.suptitle(title + "\n" + subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_nesterov_beta_focused_plots(
    suite,
    seed,
    max_iter,
    record_every,
    figures_dir,
):
    """Create the requested 2x2 beta plots for two epsilons."""

    rows = []
    histories = []
    epsilons = nesterov_beta_focused_epsilons()
    hidden_sizes = nesterov_beta_focused_hidden_sizes()

    for epsilon_index in range(len(epsilons)):
        epsilon = epsilons[epsilon_index]
        case_groups = build_nesterov_beta_case_groups(
            suite,
            seed + epsilon_index * 1000,
            hidden_sizes,
        )

        for group_index in range(len(case_groups)):
            case_title, filename, scenarios = case_groups[group_index]
            plot_items = []

            for scenario_index in range(len(scenarios)):
                scenario = scenarios[scenario_index]
                context, fixed_result, variable_result = run_nesterov_beta_pair(
                    scenario,
                    seed + epsilon_index * 1000 + group_index * 100 + scenario_index,
                    epsilon,
                    max_iter,
                    record_every,
                )

                append_analysis_result(
                    rows,
                    histories,
                    "beta_fixed_vs_variable_focused",
                    context,
                    fixed_result,
                    epsilon,
                    "hidden_width",
                    context["instance"].hidden_width,
                )
                append_analysis_result(
                    rows,
                    histories,
                    "beta_fixed_vs_variable_focused",
                    context,
                    variable_result,
                    epsilon,
                    "hidden_width",
                    context["instance"].hidden_width,
                )

                plot_items.append(
                    {
                        "context": context,
                        "results": [fixed_result, variable_result],
                    }
                )

            output_name = focused_beta_filename(filename, epsilon)
            plot_requested_beta_focus_grid(
                plot_items,
                epsilon,
                figures_dir / output_name,
                case_title,
            )

    return rows, histories


def focused_beta_filename(filename, epsilon):
    if filename.endswith(".png"):
        base_name = filename[:-4]
    else:
        base_name = filename

    epsilon_text = format_epsilon_label(epsilon)
    return base_name + "_focused_epsilon_" + epsilon_text + ".png"


def beta_plot_label(method):
    if method == "Nesterov":
        return "Nesterov fixed beta"
    return method


def plot_requested_beta_focus_grid(plot_items, epsilon, path, title):
    """Plot only Gradient Norm and Relative Gap in a compact 2x2 grid."""

    if len(plot_items) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), squeeze=False)
    metrics = [
        ("grad_norm", "Gradient Norm", "||grad f(W)||_F"),
        ("relative_gap", "Relative Gap", "(f(W) - f*) / max(1, |f*|)"),
    ]

    for row_index in range(len(plot_items)):
        item = plot_items[row_index]
        context = item["context"]
        instance = context["instance"]
        reference_objective = context["reference_objective"]
        denominator = max(1.0, abs(reference_objective))

        for metric_index in range(len(metrics)):
            field, metric_title, ylabel = metrics[metric_index]
            axis = axes[row_index][metric_index]

            for result in item["results"]:
                history = result.history
                iterations = history["iteration"]

                if field == "grad_norm":
                    values = history["grad_norm"]
                else:
                    values = []
                    for objective in history["objective"]:
                        gap = max(0.0, objective - reference_objective)
                        values.append(gap / denominator)

                safe_values = []
                for value in values:
                    safe_values.append(max(float(value), metric_floor(field)))

                style = plot_style_for_method(result.method)
                axis.semilogy(
                    iterations,
                    safe_values,
                    label=beta_plot_label(result.method),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markevery=style["markevery"],
                    alpha=style["alpha"],
                )

            subplot_title = (
                "hidden width="
                + str(instance.hidden_width)
                + ", output weights="
                + str(output_weight_count(instance))
                + "\n"
                + metric_title
            )
            axis.set_title(subplot_title)
            axis.set_xlabel("Iteration")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

    subtitle = (
        "hidden width="
        + ", ".join(str(value) for value in nesterov_beta_focused_hidden_sizes())
        + ", epsilon="
        + format_epsilon_label(epsilon)
    )
    fig.suptitle(title + "\n" + subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_requested_real_dataset_analysis(
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
    figures_dir,
):
    rows = []
    scenarios = []
    add_real_wine_scenario(scenarios, suite, seed)
    add_real_digits_scenario(scenarios, suite, seed + 1)

    for index in range(len(scenarios)):
        scenario = scenarios[index]
        context = prepare_analysis_context(scenario, seed + index)
        hb_result, nag_result = run_scratch_iterative_methods(
            context,
            epsilon,
            max_iter,
            record_every,
        )

        filename = (
            "requested_real_"
            + scenario["dataset_name"]
            + "_scratch_iterative.png"
        )
        plot_requested_two_results(
            context,
            [hb_result, nag_result],
            epsilon,
            figures_dir / filename,
            "Real Dataset Scratch Iterative Methods",
        )

        results = [context["ldlt_result"], hb_result, nag_result]
        for result in results:
            rows.append(
                {
                    "dataset": scenario["dataset_name"],
                    "algorithm": result.method,
                    "computational_time": result.elapsed_seconds,
                }
            )

    return rows


class RealComparisonInstance:
    """Small object for the large real-data ELM comparison.

    We store H directly and avoid building Q. This is important when the hidden
    layer has 50000 or 100000 weights, because Q would be unnecessarily large.
    """

    def __init__(
        self,
        dataset_name,
        y_train,
        train_labels,
        h_train_aug,
        lambda_reg,
        hidden_width,
        n_features,
        n_test,
    ):
        self.dataset_name = dataset_name
        self.y_train = y_train
        self.train_labels = train_labels
        self.h_train_aug = h_train_aug
        self.lambda_reg = lambda_reg
        self.hidden_width = hidden_width
        self.n_features = n_features
        self.n_train = h_train_aug.shape[1]
        self.n_test = n_test
        self.n_classes = y_train.shape[0]


def parse_int_list(text):
    values = []
    parts = str(text).split(",")
    for part in parts:
        stripped = part.strip()
        if stripped != "":
            values.append(int(stripped))
    return values


def parse_float_list(text):
    values = []
    parts = str(text).split(",")
    for part in parts:
        stripped = part.strip()
        if stripped != "":
            values.append(float(stripped))
    return values


def run_real_pytorch_same_initialization_comparison(
    output_weight_values,
    epsilon_values,
    max_iter,
    record_every,
    seed,
):
    """Compare our real-data optimizers with PyTorch from the same W0.

    The CSV contains every combination:
    dataset x output-weight count x epsilon x optimizer.
    """

    check_sklearn_available()

    rows = []
    dataset_names = ["wine", "digits"]

    for dataset_index in range(len(dataset_names)):
        dataset_name = dataset_names[dataset_index]

        for weight_index in range(len(output_weight_values)):
            output_weight_count_value = output_weight_values[weight_index]
            instance = create_real_comparison_instance(
                dataset_name,
                output_weight_count_value,
                seed + 100 * dataset_index + weight_index,
            )
            spectral = estimate_real_comparison_spectral_bounds(instance)
            reference_weights = reference_weights_for_real_gap(instance)
            reference_objective = objective_value_from_h(
                reference_weights,
                instance,
            )

            # Same W0 for all epsilons and all four optimizers for this
            # specific pair (dataset, hidden size).
            rng = np.random.default_rng(seed + 1000 + 100 * dataset_index + weight_index)
            w0 = 0.01 * rng.normal(
                size=(instance.n_classes, instance.hidden_width + 1)
            )
            initial_grad = elm_gradient_from_h(w0, instance)
            initial_grad_norm = float(np.sqrt(np.sum(initial_grad * initial_grad)))

            for epsilon in epsilon_values:
                hb_result = heavy_ball_from_h(
                    instance,
                    w0,
                    spectral["mu"],
                    spectral["l_smooth"],
                    epsilon,
                    max_iter,
                )

                nag_result = nesterov_from_h(
                    instance,
                    w0,
                    spectral["mu"],
                    spectral["l_smooth"],
                    epsilon,
                    max_iter,
                )

                pytorch_hb = pytorch_from_h(
                    instance,
                    w0,
                    "PyTorch SGD momentum",
                    hb_result.alpha,
                    hb_result.beta,
                    False,
                    epsilon,
                    max_iter,
                )

                pytorch_nag = pytorch_from_h(
                    instance,
                    w0,
                    "PyTorch SGD Nesterov",
                    nag_result.alpha,
                    nag_result.beta,
                    True,
                    epsilon,
                    max_iter,
                )

                results = [hb_result, nag_result, pytorch_hb, pytorch_nag]
                for result in results:
                    rows.append(
                        make_real_pytorch_comparison_row(
                            result,
                            dataset_name,
                            epsilon,
                            initial_grad_norm,
                            reference_objective,
                            instance,
                        )
                    )

    return rows


def create_real_comparison_instance(dataset_name, target_output_weights, seed):
    """Create Wine or Digits ELM with about target_output_weights output weights."""

    if dataset_name == "wine":
        data = load_wine()
        test_size = 0.30
        activation = "tanh"
        hidden_scale = 1.0
    elif dataset_name == "digits":
        data = load_digits()
        test_size = 0.25
        activation = "tanh"
        hidden_scale = 0.8
    else:
        raise ValueError("Unknown real dataset: " + dataset_name)

    x = data.data.astype(float)
    labels = data.target.astype(int)
    n_features = x.shape[1]
    n_classes = int(np.max(labels)) + 1
    hidden_width = hidden_width_from_output_weights(
        target_output_weights,
        n_classes,
    )

    x_train, x_test, train_labels, test_labels = train_test_split(
        x,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    x_train = x_train.T
    x_test = x_test.T
    x_train, x_test = standardize_train_test(x_train, x_test)

    y_train = one_hot(train_labels, n_classes)

    rng = np.random.default_rng(seed + 10000)
    hidden_weights = rng.normal(size=(hidden_width, n_features))
    hidden_weights = hidden_scale * hidden_weights / np.sqrt(float(n_features))
    hidden_bias = rng.uniform(-1.0, 1.0, size=hidden_width)

    h_train = build_hidden_matrix(
        x_train,
        hidden_weights,
        hidden_bias,
        activation,
    )
    h_train_aug = augment_hidden_matrix(h_train)

    instance = RealComparisonInstance(
        dataset_name,
        y_train,
        train_labels,
        h_train_aug,
        1e-2,
        hidden_width,
        n_features,
        x_test.shape[1],
    )
    return instance


def reference_weights_for_real_gap(instance):
    """Compute a trusted reference solution for objective gaps.

    If the number of hidden units is larger than the number of samples, the
    dual ridge formula is much faster than solving the big normal system. This
    is used only to evaluate the gap in the CSV.
    """

    h_aug = instance.h_train_aug
    y = instance.y_train
    n_samples = instance.n_train

    gamma = n_samples * instance.lambda_reg
    kernel = h_aug.T @ h_aug
    kernel = kernel + gamma * np.eye(n_samples)
    dual_solution = np.linalg.solve(kernel, h_aug.T)
    weights = y @ dual_solution
    return weights


def make_real_pytorch_comparison_row(
    result,
    dataset_name,
    epsilon,
    initial_grad_norm,
    reference_objective,
    instance,
):
    objective = objective_value_from_h(result.weights, instance)
    absolute_gap = max(0.0, objective - reference_objective)
    relative_gap = absolute_gap / max(1.0, abs(reference_objective))

    alpha = ""
    beta = ""
    if result.alpha is not None:
        alpha = result.alpha
    if result.beta is not None:
        beta = result.beta

    row = {
        "algorithm": result.method,
        "dataset": dataset_name,
        "epsilon": epsilon,
        "initial_gradient_norm": initial_grad_norm,
        "final_gradient_norm": result.final_gradient_norm,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "computational_time_seconds": result.elapsed_seconds,
        "number_of_iterations": result.iterations,
        "hidden_layer_size": instance.hidden_width,
        "alpha": alpha,
        "beta": beta,
    }
    return row


def plot_real_pytorch_time_vs_hidden_size(rows, figures_dir, epsilon):
    """Plot computational time against hidden size for Wine and Digits."""

    selected_rows = []
    for row in rows:
        if abs(float(row["epsilon"]) - float(epsilon)) <= 1e-15:
            selected_rows.append(row)

    if len(selected_rows) == 0:
        return

    dataset_names = ["wine", "digits"]
    methods = [
        "Heavy Ball",
        "Nesterov",
        "PyTorch SGD momentum",
        "PyTorch SGD Nesterov",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for dataset_index in range(len(dataset_names)):
        dataset_name = dataset_names[dataset_index]
        axis = axes[dataset_index]

        dataset_rows = []
        for row in selected_rows:
            if row["dataset"] == dataset_name:
                dataset_rows.append(row)

        hidden_sizes = []
        for row in dataset_rows:
            hidden_size = int(float(row["hidden_layer_size"]))
            if hidden_size not in hidden_sizes:
                hidden_sizes.append(hidden_size)
        hidden_sizes.sort()

        for method in methods:
            times = []
            for hidden_size in hidden_sizes:
                value = np.nan
                for row in dataset_rows:
                    same_method = row["algorithm"] == method
                    same_size = int(float(row["hidden_layer_size"])) == hidden_size
                    if same_method and same_size:
                        value = float(row["computational_time_seconds"])
                times.append(value)

            style = plot_style_for_method(method)
            axis.plot(
                hidden_sizes,
                times,
                label=short_label_for_real_pytorch_method(method),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.4,
            )

        axis.set_xlabel("Hidden layer size")
        axis.set_ylabel("Computational time (seconds)")
        axis.set_title(dataset_name)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    title = "Computational Time vs Hidden Layer Size"
    subtitle = "epsilon=" + format_epsilon_label(epsilon)
    fig.suptitle(title + "\n" + subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

    filename = "real_pytorch_time_vs_hidden_size_epsilon_"
    filename = filename + make_safe_filename(format_epsilon_label(epsilon))
    filename = filename + ".png"
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def short_label_for_real_pytorch_method(method):
    if method == "PyTorch SGD momentum":
        return "PyTorch HB"
    if method == "PyTorch SGD Nesterov":
        return "PyTorch Nesterov"
    return method


def plot_real_pytorch_time_vs_epsilon_extreme_hidden_sizes(rows, figures_dir):
    """Plot time versus epsilon separating the two extreme hidden sizes."""

    if len(rows) == 0:
        return

    dataset_names = ["wine", "digits"]
    methods = [
        "Heavy Ball",
        "Nesterov",
        "PyTorch SGD momentum",
        "PyTorch SGD Nesterov",
    ]

    plot_cases = []

    for dataset_index in range(len(dataset_names)):
        dataset_name = dataset_names[dataset_index]
        dataset_rows = []
        for row in rows:
            if row["dataset"] == dataset_name:
                dataset_rows.append(row)

        hidden_sizes = []
        for row in dataset_rows:
            hidden_size = int(float(row["hidden_layer_size"]))
            if hidden_size not in hidden_sizes:
                hidden_sizes.append(hidden_size)
        hidden_sizes.sort()

        if len(hidden_sizes) == 0:
            continue

        selected_sizes = [hidden_sizes[0]]
        if hidden_sizes[-1] != hidden_sizes[0]:
            selected_sizes.append(hidden_sizes[-1])

        for hidden_size in selected_sizes:
            plot_cases.append(
                {
                    "dataset_name": dataset_name,
                    "hidden_size": hidden_size,
                    "rows": dataset_rows,
                }
            )

    if len(plot_cases) == 0:
        return

    fig, axes = plt.subplots(1, len(plot_cases), figsize=(19, 4.8), sharey=False)
    if len(plot_cases) == 1:
        axes = [axes]

    for plot_index in range(len(plot_cases)):
        plot_case = plot_cases[plot_index]
        axis = axes[plot_index]
        dataset_name = plot_case["dataset_name"]
        hidden_size = plot_case["hidden_size"]
        dataset_rows = plot_case["rows"]

        epsilons = []
        for row in dataset_rows:
            epsilon = float(row["epsilon"])
            if epsilon not in epsilons:
                epsilons.append(epsilon)
        epsilons.sort(reverse=True)

        for method in methods:
            times = []
            for epsilon in epsilons:
                value = np.nan
                for row in dataset_rows:
                    same_method = row["algorithm"] == method
                    same_size = int(float(row["hidden_layer_size"])) == hidden_size
                    same_epsilon = abs(float(row["epsilon"]) - epsilon) <= 1e-15
                    if same_method and same_size and same_epsilon:
                        value = float(row["computational_time_seconds"])
                times.append(value)

            style = plot_style_for_method(method)
            axis.plot(
                epsilons,
                times,
                label=short_label_for_real_pytorch_method(method),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
            )

        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("epsilon")
        if plot_index == 0:
            axis.set_ylabel("Computational time (seconds)")
        axis.set_title(dataset_name + "\nhidden size = " + str(hidden_size))
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)

    fig.suptitle(
        "Computational Time vs Epsilon\nSeparate Panels for Smallest and Largest Hidden Layers",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(
        figures_dir / "real_pytorch_time_vs_epsilon_extreme_hidden_sizes.png",
        dpi=160,
    )
    plt.close(fig)

def run_ldlt_scipy_comparison(output_weight_values, seed):
    """Compare our LDLT with SciPy LDLT on Wine and Digits."""

    check_sklearn_available()
    check_scipy_available()

    return run_real_ldlt_scipy_comparison(output_weight_values, seed)


def run_real_ldlt_scipy_comparison(output_weight_values, seed):
    """Compare our LDLT with SciPy LDLT on Wine and Digits.

    The analysis uses three output-weight sizes, ending at about 10000 output
    weights. This gives hidden_width 3333 for Wine and 1000 for Digits in the
    largest case.
    """

    rows = []
    dataset_names = ["wine", "digits"]
    output_weight_count_values = ldlt_scipy_output_weight_values()

    for dataset_index in range(len(dataset_names)):
        dataset_name = dataset_names[dataset_index]
        dataset_seed = seed + 500 + 100 * dataset_index

        for size_index in range(len(output_weight_count_values)):
            output_weight_count_value = output_weight_count_values[size_index]

            instance = create_real_comparison_instance(
                dataset_name,
                output_weight_count_value,
                dataset_seed,
            )

            q, c = build_primal_elm_system(instance)

            our_row, our_weights = solve_real_primal_with_our_ldlt(
                instance,
                q,
                c,
            )
            rows.append(our_row)

            scipy_row, scipy_weights = solve_real_primal_with_scipy_ldlt(
                instance,
                q,
                c,
                our_weights,
            )
            rows.append(scipy_row)

    return rows


def ldlt_scipy_output_weight_values():
    return [2500, 5000, 10000]


def build_primal_elm_system(instance):
    h_aug = instance.h_train_aug
    q = (h_aug @ h_aug.T) / instance.n_train
    q = q + instance.lambda_reg * np.eye(h_aug.shape[0])
    c = (instance.y_train @ h_aug.T) / instance.n_train
    return q, c


def solve_real_primal_with_our_ldlt(instance, q, c):
    start = perf_counter()
    l_factor, d = ldlt_factorize(q)
    weights = solve_with_ldlt(l_factor, d, c.T).T
    elapsed = perf_counter() - start

    minimum_value = objective_value_from_h(weights, instance)
    residual = weights @ q - c
    residual_norm = float(np.sqrt(np.sum(residual * residual)))
    relative_residual = relative_residual_norm(residual, c)

    row = {
        "dataset": instance.dataset_name,
        "case_type": "real_primal",
        "algorithm": "LDLT",
        "hidden_layer_size": instance.hidden_width,
        "output_weight_count": output_weight_count(instance),
        "q_dimension": q.shape[0],
        "minimum_value": minimum_value,
        "computational_time_seconds": elapsed,
        "residual_norm": residual_norm,
        "relative_residual_norm": relative_residual,
        "absolute_error_to_known_optimum": "",
        "relative_error_to_known_optimum": "",
        "relative_difference_to_our_ldlt": 0.0,
    }
    return row, weights


def solve_real_primal_with_scipy_ldlt(instance, q, c, our_weights):
    start = perf_counter()
    weights = solve_with_scipy_ldlt(q, c.T).T
    elapsed = perf_counter() - start

    minimum_value = objective_value_from_h(weights, instance)
    residual = weights @ q - c
    residual_norm = float(np.sqrt(np.sum(residual * residual)))
    relative_residual = relative_residual_norm(residual, c)

    row = {
        "dataset": instance.dataset_name,
        "case_type": "real_primal",
        "algorithm": "SciPy LDLT",
        "hidden_layer_size": instance.hidden_width,
        "output_weight_count": output_weight_count(instance),
        "q_dimension": q.shape[0],
        "minimum_value": minimum_value,
        "computational_time_seconds": elapsed,
        "residual_norm": residual_norm,
        "relative_residual_norm": relative_residual,
        "absolute_error_to_known_optimum": "",
        "relative_error_to_known_optimum": "",
        "relative_difference_to_our_ldlt": relative_error(weights, our_weights),
    }
    return row, weights


def relative_residual_norm(residual, c):
    residual_norm = float(np.sqrt(np.sum(residual * residual)))
    c_norm = float(np.sqrt(np.sum(c * c)))
    if c_norm <= 1e-15:
        return residual_norm
    return residual_norm / c_norm


def remove_old_ldlt_scipy_synthetic_outputs(figures_dir):
    path = figures_dir / "ldlt_scipy_known_solution_accuracy.png"
    if path.exists():
        path.unlink()


def plot_ldlt_scipy_time_bars(rows, figures_dir):
    """Plot LDLT and SciPy LDLT runtime for each tested primal case."""

    if len(rows) == 0:
        return

    methods = ["LDLT", "SciPy LDLT"]
    cases = []
    case_labels = []

    preferred_order = ["wine", "digits"]
    for dataset_name in preferred_order:
        hidden_sizes = []
        for row in rows:
            if row["dataset"] == dataset_name:
                hidden_size = int(float(row["hidden_layer_size"]))
                if hidden_size not in hidden_sizes:
                    hidden_sizes.append(hidden_size)
        hidden_sizes.sort()

        for hidden_size in hidden_sizes:
            cases.append(
                {
                    "dataset": dataset_name,
                    "hidden_size": hidden_size,
                }
            )
            case_labels.append(label_for_ldlt_case(dataset_name, hidden_size))

    x_values = np.arange(len(cases), dtype=float)
    bar_width = 0.34
    fig, axis = plt.subplots(figsize=(11.5, 4.9))

    for method_index in range(len(methods)):
        method = methods[method_index]
        times = []
        for case in cases:
            value = np.nan
            for row in rows:
                same_dataset = row["dataset"] == case["dataset"]
                same_method = row["algorithm"] == method
                same_hidden = (
                    int(float(row["hidden_layer_size"])) == case["hidden_size"]
                )
                if same_dataset and same_method and same_hidden:
                    value = max(float(row["computational_time_seconds"]), 1e-12)
            times.append(value)

        offset = (method_index - 0.5) * bar_width
        axis.bar(
            x_values + offset,
            times,
            width=bar_width,
            label=method,
            color=time_plot_color(method),
        )

    axis.set_yscale("log")
    axis.set_xticks(x_values)
    axis.set_xticklabels(case_labels)
    axis.set_ylabel("Computational time (seconds, log scale)")
    axis.set_title("Our LDLT vs SciPy LDLT Runtime")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "ldlt_scipy_time_comparison.png", dpi=160)
    plt.close(fig)


def label_for_ldlt_case(dataset_name, hidden_size):
    label = dataset_name.capitalize()
    return label + "\nh=" + str(hidden_size)


def estimate_real_comparison_spectral_bounds(instance):
    """Estimate L and mu using the smaller Gram matrix H^T H."""

    h_aug = instance.h_train_aug
    gram = (h_aug.T @ h_aug) / instance.n_train
    eigenvalues = np.linalg.eigvalsh(gram)
    raw_l = float(np.max(eigenvalues))
    l_smooth = max(raw_l, instance.lambda_reg) * 1.01
    return {
        "mu": instance.lambda_reg,
        "l_smooth": l_smooth,
    }


def objective_value_from_h(weights, instance):
    return objective_value(
        weights,
        instance.h_train_aug,
        instance.y_train,
        instance.lambda_reg,
    )


def elm_gradient_from_h(weights, instance):
    h_aug = instance.h_train_aug
    residual = weights @ h_aug - instance.y_train
    grad = (residual @ h_aug.T) / instance.n_train
    grad = grad + instance.lambda_reg * weights
    return grad


def heavy_ball_from_h(instance, w0, mu, l_smooth, tol, max_iter):
    sqrt_l = np.sqrt(l_smooth)
    sqrt_mu = np.sqrt(mu)
    alpha = 4.0 / (sqrt_l + sqrt_mu) ** 2
    beta = ((sqrt_l - sqrt_mu) / (sqrt_l + sqrt_mu)) ** 2

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    converged = False
    final_grad_norm = np.inf
    iterations = 0

    start = perf_counter()
    for iteration in range(max_iter + 1):
        grad = elm_gradient_from_h(weights, instance)
        final_grad_norm = float(np.sqrt(np.sum(grad * grad)))

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = weights - alpha * grad
        next_weights = next_weights + beta * (weights - previous_weights)
        previous_weights = weights
        weights = next_weights

    elapsed = perf_counter() - start

    return OptimizationResult(
        method="Heavy Ball",
        weights=weights,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
    )


def nesterov_from_h(instance, w0, mu, l_smooth, tol, max_iter):
    alpha = 1.0 / l_smooth
    beta = (np.sqrt(l_smooth) - np.sqrt(mu)) / (
        np.sqrt(l_smooth) + np.sqrt(mu)
    )

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    final_evaluation_point = weights.copy()
    converged = False
    final_grad_norm = np.inf
    iterations = 0

    start = perf_counter()
    for iteration in range(max_iter + 1):
        evaluation_point = weights + beta * (weights - previous_weights)
        grad = elm_gradient_from_h(evaluation_point, instance)
        final_grad_norm = float(np.sqrt(np.sum(grad * grad)))
        final_evaluation_point = evaluation_point

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = evaluation_point - alpha * grad
        previous_weights = weights
        weights = next_weights

    elapsed = perf_counter() - start

    return OptimizationResult(
        method="Nesterov",
        weights=final_evaluation_point,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
    )


def pytorch_from_h(
    instance,
    w0,
    method_name,
    alpha,
    beta,
    use_nesterov,
    tol,
    max_iter,
):
    if torch is None:
        raise ImportError("PyTorch is required for " + method_name + ".")

    h_tensor = torch.tensor(instance.h_train_aug, dtype=torch.float64)
    y_tensor = torch.tensor(instance.y_train, dtype=torch.float64)
    weights = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float64))

    optimizer = torch.optim.SGD(
        [weights],
        lr=float(alpha),
        momentum=float(beta),
        dampening=0.0,
        nesterov=bool(use_nesterov),
    )

    converged = False
    final_grad_norm = np.inf
    iterations = 0

    start = perf_counter()
    for iteration in range(max_iter + 1):
        optimizer.zero_grad()
        residual = weights @ h_tensor - y_tensor
        value = 0.5 * torch.sum(residual * residual) / instance.n_train
        value = value + 0.5 * instance.lambda_reg * torch.sum(weights * weights)
        value.backward()

        grad_tensor = weights.grad.detach()
        final_grad_norm = float(torch.linalg.norm(grad_tensor).item())

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        optimizer.step()

    elapsed = perf_counter() - start
    weights_numpy = weights.detach().cpu().numpy().copy()

    return OptimizationResult(
        method=method_name,
        weights=weights_numpy,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
    )


def make_requested_history_rows(
    context,
    result,
    parameter_name,
    parameter_value,
    dimensionality,
    epsilon,
):
    rows = []
    history = result.history
    iterations = history["iteration"]
    grad_norms = history["grad_norm"]
    objectives = history["objective"]
    reference_objective = context["reference_objective"]
    denominator = max(1.0, abs(reference_objective))

    for index in range(len(iterations)):
        objective_gap = max(0.0, objectives[index] - reference_objective)
        relative_gap = objective_gap / denominator

        row = {
            "method": result.method,
            "iteration": int(iterations[index]),
            "gradient_norm": grad_norms[index],
            "relative_gap": relative_gap,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "dimensionality": dimensionality,
            "epsilon": epsilon,
        }
        rows.append(row)

    return rows


def plot_requested_dimension_scaling_times(rows, figures_dir):
    if len(rows) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    dimensions = []
    for row in rows:
        value = float(row["dimensionality"])
        if value not in dimensions:
            dimensions.append(value)
    dimensions.sort()

    fig, axis = plt.subplots(figsize=(8.5, 5.0))

    for method in methods:
        times = []
        for dimension in dimensions:
            time_value = np.nan
            for row in rows:
                same_method = row["algorithm"] == method
                same_dimension = float(row["dimensionality"]) == dimension
                if same_method and same_dimension:
                    time_value = max(float(row["computational_time"]), 1e-12)
            times.append(time_value)

        axis.plot(
            dimensions,
            times,
            label=method,
            marker=marker_for_time_method(method),
            color=time_plot_color(method),
            linewidth=2.4,
        )

    axis.set_yscale("log")
    axis.set_xlabel("Output weights")
    axis.set_ylabel("Computational time (seconds)")
    axis.set_title("Requested Runtime Scaling at Fixed Epsilon")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "requested_dimension_scaling_times.png", dpi=160)
    plt.close(fig)


def remove_old_dimension_scaling_outputs(output_dir, figures_dir):
    """Delete previous dimension-scaling files that overlap this experiment."""

    csv_names = [
        "requested_dimension_scaling_times.csv",
        "requested_real_digits_scratch_iterative.csv",
    ]
    figure_names = [
        "requested_dimension_scaling_times.png",
        "analysis_dimension_scaling_times.png",
    ]

    for csv_name in csv_names:
        path = output_dir / csv_name
        if path.exists():
            path.unlink()

    for figure_name in figure_names:
        path = figures_dir / figure_name
        if path.exists():
            path.unlink()


def plot_handwritten_dimension_scaling_times(rows, figures_dir):
    if len(rows) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    dimensions = unique_numeric_values(rows, "output_weight_count")
    epsilon = float(rows[0]["epsilon"])

    fig, axis = plt.subplots(figsize=(8.8, 5.2))

    for method in methods:
        times = []
        for dimension in dimensions:
            value = np.nan
            for row in rows:
                same_method = row["algorithm"] == method
                same_dimension = (
                    float(row["output_weight_count"]) == float(dimension)
                )
                if same_method and same_dimension:
                    value = max(
                        float(row["computational_time_seconds"]),
                        1e-12,
                    )
                    break
            times.append(value)

        axis.plot(
            dimensions,
            times,
            label=method,
            marker=marker_for_time_method(method),
            color=time_plot_color(method),
            linewidth=2.4,
        )

    axis.set_yscale("log")
    axis.set_xlabel("Output weights")
    axis.set_ylabel("Computational time (seconds)")
    axis.set_title(
        "Digits Handwritten Algorithms Runtime Scaling, epsilon="
        + format_epsilon_label(epsilon)
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "handwritten_dimension_scaling_times.png",
        dpi=160,
    )
    plt.close(fig)


def plot_handwritten_dimension_scaling_histogram(rows, figures_dir):
    """Bar plot of runtime for the latest Digits scaling experiment."""

    if len(rows) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    dimensions = unique_numeric_values(rows, "output_weight_count")
    epsilon = float(rows[0]["epsilon"])

    x_values = np.arange(len(dimensions), dtype=float)
    bar_width = 0.25

    fig, axis = plt.subplots(figsize=(9.5, 5.4))

    for method_index in range(len(methods)):
        method = methods[method_index]
        times = []

        for dimension in dimensions:
            value = np.nan
            for row in rows:
                same_method = row["algorithm"] == method
                same_dimension = (
                    float(row["output_weight_count"]) == float(dimension)
                )
                if same_method and same_dimension:
                    value = max(
                        float(row["computational_time_seconds"]),
                        1e-12,
                    )
                    break
            times.append(value)

        offset = (method_index - 1) * bar_width
        axis.bar(
            x_values + offset,
            times,
            width=bar_width,
            label=method,
            color=time_plot_color(method),
            alpha=0.88,
        )

    x_labels = []
    for dimension in dimensions:
        x_labels.append(str(int(dimension)))

    axis.set_yscale("log")
    axis.set_xticks(x_values)
    axis.set_xticklabels(x_labels)
    axis.set_xlabel("Output weights")
    axis.set_ylabel("Computational time (seconds)")
    axis.set_title(
        "Digits Runtime Histogram, epsilon="
        + format_epsilon_label(epsilon)
    )
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "handwritten_dimension_scaling_time_histogram.png",
        dpi=160,
    )
    plt.close(fig)


def plot_requested_real_digits_scratch_iterative_scaling(
    rows,
    epsilon,
    path,
):
    """Rewrite the Digits scratch-iterative plot using the scaling sizes."""

    if len(rows) == 0:
        return

    dimensions = unique_numeric_values(rows, "dimensionality")
    methods = ["Heavy Ball", "Nesterov"]
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    for dim_index in range(len(dimensions)):
        dimension = dimensions[dim_index]
        color = colors(dim_index)

        for method in methods:
            selected = []
            for row in rows:
                same_method = row["method"] == method
                same_dimension = (
                    float(row["dimensionality"]) == float(dimension)
                )
                if same_method and same_dimension:
                    selected.append(row)

            selected.sort(key=lambda row: int(row["iteration"]))
            if len(selected) == 0:
                continue

            iterations = [int(row["iteration"]) for row in selected]
            relative_gaps = [
                max(float(row["relative_gap"]), 1e-16)
                for row in selected
            ]
            gradient_norms = [
                max(float(row["gradient_norm"]), 1e-16)
                for row in selected
            ]

            if method == "Heavy Ball":
                linestyle = "-"
                marker = "o"
                short_method = "HB"
            else:
                linestyle = "--"
                marker = "s"
                short_method = "Nesterov"

            label = short_method + ", weights=" + str(int(dimension))

            axes[0].semilogy(
                iterations,
                relative_gaps,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=2,
                linewidth=2.0,
                markersize=4,
            )
            axes[1].semilogy(
                iterations,
                gradient_norms,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=2,
                linewidth=2.0,
                markersize=4,
            )

    axes[0].set_title("Relative Gap")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("(f(W)-f*) / max(1, |f*|)")

    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||grad f(W)||_F")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)

    dimension_text = []
    for dimension in dimensions:
        dimension_text.append(str(int(dimension)))

    fig.suptitle(
        "Real Dataset Scratch Iterative Methods - digits / real_dataset\n"
        + "epsilon="
        + format_epsilon_label(epsilon)
        + ", output weights="
        + ", ".join(dimension_text)
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_synthetic_parameter_convergence(rows, analysis_type, path):
    """Plot HB/Nesterov convergence while sparsity or rho changes."""

    if len(rows) == 0:
        return

    if analysis_type == "sparsity":
        parameter_label = "sparseness"
    else:
        parameter_label = "rho"

    values = unique_numeric_values(rows, "parameter_value")
    methods = ["Heavy Ball", "Nesterov"]
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))

    for value_index in range(len(values)):
        value = values[value_index]
        color = colors(value_index)

        for method in methods:
            selected = []
            for row in rows:
                same_method = row["method"] == method
                same_value = (
                    abs(float(row["parameter_value"]) - float(value)) <= 1e-15
                )
                if same_method and same_value:
                    selected.append(row)

            selected.sort(key=lambda row: int(row["iteration"]))
            if len(selected) == 0:
                continue

            iterations = [int(row["iteration"]) for row in selected]
            relative_gaps = [
                max(float(row["relative_gap"]), 1e-16)
                for row in selected
            ]
            gradient_norms = [
                max(float(row["gradient_norm"]), 1e-16)
                for row in selected
            ]

            if method == "Heavy Ball":
                linestyle = "-"
                marker = "o"
                short_method = "HB"
            else:
                linestyle = "--"
                marker = "s"
                short_method = "Nesterov"

            label = (
                short_method
                + ", "
                + parameter_label
                + "="
                + format(float(value), ".1f")
            )

            axes[0].semilogy(
                iterations,
                relative_gaps,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=3,
                linewidth=1.8,
                markersize=3.5,
            )
            axes[1].semilogy(
                iterations,
                gradient_norms,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=3,
                linewidth=1.8,
                markersize=3.5,
            )

    axes[0].set_title("Relative Gap")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("(f(W)-f*) / max(1, |f*|)")

    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||grad f(W)||_F")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=6, ncol=2)

    epsilon = float(rows[0]["epsilon"])
    output_weight_count_value = int(float(rows[0]["output_weight_count"]))
    title = (
        "Synthetic "
        + analysis_type.capitalize()
        + " Sweep - Iterative Convergence\n"
        + "epsilon="
        + format_epsilon_label(epsilon)
        + ", output weights="
        + str(output_weight_count_value)
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.87])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_synthetic_parameter_convergence_by_value(
    rows,
    analysis_type,
    shared_plot_seed,
    figures_dir,
):
    """Create one convergence plot for each rho/sparsity value."""

    if len(rows) == 0:
        return

    if analysis_type == "sparsity":
        parameter_label = "sparseness"
        filename_prefix = "synthetic_sparsity_convergence_"
    else:
        parameter_label = "rho"
        filename_prefix = "synthetic_rho_convergence_"

    values = unique_numeric_values(rows, "parameter_value")
    methods = ["Heavy Ball", "Nesterov"]

    for value in values:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        value_rows = []
        for row in rows:
            same_value = (
                abs(float(row["parameter_value"]) - float(value)) <= 1e-15
            )
            if same_value:
                value_rows.append(row)

        for method in methods:
            selected = []
            for row in value_rows:
                same_method = row["method"] == method
                if same_method:
                    selected.append(row)

            selected.sort(key=lambda row: int(row["iteration"]))
            if len(selected) == 0:
                continue

            iterations = [int(row["iteration"]) for row in selected]
            relative_gaps = [
                max(float(row["relative_gap"]), 1e-16)
                for row in selected
            ]
            gradient_norms = [
                max(float(row["gradient_norm"]), 1e-16)
                for row in selected
            ]
            style = plot_style_for_method(method)

            axes[0].semilogy(
                iterations,
                relative_gaps,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=style["markevery"],
                linewidth=style["linewidth"],
            )
            axes[1].semilogy(
                iterations,
                gradient_norms,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=style["markevery"],
                linewidth=style["linewidth"],
            )

        axes[0].set_title("Relative Gap")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("(f(W)-f*) / max(1, |f*|)")

        axes[1].set_title("Gradient Norm")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("||grad f(W)||_F")

        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        epsilon = float(rows[0]["epsilon"])
        output_weight_count_value = int(float(rows[0]["output_weight_count"]))
        plot_seed = plot_seed_for_selected_rows(value_rows, shared_plot_seed)
        plot_max_iter = plot_max_iter_for_selected_rows(value_rows)
        plot_status = plot_status_for_selected_rows(value_rows)
        title = (
            "Synthetic "
            + analysis_type.capitalize()
            + " Convergence, "
            + parameter_label
            + "="
            + format(float(value), ".1f")
            + "\n"
            + "epsilon="
            + format_epsilon_label(epsilon)
            + ", output weights="
            + str(output_weight_count_value)
            + ", initialization seed="
            + str(plot_seed)
            + ", plot max_iter="
            + str(plot_max_iter)
            + ", "
            + plot_status
        )
        fig.suptitle(title)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])

        filename = (
            filename_prefix
            + format_decimal_for_name(value)
            + ".png"
        )
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def plot_seed_for_selected_rows(rows, fallback_seed):
    for row in rows:
        if "plot_initialization_seed" in row:
            value = row["plot_initialization_seed"]
            if value != "":
                return value
    return fallback_seed


def plot_max_iter_for_selected_rows(rows):
    for row in rows:
        if "plot_max_iter" in row:
            value = row["plot_max_iter"]
            if value != "":
                return value
    max_iteration = 0
    for row in rows:
        max_iteration = max(max_iteration, int(row["iteration"]))
    return max_iteration


def plot_status_for_selected_rows(rows):
    method_status = {}
    for row in rows:
        if "plot_converged" in row:
            method_status[row["method"]] = row["plot_converged"]

    if len(method_status) == 0:
        return "plot convergence status unavailable"

    all_converged = True
    for value in method_status.values():
        if value is not True:
            all_converged = False

    if all_converged:
        return "both plot runs converged"
    return "at least one plot run did not converge"


def plot_synthetic_parameter_time_histogram(rows, analysis_type, path):
    """Bar plot of mean runtime for selected sparsity/rho values."""

    if len(rows) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    selected_values = synthetic_parameter_histogram_values()
    bar_width = 0.25
    x_values = np.arange(len(selected_values), dtype=float)

    fig, axis = plt.subplots(figsize=(10.2, 5.4))

    for method_index in range(len(methods)):
        method = methods[method_index]
        times = []

        for value in selected_values:
            found_time = np.nan
            for row in rows:
                same_method = row["algorithm"] == method
                same_analysis = row["analysis_type"] == analysis_type
                same_value = (
                    abs(float(row["parameter_value"]) - float(value)) <= 1e-15
                )
                if same_analysis and same_method and same_value:
                    found_time = max(
                        float(row["mean_computational_time_seconds"]),
                        1e-12,
                    )
                    break
            times.append(found_time)

        offset = (method_index - 1) * bar_width
        axis.bar(
            x_values + offset,
            times,
            width=bar_width,
            label=method,
            color=time_plot_color(method),
            alpha=0.88,
        )

    x_labels = []
    for value in selected_values:
        x_labels.append(format(float(value), ".1f"))

    if analysis_type == "sparsity":
        xlabel = "sparseness"
    else:
        xlabel = "rho"

    selected_rows = []
    for row in rows:
        if row["analysis_type"] == analysis_type:
            selected_rows.append(row)

    epsilon = float(selected_rows[0]["epsilon"])
    output_weight_count_value = int(float(selected_rows[0]["output_weight_count"]))
    axis.set_yscale("log")
    axis.set_xticks(x_values)
    axis.set_xticklabels(x_labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Computational time (seconds)")
    axis.set_title(
        "Synthetic "
        + analysis_type.capitalize()
        + " Sweep Runtime, epsilon="
        + format_epsilon_label(epsilon)
        + ", output weights="
        + str(output_weight_count_value)
    )
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def remove_old_synthetic_parameter_outputs(output_dir, figures_dir):
    """Remove old rho/sparsity outputs that overlap the new request."""

    output_patterns = [
        "requested_dimension_rho_times.csv",
        "requested_dimension_sparsity_times.csv",
        "requested_library_rho_times.csv",
        "requested_library_sparsity_times.csv",
        "rho_conditioning_sweep_summary.csv",
        "rho_conditioning_sweep_history.csv",
        "synthetic_rho_sweep_performance.csv",
        "synthetic_rho_sweep_history.csv",
        "synthetic_sparsity_sweep_performance.csv",
        "synthetic_sparsity_sweep_history.csv",
        "synthetic_parameter_sweep_aggregate.csv",
        "synthetic_parameter_sweep_trials.csv",
    ]
    figure_patterns = [
        "requested_conditioning_rho_*.png",
        "requested_rho_time_comparison.png",
        "requested_library_rho_*.png",
        "conditioning_effect_rho_*.png",
        "rho_conditioning_*.png",
        "requested_sparsity_*.png",
        "requested_library_sparsity_*.png",
        "analysis_synthetic_sparse_parameter_epsilon_sweep.png",
        "analysis_builtin_fixed_synthetic_synthetic_sparse_*.png",
        "analysis_beta_fixed_variable_synthetic_sparse_*.png",
        "requested_beta_sparse_70_percent.png",
        "convergence_time_synthetic_sparse.png",
        "synthetic_sparse_*.png",
        "synthetic_rho_sweep_*.png",
        "synthetic_sparsity_sweep_*.png",
        "synthetic_rho_convergence_*.png",
        "synthetic_sparsity_convergence_*.png",
    ]

    remove_paths_matching(output_dir, output_patterns)
    remove_paths_matching(figures_dir, figure_patterns)


def remove_old_nesterov_beta_size_outputs(output_dir, figures_dir):
    """Remove previous fixed/variable-beta outputs before regenerating them."""

    output_patterns = [
        "nesterov_beta_size_comparison.csv",
        "nesterov_beta_size_comparison_history.csv",
    ]
    figure_patterns = [
        "requested_beta_*.png",
        "analysis_beta_fixed_variable_*.png",
    ]

    remove_paths_matching(output_dir, output_patterns)
    remove_paths_matching(figures_dir, figure_patterns)


def remove_old_nesterov_beta_focus_outputs(output_dir, figures_dir):
    """Remove previous compact 2x2 beta plots before regenerating them."""

    output_patterns = [
        "nesterov_beta_focused_comparison.csv",
        "nesterov_beta_focused_comparison_history.csv",
    ]
    figure_patterns = [
        "requested_beta_*_focused_epsilon_*.png",
    ]

    remove_paths_matching(output_dir, output_patterns)
    remove_paths_matching(figures_dir, figure_patterns)


def remove_paths_matching(base_dir, patterns):
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def plot_requested_parameter_history(
    history_rows,
    parameter_name,
    filename_prefix,
    title_prefix,
    figures_dir,
    epsilon,
):
    if len(history_rows) == 0:
        return

    values = []
    for row in history_rows:
        value = float(row["parameter_value"])
        if value not in values:
            values.append(value)
    values.sort()

    for value in values:
        selected = []
        for row in history_rows:
            if abs(float(row["parameter_value"]) - value) <= 1e-15:
                selected.append(row)

        filename = (
            filename_prefix
            + "_"
            + format_decimal_for_name(value)
            + ".png"
        )
        title = (
            title_prefix
            + " - "
            + parameter_name
            + "="
            + format(value, ".1f")
        )
        plot_requested_history_rows(
            selected,
            title,
            epsilon,
            figures_dir / filename,
        )


def plot_requested_history_rows(rows, title, epsilon, path):
    methods = ["Heavy Ball", "Nesterov"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for method in methods:
        method_rows = []
        for row in rows:
            if row["method"] == method:
                method_rows.append(row)
        method_rows.sort(key=lambda row: int(row["iteration"]))

        if len(method_rows) == 0:
            continue

        iterations = [int(row["iteration"]) for row in method_rows]
        relative_gaps = [
            max(float(row["relative_gap"]), 1e-16)
            for row in method_rows
        ]
        gradient_norms = [
            max(float(row["gradient_norm"]), 1e-16)
            for row in method_rows
        ]
        style = plot_style_for_method(method)

        axes[0].semilogy(
            iterations,
            relative_gaps,
            label=method,
            color=style["color"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )
        axes[1].semilogy(
            iterations,
            gradient_norms,
            label=method,
            color=style["color"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    dimensionality = rows[0]["dimensionality"]
    axes[0].set_title("Relative Gap")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("(f(W)-f*) / max(1, |f*|)")

    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||grad f(W)||_F")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    subtitle = (
        "epsilon="
        + format_epsilon_label(epsilon)
        + ", output weights="
        + str(int(float(dimensionality)))
    )
    fig.suptitle(title + "\n" + subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_requested_rho_time_comparison(rows, figures_dir):
    if len(rows) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    rho_values = []
    for row in rows:
        rho = float(row["rho"])
        if rho not in rho_values:
            rho_values.append(rho)
    rho_values.sort()

    fig, axis = plt.subplots(figsize=(8.5, 5.0))

    for method in methods:
        times = []
        for rho in rho_values:
            time_value = np.nan
            for row in rows:
                same_method = row["algorithm"] == method
                same_rho = abs(float(row["rho"]) - rho) <= 1e-15
                if same_method and same_rho:
                    time_value = max(float(row["computational_time"]), 1e-12)
            times.append(time_value)

        axis.plot(
            rho_values,
            times,
            label=method,
            marker=marker_for_time_method(method),
            color=time_plot_color(method),
            linewidth=2.4,
        )

    axis.set_yscale("log")
    axis.set_xlabel("rho")
    axis.set_ylabel("Computational time (seconds)")
    axis.set_title("Execution Time as Correlation rho Changes")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "requested_rho_time_comparison.png", dpi=160)
    plt.close(fig)


def remove_old_rho_conditioning_outputs(output_dir, figures_dir):
    """Remove old rho/conditioning plots that overlap this request."""

    patterns = [
        "requested_conditioning_rho_*.png",
        "requested_rho_time_comparison.png",
        "requested_library_rho_*.png",
        "rho_conditioning_final_metrics.png",
        "rho_conditioning_gradient_norm.png",
        "rho_conditioning_objective_gap.png",
        "rho_conditioning_relative_error.png",
        "conditioning_effect_rho_*.png",
        "analysis_synthetic_well_conditioned_parameter_epsilon_sweep.png",
        "analysis_synthetic_ill_conditioned_parameter_epsilon_sweep.png",
        "analysis_beta_fixed_variable_synthetic_well_conditioned_*.png",
        "analysis_beta_fixed_variable_synthetic_ill_conditioned_*.png",
        "analysis_builtin_fixed_synthetic_synthetic_well_conditioned_*.png",
        "analysis_builtin_fixed_synthetic_synthetic_ill_conditioned_*.png",
        "synthetic_well_conditioned_well_conditioned_corr_*.png",
        "synthetic_ill_conditioned_ill_conditioned_corr_*.png",
        "convergence_time_synthetic_well_conditioned.png",
        "convergence_time_synthetic_ill_conditioned.png",
        "conditioning_overview.png",
    ]

    for pattern in patterns:
        paths = list(figures_dir.glob(pattern))
        for path in paths:
            path.unlink()


def plot_requested_conditioning_effect(history_rows, summary_rows, figures_dir):
    """Create one two-panel plot for each requested rho value."""

    if len(history_rows) == 0:
        return

    rho_values = unique_numeric_values(history_rows, "rho")

    for rho in rho_values:
        selected = []
        for row in history_rows:
            if abs(float(row["rho"]) - rho) <= 1e-15:
                selected.append(row)

        if len(selected) == 0:
            continue

        epsilon = float(selected[0]["epsilon"])
        output_weights = int(float(selected[0]["output_weight_count"]))
        condition_number = float(selected[0]["condition_number"])

        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
        methods = ["Heavy Ball", "Nesterov"]

        for method in methods:
            method_rows = []
            for row in selected:
                if row["method"] == method:
                    method_rows.append(row)
            method_rows.sort(key=lambda row: int(row["iteration"]))

            if len(method_rows) == 0:
                continue

            iterations = [int(row["iteration"]) for row in method_rows]
            relative_gaps = [
                max(float(row["relative_gap"]), 1e-300)
                for row in method_rows
            ]
            gradient_norms = [
                max(float(row["grad_norm"]), 1e-300)
                for row in method_rows
            ]
            style = plot_style_for_method(method)

            axes[0].semilogy(
                iterations,
                relative_gaps,
                label=method,
                color=style["color"],
                marker=style["marker"],
                markevery=style["markevery"],
                linewidth=style["linewidth"],
            )
            axes[1].semilogy(
                iterations,
                gradient_norms,
                label=method,
                color=style["color"],
                marker=style["marker"],
                markevery=style["markevery"],
                linewidth=style["linewidth"],
            )

        axes[0].set_title("Relative Gap")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("(f(W) - f*) / max(1, |f*|)")

        axes[1].set_title("Gradient Norm")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("||grad f(W)||_F")

        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        subtitle = (
            "rho="
            + format(rho, ".1f")
            + ", epsilon="
            + format_epsilon_label(epsilon)
            + ", n_train=1000, output weights="
            + str(output_weights)
            + ", kappa~"
            + format(condition_number, ".2e")
        )
        fig.suptitle("Conditioning Number Effect\n" + subtitle)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])

        filename = (
            "conditioning_effect_rho_"
            + format_decimal_for_name(rho)
            + ".png"
        )
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def plot_rho_conditioning_final_metrics(rows, figures_dir):
    if len(rows) == 0:
        return

    rho_values = unique_numeric_values(rows, "rho")
    methods = ["Heavy Ball", "Nesterov"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    flat_axes = axes.ravel()

    condition_values = []
    for rho in rho_values:
        value = np.nan
        for row in rows:
            same_rho = abs(float(row["rho"]) - rho) <= 1e-15
            if same_rho:
                value = float(row["condition_number"])
                break
        condition_values.append(value)

    flat_axes[0].plot(
        rho_values,
        condition_values,
        marker="o",
        linewidth=2.2,
        color="#4c78a8",
    )
    flat_axes[0].set_yscale("log")
    flat_axes[0].set_title("Condition Number")
    flat_axes[0].set_xlabel("rho")
    flat_axes[0].set_ylabel("estimated L / mu")

    plot_rho_summary_metric(
        flat_axes[1],
        rows,
        rho_values,
        methods,
        "time_seconds",
        "Computational Time",
        "seconds",
        True,
    )
    plot_rho_summary_metric(
        flat_axes[2],
        rows,
        rho_values,
        methods,
        "final_gradient_norm",
        "Final Gradient Norm",
        "||grad f(W)||_F",
        True,
    )
    plot_rho_summary_metric(
        flat_axes[3],
        rows,
        rho_values,
        methods,
        "relative_error_to_reference",
        "Relative Error to Reference",
        "||W - W*|| / ||W*||",
        True,
    )

    for axis in flat_axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.set_xticks(rho_values)

    fig.suptitle("Rho Conditioning Sweep, Hidden Width = 10000")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(figures_dir / "rho_conditioning_final_metrics.png", dpi=160)
    plt.close(fig)


def plot_rho_summary_metric(
    axis,
    rows,
    rho_values,
    methods,
    field,
    title,
    ylabel,
    use_log_scale,
):
    for method in methods:
        y_values = []
        for rho in rho_values:
            value = np.nan
            for row in rows:
                same_method = row["method"] == method
                same_rho = abs(float(row["rho"]) - rho) <= 1e-15
                if same_method and same_rho:
                    value = max(float(row[field]), 1e-16)
                    break
            y_values.append(value)

        style = plot_style_for_method(method)
        axis.plot(
            rho_values,
            y_values,
            marker=style["marker"],
            linewidth=2.3,
            color=style["color"],
            label=method,
        )

    if use_log_scale:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("rho")
    axis.set_ylabel(ylabel)
    axis.legend()


def plot_rho_conditioning_history_metric(
    rows,
    field,
    title,
    ylabel,
    path,
):
    if len(rows) == 0:
        return

    rho_values = unique_numeric_values(rows, "rho")
    methods = ["Heavy Ball", "Nesterov"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho_values)))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)

    for method_index in range(len(methods)):
        method = methods[method_index]
        axis = axes[method_index]

        for rho_index in range(len(rho_values)):
            rho = rho_values[rho_index]
            selected = []
            for row in rows:
                same_method = row["method"] == method
                same_rho = abs(float(row["rho"]) - rho) <= 1e-15
                if same_method and same_rho:
                    selected.append(row)

            selected.sort(key=lambda row: int(row["iteration"]))
            if len(selected) == 0:
                continue

            iterations = [int(row["iteration"]) for row in selected]
            values = [max(float(row[field]), 1e-16) for row in selected]
            axis.semilogy(
                iterations,
                values,
                label="rho=" + format(rho, ".1f"),
                color=colors[rho_index],
                linewidth=1.9,
            )

        axis.set_title(method)
        axis.set_xlabel("iteration")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title + " as rho changes, hidden width = 10000")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_requested_library_pair(
    context,
    scratch_result,
    library_result,
    parameter_name,
    parameter_value,
    dimensionality,
    epsilon,
    path,
):
    title = (
        scratch_result.method
        + " vs "
        + library_result.method
        + " - "
        + parameter_name
        + "="
        + format(float(parameter_value), ".1f")
    )
    history_rows = []
    history_rows.extend(
        make_requested_history_rows(
            context,
            scratch_result,
            parameter_name,
            parameter_value,
            dimensionality,
            epsilon,
        )
    )
    history_rows.extend(
        make_requested_history_rows(
            context,
            library_result,
            parameter_name,
            parameter_value,
            dimensionality,
            epsilon,
        )
    )
    plot_requested_history_rows_for_methods(
        history_rows,
        [scratch_result.method, library_result.method],
        title,
        epsilon,
        path,
    )


def plot_requested_two_results(context, results, epsilon, path, title_prefix):
    history_rows = []
    dimensionality = output_weight_count(context["instance"])

    for result in results:
        history_rows.extend(
            make_requested_history_rows(
                context,
                result,
                "case",
                0.0,
                dimensionality,
                epsilon,
            )
        )

    title = (
        title_prefix
        + " - "
        + context["scenario"]["dataset_name"]
        + " / "
        + context["scenario"]["scenario_type"]
    )
    method_names = []
    for result in results:
        method_names.append(result.method)

    plot_requested_history_rows_for_methods(
        history_rows,
        method_names,
        title,
        epsilon,
        path,
    )


def plot_requested_history_rows_for_methods(
    rows,
    methods,
    title,
    epsilon,
    path,
):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for method in methods:
        method_rows = []
        for row in rows:
            if row["method"] == method:
                method_rows.append(row)
        method_rows.sort(key=lambda row: int(row["iteration"]))

        if len(method_rows) == 0:
            continue

        iterations = [int(row["iteration"]) for row in method_rows]
        relative_gaps = [
            max(float(row["relative_gap"]), 1e-16)
            for row in method_rows
        ]
        gradient_norms = [
            max(float(row["gradient_norm"]), 1e-16)
            for row in method_rows
        ]
        style = plot_style_for_method(method)

        axes[0].semilogy(
            iterations,
            relative_gaps,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )
        axes[1].semilogy(
            iterations,
            gradient_norms,
            label=method,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    dimensionality = rows[0]["dimensionality"]
    axes[0].set_title("Relative Gap")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("(f(W)-f*) / max(1, |f*|)")

    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||grad f(W)||_F")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    subtitle = (
        "epsilon="
        + format_epsilon_label(epsilon)
        + ", output weights="
        + str(int(float(dimensionality)))
    )
    fig.suptitle(title + "\n" + subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def numpy_solve_reference(q, c):
    """Built-in reference using np.linalg.solve.

    This is deliberately kept outside the project algorithms. Its role is to
    verify whether the hand-written LDLT solution is numerically correct.
    """

    start = perf_counter()
    weights = np.linalg.solve(q, c.T).T
    elapsed = perf_counter() - start

    grad = weights @ q - c
    grad_norm = float(np.sqrt(np.sum(grad * grad)))

    return OptimizationResult(
        method="NumPy solve",
        weights=weights,
        iterations=0,
        converged=True,
        elapsed_seconds=elapsed,
        final_gradient_norm=grad_norm,
    )


def scipy_ldlt_reference(q, c):
    """Built-in SciPy LDLT benchmark.

    This is used only as a reference implementation of the same factorization
    family as our hand-written LDLT. The project algorithms do not call SciPy.
    """

    start = perf_counter()
    weights = solve_with_scipy_ldlt(q, c.T).T
    elapsed = perf_counter() - start

    grad = weights @ q - c
    grad_norm = float(np.sqrt(np.sum(grad * grad)))

    return OptimizationResult(
        method="SciPy LDLT",
        weights=weights,
        iterations=0,
        converged=True,
        elapsed_seconds=elapsed,
        final_gradient_norm=grad_norm,
    )


def solve_with_scipy_ldlt(matrix, rhs):
    """Solve matrix * X = rhs using scipy.linalg.ldl."""

    check_scipy_available()

    rhs_array = np.asarray(rhs, dtype=float)
    was_vector = False
    if rhs_array.ndim == 1:
        rhs_array = rhs_array[:, None]
        was_vector = True

    lu, d_matrix, perm = scipy_ldl(
        matrix,
        lower=True,
        hermitian=True,
        check_finite=True,
    )

    lower_factor = lu[perm, :]
    rhs_permuted = rhs_array[perm, :]

    z = scipy_solve_triangular(
        lower_factor,
        rhs_permuted,
        lower=True,
        unit_diagonal=True,
        check_finite=False,
    )

    diagonal_only = np.diag(np.diag(d_matrix))
    off_diagonal = d_matrix - diagonal_only
    if np.max(np.abs(off_diagonal)) <= 1e-14:
        diagonal_values = np.diag(d_matrix)
        y = z / diagonal_values[:, None]
    else:
        # This branch handles the rare case where SciPy returns 2x2 blocks.
        y = np.linalg.solve(d_matrix, z)

    solution_permuted = scipy_solve_triangular(
        lower_factor.T,
        y,
        lower=False,
        unit_diagonal=True,
        check_finite=False,
    )

    solution = np.empty_like(solution_permuted)
    solution[perm, :] = solution_permuted

    if was_vector:
        return solution[:, 0]
    return solution


def nesterov_variable_beta(
    q,
    c,
    w0,
    l_smooth,
    tol,
    max_iter,
    objective_fn,
    reference_weights,
    record_every,
):
    """Run Nesterov with the classical variable beta sequence.

    This version is useful as a comparison because it is usually presented for
    general convex functions. The project claim is that the constant-beta
    strongly-convex version is more appropriate for our regularized ELM
    quadratic problem.
    """

    alpha = 1.0 / l_smooth
    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    t_value = 1.0
    history = new_local_history()

    start = perf_counter()
    converged = False
    final_grad_norm = np.inf
    iterations = 0
    final_evaluation_point = weights.copy()
    beta = 0.0

    for iteration in range(max_iter + 1):
        if iteration == 0:
            beta = 0.0
        else:
            beta = (t_value - 1.0) / next_t_value(t_value)

        evaluation_point = weights + beta * (weights - previous_weights)
        grad = evaluation_point @ q - c
        final_grad_norm = float(np.sqrt(np.sum(grad * grad)))
        final_evaluation_point = evaluation_point

        if iteration % record_every == 0:
            record_local_history(
                history,
                iteration,
                evaluation_point,
                final_grad_norm,
                objective_fn,
                reference_weights,
            )

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = evaluation_point - alpha * grad
        next_t = next_t_value(t_value)

        previous_weights = weights
        weights = next_weights
        t_value = next_t

    elapsed = perf_counter() - start

    return OptimizationResult(
        method="Nesterov variable beta",
        weights=final_evaluation_point,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
        history=history,
    )


def next_t_value(t_value):
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value))


def pytorch_sgd_reference(
    q,
    c,
    w0,
    method_name,
    alpha,
    beta,
    use_nesterov,
    tol,
    max_iter,
    objective_fn,
    reference_weights,
    record_every,
):
    """Run a PyTorch optimizer as an external library reference.

    PyTorch is not used by the project algorithms. It is included here only to
    compare our hand-written Heavy Ball and Nesterov implementations with a
    well-known library implementation of momentum methods.
    """

    if torch is None:
        raise ImportError("PyTorch is required for " + method_name + ".")

    q_tensor = torch.tensor(q, dtype=torch.float64)
    c_tensor = torch.tensor(c, dtype=torch.float64)
    start_weights = torch.tensor(w0, dtype=torch.float64)
    weights = torch.nn.Parameter(start_weights)

    optimizer = torch.optim.SGD(
        [weights],
        lr=float(alpha),
        momentum=float(beta),
        dampening=0.0,
        nesterov=bool(use_nesterov),
    )

    history = new_local_history()
    start = perf_counter()
    converged = False
    iterations = 0
    final_grad_norm = np.inf

    for iteration in range(max_iter + 1):
        optimizer.zero_grad()

        # This quadratic has gradient W Q - C, the same gradient used by our
        # scratch methods. The missing constant does not change the optimizer.
        value = 0.5 * torch.sum((weights @ q_tensor) * weights)
        value = value - torch.sum(weights * c_tensor)
        value.backward()

        grad_tensor = weights.grad.detach()
        final_grad_norm = float(torch.linalg.norm(grad_tensor).item())
        weights_numpy = weights.detach().cpu().numpy().copy()

        if iteration % record_every == 0:
            record_local_history(
                history,
                iteration,
                weights_numpy,
                final_grad_norm,
                objective_fn,
                reference_weights,
            )

        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break

        if iteration == max_iter:
            iterations = iteration
            break

        optimizer.step()

    elapsed = perf_counter() - start
    weights_numpy = weights.detach().cpu().numpy().copy()

    return OptimizationResult(
        method=method_name,
        weights=weights_numpy,
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        final_gradient_norm=final_grad_norm,
        alpha=alpha,
        beta=beta,
        history=history,
    )


def new_local_history():
    history = {
        "iteration": [],
        "grad_norm": [],
        "objective": [],
        "relative_error": [],
    }
    return history


def record_local_history(
    history,
    iteration,
    weights,
    grad_norm,
    objective_fn,
    reference_weights,
):
    history["iteration"].append(float(iteration))
    history["grad_norm"].append(float(grad_norm))
    history["objective"].append(float(objective_fn(weights)))
    history["relative_error"].append(relative_error(weights, reference_weights))


def make_conditioning_row(scenario, instance, spectral):
    """Create the table row with conditioning information for one instance."""

    q = instance.q
    q_dimension = q.shape[0]

    # Exact eigenvalues are diagnostic only. The algorithms use the Power
    # Method estimate and the lambda lower bound.
    exact_min = ""
    exact_max = ""
    exact_condition = ""
    if q_dimension <= 300:
        eigenvalues = np.linalg.eigvalsh(q)
        exact_min = float(np.min(eigenvalues))
        exact_max = float(np.max(eigenvalues))
        exact_condition = exact_max / exact_min

    row = {
        "dataset_name": scenario["dataset_name"],
        "scenario_type": scenario["scenario_type"],
        "correlation_strength": scenario["correlation_strength"],
        "zero_probability": scenario["zero_probability"],
        "scenario_description": scenario["description"],
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "hidden_width": instance.hidden_width,
        "total_output_weights": output_weight_count(instance),
        "q_dimension": q_dimension,
        "lambda_reg": instance.lambda_reg,
        "power_method_lambda_max_estimate": spectral.raw_largest_eigenvalue_estimate,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "estimated_condition_number": spectral.condition_estimate,
        "power_iterations": spectral.power_iterations,
        "exact_lambda_min_diagnostic": exact_min,
        "exact_lambda_max_diagnostic": exact_max,
        "exact_condition_number_diagnostic": exact_condition,
    }
    return row


def summarize_result(
    scenario,
    instance,
    result,
    spectral,
    reference_weights,
    ldlt_weights,
    objective_fn,
):
    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug
    objective = objective_fn(result.weights)

    grad = gradient(result.weights, instance.q, instance.c)
    final_grad_norm = float(np.sqrt(np.sum(grad * grad)))
    initial_grad_norm = float(np.sqrt(np.sum(instance.c * instance.c)))
    reduction = final_grad_norm / max(initial_grad_norm, 1e-300)

    method_type = "scratch"
    if result.method in ["NumPy solve", "SciPy LDLT"]:
        method_type = "built-in benchmark"
    if result.method in ["PyTorch SGD momentum", "PyTorch SGD Nesterov"]:
        method_type = "library optimizer"

    row = {
        "dataset_name": scenario["dataset_name"],
        "scenario_type": scenario["scenario_type"],
        "correlation_strength": scenario["correlation_strength"],
        "zero_probability": scenario["zero_probability"],
        "method": result.method,
        "method_type": method_type,
        "beta_rule": beta_rule_for_method(result.method),
        "condition_number": spectral.condition_estimate,
        "converged": result.converged,
        "iterations": result.iterations,
        "time_seconds": result.elapsed_seconds,
        "initial_gradient_norm": initial_grad_norm,
        "final_gradient_norm": final_grad_norm,
        "gradient_reduction_factor": reduction,
        "objective_value": objective,
        "train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
        "train_mse": mean_squared_error(train_scores, instance.y_train),
        "test_mse": mean_squared_error(test_scores, instance.y_test),
        "relative_error_to_reference": relative_error(
            result.weights,
            reference_weights,
        ),
        "relative_error_to_ldlt": relative_error(result.weights, ldlt_weights),
        "q_dimension": instance.q.shape[0],
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "power_method_lambda_max_estimate": spectral.raw_largest_eigenvalue_estimate,
        "mu": spectral.mu,
        "activation": instance.activation,
        "hidden_width": instance.hidden_width,
        "total_output_weights": output_weight_count(instance),
        "alpha": "" if result.alpha is None else result.alpha,
        "beta": "" if result.beta is None else result.beta,
    }
    return row


def beta_rule_for_method(method):
    if method == "Heavy Ball":
        return "constant momentum"
    if method == "Nesterov":
        return "constant beta"
    if method == "Nesterov variable beta":
        return "variable beta"
    if method == "PyTorch SGD momentum":
        return "library constant momentum"
    if method == "PyTorch SGD Nesterov":
        return "library nesterov"
    return ""


def history_rows_for_result(scenario, result, reference_objective):
    rows = []

    iterations = result.history["iteration"]
    grad_norms = result.history["grad_norm"]
    objectives = result.history["objective"]
    relative_errors = result.history["relative_error"]

    for index in range(len(iterations)):
        objective = objectives[index]
        gap = max(0.0, objective - reference_objective)

        row = {
            "dataset_name": scenario["dataset_name"],
            "scenario_type": scenario["scenario_type"],
            "correlation_strength": scenario["correlation_strength"],
            "zero_probability": scenario["zero_probability"],
            "method": result.method,
            "iteration": int(iterations[index]),
            "grad_norm": grad_norms[index],
            "objective_value": objective,
            "objective_gap_to_reference": gap,
            "relative_error_to_reference": relative_errors[index],
        }
        rows.append(row)

    return rows


def write_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence_by_scenario(history_rows, figures_dir):
    """Save one main convergence plot for each scenario.

    The main plot compares our hand-written accelerated methods with the
    corresponding PyTorch library optimizers. The variable-beta Nesterov method
    is intentionally not included here; it has its own focused plot below.
    """

    scenario_keys = []
    for row in history_rows:
        key = (row["dataset_name"], row["scenario_type"])
        if key not in scenario_keys:
            scenario_keys.append(key)

    for dataset_name, scenario_type in scenario_keys:
        selected_rows = []
        for row in history_rows:
            same_dataset = row["dataset_name"] == dataset_name
            same_scenario = row["scenario_type"] == scenario_type
            if same_dataset and same_scenario:
                selected_rows.append(row)

        if len(selected_rows) == 0:
            continue

        methods = main_plot_methods()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

        for method in methods:
            method_rows = []
            for row in selected_rows:
                if row["method"] == method:
                    method_rows.append(row)

            if len(method_rows) == 0:
                continue

            method_rows.sort(key=lambda row: int(row["iteration"]))

            iterations = [int(row["iteration"]) for row in method_rows]
            grad_norms = [
                max(float(row["grad_norm"]), 1e-300)
                for row in method_rows
            ]
            objective_gaps = [
                max(float(row["objective_gap_to_reference"]), 1e-300)
                for row in method_rows
            ]
            relative_errors = [
                max(float(row["relative_error_to_reference"]), 1e-300)
                for row in method_rows
            ]

            style = plot_style_for_method(method)
            plot_iterations = shifted_iterations_for_plot(iterations, method)
            axes[0].semilogy(
                plot_iterations,
                grad_norms,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markevery=style["markevery"],
                alpha=style["alpha"],
            )
            axes[1].semilogy(
                plot_iterations,
                objective_gaps,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markevery=style["markevery"],
                alpha=style["alpha"],
            )
            axes[2].semilogy(
                plot_iterations,
                relative_errors,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markevery=style["markevery"],
                alpha=style["alpha"],
            )

        axes[0].set_title("Gradient Norm")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("||grad f(W)||_F")

        axes[1].set_title("Objective Gap")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("f(W) - f(W_ref)")

        axes[2].set_title("Relative Error")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("||W - W_ref|| / ||W_ref||")

        title = dataset_name + " - " + scenario_type + " - main comparison"
        fig.suptitle(title)

        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        fig.text(
            0.5,
            0.01,
            "Horizontal display offsets separate overlapping curves; CSV values are unchanged.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
        filename = make_safe_filename(dataset_name + "_" + scenario_type)
        fig.savefig(figures_dir / (filename + "_convergence.png"), dpi=160)
        plt.close(fig)


def plot_nesterov_beta_comparison(history_rows, figures_dir):
    """Plot only fixed-beta Nesterov against variable-beta Nesterov.

    This plot is separate because it supports a specific theoretical claim:
    for strongly convex quadratics, the constant-beta version is the natural
    accelerated method.
    """

    scenario_keys = []
    for row in history_rows:
        key = (row["dataset_name"], row["scenario_type"])
        if key not in scenario_keys:
            scenario_keys.append(key)

    for dataset_name, scenario_type in scenario_keys:
        selected_rows = []
        for row in history_rows:
            same_dataset = row["dataset_name"] == dataset_name
            same_scenario = row["scenario_type"] == scenario_type
            is_beta_method = row["method"] in [
                "Nesterov",
                "Nesterov variable beta",
            ]
            if same_dataset and same_scenario and is_beta_method:
                selected_rows.append(row)

        if len(selected_rows) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        methods = ["Nesterov variable beta", "Nesterov"]

        for method in methods:
            method_rows = []
            for row in selected_rows:
                if row["method"] == method:
                    method_rows.append(row)

            if len(method_rows) == 0:
                continue

            method_rows.sort(key=lambda row: int(row["iteration"]))

            iterations = [int(row["iteration"]) for row in method_rows]
            grad_norms = [
                max(float(row["grad_norm"]), 1e-300)
                for row in method_rows
            ]
            objective_gaps = [
                max(float(row["objective_gap_to_reference"]), 1e-300)
                for row in method_rows
            ]
            relative_errors = [
                max(float(row["relative_error_to_reference"]), 1e-300)
                for row in method_rows
            ]

            style = plot_style_for_method(method)
            axes[0].semilogy(
                iterations,
                grad_norms,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markevery=style["markevery"],
                alpha=style["alpha"],
            )
            axes[1].semilogy(
                iterations,
                objective_gaps,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markevery=style["markevery"],
                alpha=style["alpha"],
            )
            axes[2].semilogy(
                iterations,
                relative_errors,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markevery=style["markevery"],
                alpha=style["alpha"],
            )

        axes[0].set_title("Gradient Norm")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("||grad f(W)||_F")

        axes[1].set_title("Objective Gap")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("f(W) - f(W_ref)")

        axes[2].set_title("Relative Error")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("||W - W_ref|| / ||W_ref||")

        title = dataset_name + " - " + scenario_type + " - Nesterov beta"
        fig.suptitle(title)

        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        fig.tight_layout()
        filename = make_safe_filename(dataset_name + "_" + scenario_type)
        fig.savefig(
            figures_dir / (filename + "_nesterov_beta_comparison.png"),
            dpi=160,
        )
        plt.close(fig)


def main_plot_methods():
    """Return methods in drawing order.

    PyTorch curves are drawn first because they often coincide numerically with
    our hand-written curves. Drawing our methods last makes them visible.
    """

    return [
        "PyTorch SGD momentum",
        "PyTorch SGD Nesterov",
        "Heavy Ball",
        "Nesterov",
    ]


def plot_style_for_method(method):
    if method == "Heavy Ball":
        return {
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 2.8,
            "marker": "o",
            "markevery": 2,
            "alpha": 1.0,
        }
    if method == "Nesterov":
        return {
            "color": "tab:orange",
            "linestyle": "-",
            "linewidth": 2.8,
            "marker": "s",
            "markevery": 2,
            "alpha": 1.0,
        }
    if method == "Nesterov variable beta":
        return {
            "color": "tab:green",
            "linestyle": "--",
            "linewidth": 2.2,
            "marker": "^",
            "markevery": 2,
            "alpha": 0.95,
        }
    if method == "PyTorch SGD momentum":
        return {
            "color": "tab:cyan",
            "linestyle": "--",
            "linewidth": 2.4,
            "marker": "x",
            "markevery": 2,
            "alpha": 0.95,
        }
    if method == "PyTorch SGD Nesterov":
        return {
            "color": "tab:red",
            "linestyle": "--",
            "linewidth": 2.4,
            "marker": "D",
            "markevery": 2,
            "alpha": 0.95,
        }

    return {
        "color": None,
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "",
        "markevery": 1,
        "alpha": 1.0,
    }


def shifted_iterations_for_plot(iterations, method):
    """Add a display-only shift so coincident curves are all visible."""

    offset = 0.0
    if method == "PyTorch SGD momentum":
        offset = 0.0
    elif method == "Heavy Ball":
        offset = 8.0
    elif method == "PyTorch SGD Nesterov":
        offset = 16.0
    elif method == "Nesterov":
        offset = 24.0

    shifted = []
    for iteration in iterations:
        shifted.append(iteration + offset)
    return shifted


def plot_conditioning_overview(conditioning_rows, figures_dir):
    """Plot the estimated condition number of each generated Q matrix."""

    if len(conditioning_rows) == 0:
        return

    labels = []
    values = []
    for row in conditioning_rows:
        label = row["dataset_name"] + "\n" + row["scenario_type"]
        labels.append(label)
        values.append(float(row["estimated_condition_number"]))

    figure_width = max(8.0, 0.45 * len(labels))
    fig, axis = plt.subplots(figsize=(figure_width, 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, values)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Estimated condition number L / mu")
    axis.set_title("Conditioning of the ELM Normal Matrices")
    axis.grid(True, axis="y", which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "conditioning_overview.png", dpi=160)
    plt.close(fig)


def plot_convergence_time_bars(summary_rows, figures_dir):
    """Create bar plots with the elapsed time of every method.

    For iterative methods, a hatched bar means that the method stopped at
    max_iter before reaching the requested tolerance. In that case the time is
    still useful, but it is not a true convergence time.
    """

    if len(summary_rows) == 0:
        return

    plot_convergence_time_group(
        summary_rows,
        figures_dir / "convergence_time_all_cases.png",
        "Convergence Time - All Test Cases",
    )

    dataset_names = []
    for row in summary_rows:
        dataset_name = row["dataset_name"]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)

    for dataset_name in dataset_names:
        selected_rows = []
        for row in summary_rows:
            if row["dataset_name"] == dataset_name:
                selected_rows.append(row)

        filename = "convergence_time_" + make_safe_filename(dataset_name) + ".png"
        title = "Convergence Time - " + dataset_name
        plot_convergence_time_group(
            selected_rows,
            figures_dir / filename,
            title,
        )


def plot_convergence_time_group(rows, path, title):
    """Plot elapsed times for one group of scenarios."""

    scenario_keys = []
    for row in rows:
        key = (row["dataset_name"], row["scenario_type"])
        if key not in scenario_keys:
            scenario_keys.append(key)

    methods = convergence_time_methods()
    n_scenarios = len(scenario_keys)
    n_methods = len(methods)

    if n_scenarios == 0:
        return

    x_positions = np.arange(n_scenarios)
    total_width = 0.86
    bar_width = total_width / n_methods
    figure_width = max(10.0, 0.75 * n_scenarios)
    fig, axis = plt.subplots(figsize=(figure_width, 6.2))

    for method_index in range(n_methods):
        method = methods[method_index]
        values = []
        converged_flags = []

        for key in scenario_keys:
            row = find_summary_row(rows, key, method)
            if row is None:
                values.append(np.nan)
                converged_flags.append(True)
            else:
                time_value = float(row["time_seconds"])
                values.append(max(time_value, 1e-12))
                converged_flags.append(as_boolean(row["converged"]))

        offset = -0.5 * total_width + method_index * bar_width + 0.5 * bar_width
        bar_positions = x_positions + offset
        bars = axis.bar(
            bar_positions,
            values,
            width=bar_width,
            label=method,
            color=time_plot_color(method),
            edgecolor="black",
            linewidth=0.25,
        )

        for index in range(len(bars)):
            if not converged_flags[index]:
                bars[index].set_hatch("//")
                bars[index].set_edgecolor("black")
                bars[index].set_linewidth(0.8)

    labels = []
    for key in scenario_keys:
        labels.append(label_for_time_plot(key))

    axis.set_yscale("log")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Elapsed seconds (log scale)")
    axis.set_title(title)
    axis.grid(True, axis="y", which="both", alpha=0.3)

    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="//",
            label="stopped at max_iter",
        )
    )
    labels.append("stopped at max_iter")
    axis.legend(handles, labels, fontsize=8, ncol=2)

    fig.text(
        0.5,
        0.01,
        "For non-converged iterative methods, the bar is runtime until max_iter.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def find_summary_row(rows, scenario_key, method):
    for row in rows:
        key = (row["dataset_name"], row["scenario_type"])
        if key == scenario_key and row["method"] == method:
            return row
    return None


def convergence_time_methods():
    return [
        "LDLT",
        "Heavy Ball",
        "Nesterov",
        "Nesterov variable beta",
        "PyTorch SGD momentum",
        "PyTorch SGD Nesterov",
        "NumPy solve",
        "SciPy LDLT",
    ]


def time_plot_color(method):
    colors = {
        "LDLT": "#4c78a8",
        "Heavy Ball": "#1f77b4",
        "Nesterov": "#ff7f0e",
        "Nesterov variable beta": "#2ca02c",
        "PyTorch SGD momentum": "#17becf",
        "PyTorch SGD Nesterov": "#d62728",
        "NumPy solve": "#9467bd",
        "SciPy LDLT": "#8c564b",
    }
    if method in colors:
        return colors[method]
    return "#7f7f7f"


def label_for_time_plot(key):
    dataset_name = key[0]
    scenario_type = key[1]

    if "corr_" in scenario_type:
        value = scenario_type.split("corr_")[-1].replace("_", ".")
        if dataset_name == "synthetic_well_conditioned":
            return "well\nrho=" + value
        if dataset_name == "synthetic_ill_conditioned":
            return "ill\nrho=" + value

    if "zero_" in scenario_type:
        value = scenario_type.split("zero_")[-1].replace("_", ".")
        return "sparse\nzero=" + value

    return dataset_name


def as_boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def plot_dimension_scaling_times(rows, figures_dir):
    """Plot runtime at fixed epsilon while the problem size increases."""

    selected = []
    for row in rows:
        if row["analysis_group"] == "dimension_scaling":
            selected.append(row)

    if len(selected) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    x_values = unique_numeric_values(selected, "sweep_value")

    fig, axis = plt.subplots(figsize=(8.5, 5.0))

    for method in methods:
        y_values = []
        for value in x_values:
            row = find_analysis_row(
                selected,
                method,
                "sweep_value",
                value,
            )
            if row is None:
                y_values.append(np.nan)
            else:
                y_values.append(max(float(row["time_seconds"]), 1e-12))

        axis.plot(
            x_values,
            y_values,
            label=method,
            marker=marker_for_time_method(method),
            linewidth=2.4,
            color=time_plot_color(method),
        )

    axis.set_yscale("log")
    axis.set_xlabel("Output weights")
    axis.set_ylabel("Elapsed seconds (log scale)")
    axis.set_title("Runtime Scaling at Fixed Epsilon")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "analysis_dimension_scaling_times.png", dpi=160)
    plt.close(fig)


def plot_synthetic_parameter_epsilon_sweeps(rows, figures_dir):
    """Plot final iterative metrics while rho/sparsity and epsilon vary."""

    selected = []
    for row in rows:
        if row["analysis_group"] == "synthetic_parameter_epsilon":
            if row["method"] in ["Heavy Ball", "Nesterov"]:
                selected.append(row)

    if len(selected) == 0:
        return

    dataset_names = unique_values(selected, "dataset_name")
    metric_list = detailed_metric_list()

    for dataset_name in dataset_names:
        dataset_rows = []
        for row in selected:
            if row["dataset_name"] == dataset_name:
                dataset_rows.append(row)

        if len(dataset_rows) == 0:
            continue

        parameter = dataset_rows[0]["sweep_parameter"]
        parameter_values = unique_numeric_values(dataset_rows, "sweep_value")
        epsilons = unique_numeric_values(dataset_rows, "epsilon")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for metric_index in range(len(metric_list)):
            field, title, ylabel = metric_list[metric_index]
            axis = axes[metric_index]

            for epsilon in epsilons:
                for method in ["Heavy Ball", "Nesterov"]:
                    y_values = []
                    for value in parameter_values:
                        row = find_analysis_row_with_epsilon(
                            dataset_rows,
                            method,
                            epsilon,
                            value,
                        )
                        if row is None:
                            y_values.append(np.nan)
                        else:
                            y_values.append(
                                max(float(row[field]), metric_floor(field))
                            )

                    style = plot_style_for_method(method)
                    label = method + ", eps=" + format_epsilon_label(epsilon)
                    axis.plot(
                        parameter_values,
                        y_values,
                        label=label,
                        color=style["color"],
                        linestyle=line_style_for_epsilon(epsilon, epsilons),
                        marker=style["marker"],
                        linewidth=2.0,
                    )

            axis.set_yscale("log")
            axis.set_xlabel(parameter)
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=8)

        fig.suptitle(dataset_name + " - epsilon and " + parameter + " sweep")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        filename = "analysis_" + make_safe_filename(dataset_name)
        filename = filename + "_parameter_epsilon_sweep.png"
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def plot_synthetic_ldlt_times(rows, figures_dir):
    """Plot LDLT time while correlation or sparsity changes."""

    selected = []
    seen = []

    for row in rows:
        if row["analysis_group"] != "synthetic_parameter_epsilon":
            continue
        if row["method"] != "LDLT":
            continue

        key = (row["dataset_name"], row["scenario_type"])
        if key in seen:
            continue

        seen.append(key)
        selected.append(row)

    if len(selected) == 0:
        return

    dataset_names = unique_values(selected, "dataset_name")
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(5.2 * len(dataset_names), 4.5))

    if len(dataset_names) == 1:
        axes = [axes]

    for index in range(len(dataset_names)):
        dataset_name = dataset_names[index]
        axis = axes[index]
        dataset_rows = []
        for row in selected:
            if row["dataset_name"] == dataset_name:
                dataset_rows.append(row)

        dataset_rows.sort(key=lambda row: float(row["sweep_value"]))
        x_values = [float(row["sweep_value"]) for row in dataset_rows]
        y_values = [
            max(float(row["time_seconds"]), 1e-12)
            for row in dataset_rows
        ]

        axis.plot(x_values, y_values, marker="o", linewidth=2.2)
        axis.set_yscale("log")
        axis.set_xlabel(dataset_rows[0]["sweep_parameter"])
        axis.set_ylabel("LDLT elapsed seconds")
        axis.set_title(dataset_name)
        axis.grid(True, which="both", alpha=0.3)

    fig.suptitle("LDLT Runtime on Synthetic Sweeps")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(figures_dir / "analysis_synthetic_ldlt_times.png", dpi=160)
    plt.close(fig)


def plot_builtin_fixed_comparisons(rows, figures_dir):
    """Plot fixed-case comparisons between scratch and built-in methods."""

    for analysis_group in ["builtin_fixed_synthetic", "builtin_fixed_real"]:
        selected = []
        for row in rows:
            if row["analysis_group"] == analysis_group:
                selected.append(row)

        if len(selected) == 0:
            continue

        keys = unique_scenario_keys(selected)
        for key in keys:
            scenario_rows = rows_for_scenario_key(selected, key)
            plot_builtin_performance_for_case(
                scenario_rows,
                figures_dir,
                analysis_group,
            )
            plot_builtin_times_for_case(
                scenario_rows,
                figures_dir,
                analysis_group,
            )


def plot_builtin_performance_for_case(rows, figures_dir, analysis_group):
    methods = builtin_comparison_methods()
    metric_list = detailed_metric_list()
    labels = short_method_labels(methods)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    positions = np.arange(len(methods))

    for metric_index in range(len(metric_list)):
        field, title, ylabel = metric_list[metric_index]
        values = []

        for method in methods:
            row = find_analysis_row_by_method(rows, method)
            if row is None:
                values.append(np.nan)
            else:
                values.append(max(float(row[field]), metric_floor(field)))

        axis = axes[metric_index]
        axis.bar(positions, values, color=[time_plot_color(m) for m in methods])
        axis.set_yscale("log")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, axis="y", which="both", alpha=0.3)

    title = rows[0]["dataset_name"] + " - " + rows[0]["scenario_type"]
    fig.suptitle(title + " - Built-in Performance Comparison")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    filename = "analysis_" + analysis_group + "_"
    filename = filename + make_safe_filename(title) + "_performance.png"
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def plot_builtin_times_for_case(rows, figures_dir, analysis_group):
    methods = builtin_comparison_methods()
    labels = short_method_labels(methods)
    positions = np.arange(len(methods))
    values = []
    converged = []

    for method in methods:
        row = find_analysis_row_by_method(rows, method)
        if row is None:
            values.append(np.nan)
            converged.append(True)
        else:
            values.append(max(float(row["time_seconds"]), 1e-12))
            converged.append(as_boolean(row["converged"]))

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar(
        positions,
        values,
        color=[time_plot_color(m) for m in methods],
        edgecolor="black",
        linewidth=0.3,
    )

    for index in range(len(bars)):
        if not converged[index]:
            bars[index].set_hatch("//")
            bars[index].set_linewidth(0.8)

    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("Elapsed seconds (log scale)")
    title = rows[0]["dataset_name"] + " - " + rows[0]["scenario_type"]
    axis.set_title(title + " - Built-in Runtime Comparison")
    axis.grid(True, axis="y", which="both", alpha=0.3)

    fig.tight_layout()
    filename = "analysis_" + analysis_group + "_"
    filename = filename + make_safe_filename(title) + "_times.png"
    fig.savefig(figures_dir / filename, dpi=160)
    plt.close(fig)


def plot_real_epsilon_sweeps(rows, figures_dir):
    """Plot Wine and Digits iterative metrics while epsilon changes."""

    selected = []
    for row in rows:
        if row["analysis_group"] == "real_epsilon":
            if row["method"] in ["Heavy Ball", "Nesterov"]:
                selected.append(row)

    if len(selected) == 0:
        return

    dataset_names = unique_values(selected, "dataset_name")
    metric_list = detailed_metric_list()

    for dataset_name in dataset_names:
        dataset_rows = []
        for row in selected:
            if row["dataset_name"] == dataset_name:
                dataset_rows.append(row)

        epsilons = unique_numeric_values(dataset_rows, "epsilon")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for metric_index in range(len(metric_list)):
            field, title, ylabel = metric_list[metric_index]
            axis = axes[metric_index]

            for method in ["Heavy Ball", "Nesterov"]:
                y_values = []
                for epsilon in epsilons:
                    row = find_analysis_row_with_epsilon(
                        dataset_rows,
                        method,
                        epsilon,
                        epsilon,
                    )
                    if row is None:
                        y_values.append(np.nan)
                    else:
                        y_values.append(
                            max(float(row[field]), metric_floor(field))
                        )

                style = plot_style_for_method(method)
                axis.plot(
                    epsilons,
                    y_values,
                    label=method,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2.3,
                )

            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.set_yscale("log")
            axis.set_xlabel("epsilon")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        fig.suptitle(dataset_name + " - epsilon sweep")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        filename = "analysis_" + make_safe_filename(dataset_name)
        filename = filename + "_epsilon_sweep.png"
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)

    plot_real_ldlt_times(rows, figures_dir)


def plot_real_ldlt_times(rows, figures_dir):
    selected = []
    seen = []

    for row in rows:
        if row["analysis_group"] != "real_epsilon":
            continue
        if row["method"] != "LDLT":
            continue

        dataset_name = row["dataset_name"]
        if dataset_name in seen:
            continue

        seen.append(dataset_name)
        selected.append(row)

    if len(selected) == 0:
        return

    labels = [row["dataset_name"] for row in selected]
    values = [max(float(row["time_seconds"]), 1e-12) for row in selected]
    positions = np.arange(len(labels))

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.bar(positions, values, color="#4c78a8")
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("LDLT elapsed seconds")
    axis.set_title("LDLT Runtime on Real Datasets")
    axis.grid(True, axis="y", which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "analysis_real_ldlt_times.png", dpi=160)
    plt.close(fig)


def plot_beta_fixed_variable_analysis(history_rows, figures_dir):
    """Plot fixed-beta Nesterov against variable-beta Nesterov."""

    selected = []
    for row in history_rows:
        if row["analysis_group"] == "beta_fixed_vs_variable":
            selected.append(row)

    if len(selected) == 0:
        return

    keys = unique_scenario_keys(selected)
    metrics = [
        ("grad_norm", "Gradient Norm", "||grad f(W)||_F"),
        ("objective_gap_to_reference", "Objective Gap", "f(W) - f(W_ref)"),
        (
            "relative_error_to_reference",
            "Relative Error",
            "||W - W_ref|| / ||W_ref||",
        ),
    ]

    for key in keys:
        scenario_rows = rows_for_scenario_key(selected, key)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for metric_index in range(len(metrics)):
            field, title, ylabel = metrics[metric_index]
            axis = axes[metric_index]

            for method in ["Nesterov variable beta", "Nesterov"]:
                method_rows = []
                for row in scenario_rows:
                    if row["method"] == method:
                        method_rows.append(row)
                method_rows.sort(key=lambda row: int(row["iteration"]))

                if len(method_rows) == 0:
                    continue

                iterations = [int(row["iteration"]) for row in method_rows]
                values = [
                    max(float(row[field]), metric_floor(field))
                    for row in method_rows
                ]
                style = plot_style_for_method(method)

                axis.semilogy(
                    iterations,
                    values,
                    label=method,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markevery=style["markevery"],
                    linewidth=style["linewidth"],
                )

            axis.set_xlabel("Iteration")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        title = key[0] + " - " + key[1]
        fig.suptitle(title + " - fixed beta vs variable beta")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        filename = "analysis_beta_fixed_variable_"
        filename = filename + make_safe_filename(title) + ".png"
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def detailed_metric_list():
    return [
        (
            "final_gradient_norm",
            "Gradient Norm",
            "||grad f(W)||_F",
        ),
        (
            "objective_gap_to_reference",
            "Objective Gap",
            "f(W) - f(W_ref)",
        ),
        (
            "relative_error_to_reference",
            "Relative Error",
            "||W - W_ref|| / ||W_ref||",
        ),
    ]


def metric_floor(field):
    return 1e-16


def marker_for_time_method(method):
    markers = {
        "LDLT": "o",
        "Heavy Ball": "s",
        "Nesterov": "^",
        "Nesterov variable beta": "D",
        "PyTorch SGD momentum": "x",
        "PyTorch SGD Nesterov": "P",
        "NumPy solve": "*",
        "SciPy LDLT": "v",
    }
    if method in markers:
        return markers[method]
    return "o"


def unique_values(rows, field):
    values = []
    for row in rows:
        value = row[field]
        if value not in values:
            values.append(value)
    return values


def unique_numeric_values(rows, field):
    values = []
    for row in rows:
        value = float(row[field])
        if value not in values:
            values.append(value)
    values.sort()
    return values


def unique_scenario_keys(rows):
    keys = []
    for row in rows:
        key = (row["dataset_name"], row["scenario_type"])
        if key not in keys:
            keys.append(key)
    return keys


def rows_for_scenario_key(rows, key):
    selected = []
    for row in rows:
        same_dataset = row["dataset_name"] == key[0]
        same_scenario = row["scenario_type"] == key[1]
        if same_dataset and same_scenario:
            selected.append(row)
    return selected


def find_analysis_row(rows, method, field, value):
    for row in rows:
        if row["method"] != method:
            continue
        if float(row[field]) == float(value):
            return row
    return None


def find_analysis_row_with_epsilon(rows, method, epsilon, sweep_value):
    for row in rows:
        if row["method"] != method:
            continue
        if abs(float(row["epsilon"]) - float(epsilon)) > 1e-15:
            continue
        if abs(float(row["sweep_value"]) - float(sweep_value)) > 1e-15:
            continue
        return row
    return None


def find_analysis_row_by_method(rows, method):
    for row in rows:
        if row["method"] == method:
            return row
    return None


def format_epsilon_label(value):
    return format(float(value), ".0e")


def line_style_for_epsilon(epsilon, epsilons):
    index = epsilons.index(epsilon)
    styles = ["-", "--", ":", "-."]
    return styles[index % len(styles)]


def builtin_comparison_methods():
    return [
        "LDLT",
        "Heavy Ball",
        "Nesterov",
        "PyTorch SGD momentum",
        "PyTorch SGD Nesterov",
        "NumPy solve",
        "SciPy LDLT",
    ]


def short_method_labels(methods):
    labels = []
    for method in methods:
        if method == "PyTorch SGD momentum":
            labels.append("PyTorch HB")
        elif method == "PyTorch SGD Nesterov":
            labels.append("PyTorch NAG")
        elif method == "SciPy LDLT":
            labels.append("SciPy LDLT")
        else:
            labels.append(method)
    return labels


def make_safe_filename(text):
    safe = text.replace(" ", "_")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace(":", "_")
    return safe


def print_short_scenario_report(instance, conditioning_row, summary_rows):
    print(
        "  Q dimension: "
        + str(instance.q.shape[0])
        + ", lambda: "
        + format(instance.lambda_reg, ".1e")
        + ", estimated kappa: "
        + format(float(conditioning_row["estimated_condition_number"]), ".3e")
    )

    for row in summary_rows:
        method = row["method"]
        grad_norm = format(float(row["final_gradient_norm"]), ".3e")
        rel_err = format(float(row["relative_error_to_reference"]), ".3e")
        test_acc = format(float(row["test_accuracy"]), ".3f")
        print(
            "    "
            + method
            + ": grad="
            + grad_norm
            + ", rel_err_ref="
            + rel_err
            + ", test_acc="
            + test_acc
        )


def validate_results(summary_rows, history_rows):
    """Print simple checks after the experiments."""

    ldlt_errors = []
    iterative_warnings = []

    for row in summary_rows:
        if row["method"] == "LDLT":
            ldlt_errors.append(float(row["relative_error_to_reference"]))

        is_iterative = is_iterative_method(row["method"])
        if is_iterative:
            initial_grad = float(row["initial_gradient_norm"])
            final_grad = float(row["final_gradient_norm"])
            if final_grad > initial_grad:
                iterative_warnings.append(
                    row["dataset_name"]
                    + "/"
                    + row["scenario_type"]
                    + "/"
                    + row["method"]
                )

    max_ldlt_error = 0.0
    if len(ldlt_errors) > 0:
        max_ldlt_error = max(ldlt_errors)

    print("")
    print("Validation checks")
    print("  max relative difference LDLT vs NumPy solve: " + format(max_ldlt_error, ".3e"))

    if len(iterative_warnings) == 0:
        print("  all iterative methods reduced the gradient norm in all scenarios.")
    else:
        print("  Warning: gradient norm did not decrease in:")
        for item in iterative_warnings:
            print("    " + item)

    # This count is a useful practical indicator, especially on the ill
    # conditioned instance where convergence can take many iterations.
    not_converged = []
    for row in summary_rows:
        if is_iterative_method(row["method"]):
            if not bool(row["converged"]):
                not_converged.append(
                    row["dataset_name"]
                    + "/"
                    + row["scenario_type"]
                    + "/"
                    + row["method"]
                )

    if len(not_converged) == 0:
        print("  all iterative methods reached the requested tolerance.")
    else:
        print("  iterative methods that stopped at max_iter:")
        for item in not_converged:
            print("    " + item)


def is_iterative_method(method):
    return method in [
        "Heavy Ball",
        "Nesterov",
        "Nesterov variable beta",
        "PyTorch SGD momentum",
        "PyTorch SGD Nesterov",
    ]


if __name__ == "__main__":
    main()
