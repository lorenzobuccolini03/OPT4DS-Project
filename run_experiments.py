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
    ldlt_solve_weights,
    nesterov_accelerated_gradient,
)
from elm_optimization.elm import (
    apply_sparse_feature_mask,
    create_elm_instance_from_arrays,
    generate_correlated_classification_data,
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
    "total_hidden_weights",
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

REQUESTED_DIMENSION_TIME_FIELDS = [
    "algorithm",
    "dimensionality",
    "computational_time",
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
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
            "optimizer references."
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


def add_real_wine_scenario(scenarios, suite, seed):
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

    if suite == "full":
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


def add_real_digits_scenario(scenarios, suite, seed):
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

    if suite == "full":
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

    numpy_cholesky = numpy_cholesky_reference(q, c)

    results = [
        ldlt_result,
        hb_result,
        nag_result,
        nag_variable_result,
        pytorch_hb_result,
        pytorch_nesterov_result,
        numpy_reference,
        numpy_cholesky,
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
    samples and about 10000 hidden-layer weights. The quick suite uses a
    smaller version so that the code can be checked quickly during development.
    """

    if suite == "full":
        return {
            "n_train": 1000,
            "n_test": 350,
            "n_features": 20,
            "total_hidden_weights": 10000,
        }

    return {
        "n_train": 260,
        "n_test": 100,
        "n_features": 10,
        "total_hidden_weights": 600,
    }


def dimension_scaling_configs(suite):
    """Return dimensions for the scaling test.

    The full suite keeps the input size at 1000 training samples and uses a
    much more visible hidden-weight sweep: 10000, 50000, and 100000. With
    100 input features, this corresponds to 100, 500, and 1000 hidden neurons.
    This keeps the test meaningful without making the scratch LDLT factorization
    completely impractical.
    """

    configs = []

    if suite == "full":
        levels = [
            (1000, 350, 100, 10000),
            (1000, 350, 100, 50000),
            (1000, 350, 100, 100000),
        ]
    else:
        levels = [
            (180, 70, 10, 300),
            (240, 90, 10, 600),
            (300, 110, 10, 900),
        ]

    for n_train, n_test, n_features, total_weights in levels:
        configs.append(
            {
                "n_train": n_train,
                "n_test": n_test,
                "n_features": n_features,
                "total_hidden_weights": total_weights,
            }
        )

    return configs


def hidden_width_from_total_weights(config):
    width = int(round(config["total_hidden_weights"] / config["n_features"]))
    return max(2, width)


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
    hidden_width = hidden_width_from_total_weights(dimension_config)

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


def create_dimension_scaling_scenario(config, seed):
    scenario = create_synthetic_analysis_scenario(
        "well",
        0.3,
        config,
        seed,
    )
    total_weights = config["n_features"] * scenario["instance"].hidden_width
    scenario["dataset_name"] = "dimension_scaling"
    scenario["scenario_type"] = "weights_" + str(total_weights)
    scenario["description"] = (
        "Well-conditioned scaling case with "
        + str(config["n_train"])
        + " training samples and "
        + str(total_weights)
        + " hidden-layer weights."
    )
    scenario["sweep_parameter"] = "total_hidden_weights"
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
    numpy_cholesky = numpy_cholesky_reference(q, c)

    context = {
        "scenario": scenario,
        "instance": instance,
        "spectral": spectral,
        "objective_fn": objective_fn,
        "numpy_reference": numpy_reference,
        "numpy_cholesky": numpy_cholesky,
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
            "total_hidden_weights",
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
            "total_hidden_weights",
            sweep_value,
        )
        append_analysis_result(
            rows,
            histories,
            "dimension_scaling",
            context,
            nag_result,
            epsilon,
            "total_hidden_weights",
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
    config = fixed_dimension_config(suite)

    scenarios = [
        create_synthetic_analysis_scenario("well", 0.1, config, seed),
        create_synthetic_analysis_scenario("ill", 0.9, config, seed + 1),
        create_synthetic_analysis_scenario("sparse", 0.7, config, seed + 2),
    ]
    add_real_wine_scenario(scenarios, suite, seed + 3)
    add_real_digits_scenario(scenarios, suite, seed + 4)

    for index in range(len(scenarios)):
        scenario = scenarios[index]
        if "sweep_parameter" not in scenario:
            scenario["sweep_parameter"] = "dataset"
            scenario["sweep_value"] = scenario["dataset_name"]

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
            scenario["sweep_parameter"],
            scenario["sweep_value"],
        )
        append_analysis_result(
            rows,
            histories,
            "beta_fixed_vs_variable",
            context,
            variable_result,
            epsilon,
            scenario["sweep_parameter"],
            scenario["sweep_value"],
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
        context["numpy_cholesky"],
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

    total_hidden_weights = instance.n_features * instance.hidden_width

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
        "total_hidden_weights": total_hidden_weights,
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
    if method in ["NumPy solve", "NumPy Cholesky"]:
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

    run_requested_beta_comparisons(
        fixed_config,
        suite,
        seed + 900,
        epsilon,
        max_iter,
        record_every,
        figures_dir,
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
            "total_hidden_weights": 10000,
        }

    return {
        "n_train": 260,
        "n_test": 100,
        "n_features": 10,
        "total_hidden_weights": 600,
    }


def requested_dimension_configs(suite):
    if suite == "full":
        total_weights = [10000, 50000, 100000]
        n_train = 1000
        n_test = 350
        n_features = 100
    else:
        total_weights = [600, 900, 1200]
        n_train = 260
        n_test = 100
        n_features = 10

    configs = []
    for value in total_weights:
        configs.append(
            {
                "n_train": n_train,
                "n_test": n_test,
                "n_features": n_features,
                "total_hidden_weights": value,
            }
        )
    return configs


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
                    "dimensionality": config["total_hidden_weights"],
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
                    "dimensionality": fixed_config["total_hidden_weights"],
                    "computational_time": result.elapsed_seconds,
                }
            )

        history_rows.extend(
            make_requested_history_rows(
                context,
                hb_result,
                "rho",
                rho,
                fixed_config["total_hidden_weights"],
                epsilon,
            )
        )
        history_rows.extend(
            make_requested_history_rows(
                context,
                nag_result,
                "rho",
                rho,
                fixed_config["total_hidden_weights"],
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
                    "dimensionality": fixed_config["total_hidden_weights"],
                    "computational_time": result.elapsed_seconds,
                }
            )

        history_rows.extend(
            make_requested_history_rows(
                context,
                hb_result,
                "sparseness_percentage",
                sparsity,
                fixed_config["total_hidden_weights"],
                epsilon,
            )
        )
        history_rows.extend(
            make_requested_history_rows(
                context,
                nag_result,
                "sparseness_percentage",
                sparsity,
                fixed_config["total_hidden_weights"],
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
                    "dimensionality": config["total_hidden_weights"],
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
            fixed_config["total_hidden_weights"],
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
            fixed_config["total_hidden_weights"],
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
            fixed_config["total_hidden_weights"],
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
            fixed_config["total_hidden_weights"],
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
        results = [
            context["ldlt_result"],
            context["numpy_reference"],
            context["numpy_cholesky"],
        ]

        for result in results:
            rows.append(
                {
                    "algorithm": result.method,
                    "dimensionality": config["total_hidden_weights"],
                    "computational_time": result.elapsed_seconds,
                    "relative_error_to_our_ldlt": relative_error(
                        result.weights,
                        context["ldlt_weights"],
                    ),
                }
            )

    return rows


def run_requested_beta_comparisons(
    fixed_config,
    suite,
    seed,
    epsilon,
    max_iter,
    record_every,
    figures_dir,
):
    cases = [
        (
            create_synthetic_analysis_scenario(
                "well",
                0.1,
                fixed_config,
                seed,
            ),
            "requested_beta_well_conditioned.png",
        ),
        (
            create_synthetic_analysis_scenario(
                "ill",
                0.9,
                fixed_config,
                seed + 1,
            ),
            "requested_beta_ill_conditioned.png",
        ),
        (
            create_synthetic_analysis_scenario(
                "sparse",
                0.7,
                fixed_config,
                seed + 2,
            ),
            "requested_beta_sparse_70_percent.png",
        ),
    ]

    real_scenarios = []
    add_real_wine_scenario(real_scenarios, suite, seed + 3)
    add_real_digits_scenario(real_scenarios, suite, seed + 4)
    cases.append((real_scenarios[0], "requested_beta_wine.png"))
    cases.append((real_scenarios[1], "requested_beta_digits.png"))

    for index in range(len(cases)):
        scenario, filename = cases[index]
        context = prepare_analysis_context(scenario, seed + index)
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
        plot_requested_two_results(
            context,
            [fixed_result, variable_result],
            epsilon,
            figures_dir / filename,
            "Nesterov Fixed Beta vs Variable Beta",
        )


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
    axis.set_xlabel("Hidden-layer weights")
    axis.set_ylabel("Computational time (seconds)")
    axis.set_title("Requested Runtime Scaling at Fixed Epsilon")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "requested_dimension_scaling_times.png", dpi=160)
    plt.close(fig)


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
        + ", hidden weights="
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
    dimensionality = context["instance"].n_features * context["instance"].hidden_width

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
        + ", hidden weights="
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


def numpy_cholesky_reference(q, c):
    """Built-in Cholesky benchmark.

    This is another validation method only. The scratch LDLT implementation in
    ``algorithms.py`` does not call Cholesky.
    """

    start = perf_counter()
    lower = np.linalg.cholesky(q)
    z = np.linalg.solve(lower, c.T)
    weights = np.linalg.solve(lower.T, z).T
    elapsed = perf_counter() - start

    grad = weights @ q - c
    grad_norm = float(np.sqrt(np.sum(grad * grad)))

    return OptimizationResult(
        method="NumPy Cholesky",
        weights=weights,
        iterations=0,
        converged=True,
        elapsed_seconds=elapsed,
        final_gradient_norm=grad_norm,
    )


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
    if result.method in ["NumPy solve", "NumPy Cholesky"]:
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
        "NumPy Cholesky",
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
        "NumPy Cholesky": "#8c564b",
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
    axis.set_xlabel("Hidden-layer weights")
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
        "NumPy Cholesky": "v",
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
        "NumPy Cholesky",
    ]


def short_method_labels(methods):
    labels = []
    for method in methods:
        if method == "PyTorch SGD momentum":
            labels.append("PyTorch HB")
        elif method == "PyTorch SGD Nesterov":
            labels.append("PyTorch NAG")
        elif method == "NumPy Cholesky":
            labels.append("NumPy Chol")
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
