"""Run the ELM optimization experiments.

The goal of this file is to test the algorithms from the project on several
ELM training problems:

1. a well-conditioned synthetic problem;
2. an ill-conditioned synthetic problem;
3. a partially sparse synthetic problem;
4. a real classification dataset from scikit-learn.

The project algorithms are still the hand-written LDLT factorization, Heavy
Ball, and Nesterov. Built-in routines such as ``np.linalg.solve`` are used only
in the benchmark part of this script, so that we can check the numerical
correctness of our own implementations.
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

try:
    from sklearn.datasets import load_digits, load_wine
    from sklearn.model_selection import train_test_split
except ImportError:
    load_digits = None
    load_wine = None
    train_test_split = None

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
    "method",
    "method_type",
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
    "method",
    "iteration",
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
    parser.add_argument("--record-every", type=int, default=10)
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

    benchmark_rows = []
    for row in summary_rows:
        if row["method_type"] == "built-in benchmark":
            benchmark_rows.append(row)

    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(conditioning_path, CONDITIONING_FIELDS, conditioning_rows)
    write_csv(history_path, HISTORY_FIELDS, history_rows)
    write_csv(benchmark_path, SUMMARY_FIELDS, benchmark_rows)

    plot_convergence_by_scenario(history_rows, figures_dir)
    plot_conditioning_overview(conditioning_rows, figures_dir)

    metadata = {
        "suite": args.suite,
        "seed": args.seed,
        "tol": args.tol,
        "max_iter": max_iter,
        "record_every": args.record_every,
        "notes": (
            "LDLT, Heavy Ball, and Nesterov are the hand-written project "
            "algorithms. Built-in routines are used only in the benchmark "
            "section to validate the numerical results."
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
    print("Wrote plots to " + str(figures_dir))


def choose_max_iter(suite, max_iter_override):
    if max_iter_override is not None:
        return max_iter_override
    if suite == "full":
        return 25000
    return 12000


def build_scenarios(suite, seed):
    """Create all ELM test cases used in the numerical experiments."""

    scenarios = []

    add_well_conditioned_scenario(scenarios, suite, seed)
    add_ill_conditioned_scenario(scenarios, suite, seed + 100)
    add_sparse_scenario(scenarios, suite, seed + 200)
    add_real_wine_scenario(scenarios, suite, seed + 300)

    if suite == "full":
        add_real_digits_scenario(scenarios, seed + 400)

    return scenarios


def add_well_conditioned_scenario(scenarios, suite, seed):
    """Easy case: low correlation and enough regularization.

    This case checks that all algorithms behave correctly when Q is not close
    to singular. It is the baseline test before looking at harder cases.
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
    data = generate_correlated_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_classes=3,
        class_sep=2.5,
        noise=0.7,
        correlation_strength=0.05,
        feature_scales=scales,
        seed=seed,
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
        seed=seed,
        standardize_data=False,
    )

    scenario = {
        "dataset_name": "synthetic_well_conditioned",
        "scenario_type": "well_conditioned",
        "description": (
            "Low feature correlation and moderate regularization: this is the "
            "easy ELM problem."
        ),
        "instance": instance,
    }
    scenarios.append(scenario)


def add_ill_conditioned_scenario(scenarios, suite, seed):
    """Difficult case: correlated features, wide hidden layer, small lambda.

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
    data = generate_correlated_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_classes=3,
        class_sep=1.4,
        noise=0.9,
        correlation_strength=0.95,
        feature_scales=scales,
        seed=seed,
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
        seed=seed,
        standardize_data=False,
    )

    scenario = {
        "dataset_name": "synthetic_ill_conditioned",
        "scenario_type": "ill_conditioned",
        "description": (
            "Strong correlation, small regularization, and a wider hidden "
            "layer: this stresses first-order methods."
        ),
        "instance": instance,
    }
    scenarios.append(scenario)


def add_sparse_scenario(scenarios, suite, seed):
    """Partially sparse case: many input entries are set to zero.

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
    data = generate_correlated_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_classes=4,
        class_sep=1.8,
        noise=1.0,
        correlation_strength=0.25,
        feature_scales=scales,
        seed=seed,
    )
    x_train, train_labels, x_test, test_labels = data
    x_train, x_test = apply_sparse_feature_mask(
        x_train,
        x_test,
        zero_probability=0.70,
        seed=seed + 1,
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
        seed=seed,
        standardize_data=False,
    )

    scenario = {
        "dataset_name": "synthetic_sparse",
        "scenario_type": "partially_sparse",
        "description": (
            "About 70 percent of input entries are zeroed before the hidden "
            "layer is built."
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
        "description": (
            "Real multiclass classification data from scikit-learn."
        ),
        "instance": instance,
    }
    scenarios.append(scenario)


def add_real_digits_scenario(scenarios, seed):
    """Extra real dataset used only in the full experiment suite."""

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

    instance = create_elm_instance_from_arrays(
        x_train.T,
        train_labels,
        x_test.T,
        test_labels,
        hidden_width=120,
        lambda_reg=1e-2,
        activation="tanh",
        hidden_scale=0.8,
        seed=seed,
        standardize_data=True,
    )

    scenario = {
        "dataset_name": "digits",
        "scenario_type": "real_dataset_full_suite",
        "description": (
            "Larger real handwritten-digit dataset, included only in the "
            "full suite."
        ),
        "instance": instance,
    }
    scenarios.append(scenario)


def check_sklearn_available():
    if load_wine is None or train_test_split is None:
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

    numpy_cholesky = numpy_cholesky_reference(q, c)

    results = [
        ldlt_result,
        hb_result,
        nag_result,
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
    for result in [hb_result, nag_result]:
        rows = history_rows_for_result(
            scenario,
            result,
            reference_objective,
        )
        history.extend(rows)

    return summary, conditioning_row, history


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

    row = {
        "dataset_name": scenario["dataset_name"],
        "scenario_type": scenario["scenario_type"],
        "method": result.method,
        "method_type": method_type,
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
    """Save one convergence plot for each scenario."""

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

        methods = sorted(set(row["method"] for row in selected_rows))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

        for method in methods:
            method_rows = []
            for row in selected_rows:
                if row["method"] == method:
                    method_rows.append(row)

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

            axes[0].semilogy(iterations, grad_norms, label=method)
            axes[1].semilogy(iterations, objective_gaps, label=method)
            axes[2].semilogy(iterations, relative_errors, label=method)

        axes[0].set_title("Gradient Norm")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("||grad f(W)||_F")

        axes[1].set_title("Objective Gap")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("f(W) - f(W_ref)")

        axes[2].set_title("Relative Error")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("||W - W_ref|| / ||W_ref||")

        title = dataset_name + " - " + scenario_type
        fig.suptitle(title)

        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        fig.tight_layout()
        filename = make_safe_filename(dataset_name + "_" + scenario_type)
        fig.savefig(figures_dir / (filename + "_convergence.png"), dpi=160)
        plt.close(fig)


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

    fig, axis = plt.subplots(figsize=(8.0, 4.8))
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

        is_iterative = row["method"] in ["Heavy Ball", "Nesterov"]
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
        print("  Heavy Ball and Nesterov reduced the gradient norm in all scenarios.")
    else:
        print("  Warning: gradient norm did not decrease in:")
        for item in iterative_warnings:
            print("    " + item)

    # This count is a useful practical indicator, especially on the ill
    # conditioned instance where convergence can take many iterations.
    not_converged = []
    for row in summary_rows:
        if row["method"] in ["Heavy Ball", "Nesterov"]:
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


if __name__ == "__main__":
    main()
