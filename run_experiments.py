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

    metadata = {
        "suite": args.suite,
        "seed": args.seed,
        "tol": args.tol,
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
