"""Run the ELM optimization experiments.

The script produces reproducible CSV files and plots under ``results/`` by
default.  Each row compares algorithms on the same fixed mathematical problem,
which is essential for the course requirement that algorithmic performance
must not be mixed with model hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

# Keep matplotlib from writing its cache into the user home directory.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.getenv("TMPDIR", "/tmp")) / "elm_optimization_matplotlib_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elm_optimization.algorithms import (
    OptimizationResult,
    estimate_spectral_bounds,
    heavy_ball,
    ldlt_solve_weights,
    nesterov_accelerated_gradient,
)
from elm_optimization.elm import ELMInstance, create_elm_classification_instance
from elm_optimization.metrics import (
    classification_accuracy,
    gradient,
    mean_squared_error,
    objective_value,
    relative_error,
)


SUMMARY_FIELDS = [
    "scenario",
    "instance_id",
    "method",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "hidden_width",
    "n_variables_per_output",
    "lambda_reg",
    "activation",
    "mu_bound",
    "l_smooth_estimate",
    "condition_estimate",
    "power_iterations",
    "raw_largest_eigenvalue_estimate",
    "iterations",
    "converged",
    "elapsed_seconds",
    "final_gradient_norm",
    "objective",
    "objective_gap_to_ldlt",
    "relative_weight_error_to_ldlt",
    "train_mse",
    "test_mse",
    "train_accuracy",
    "test_accuracy",
    "alpha",
    "beta",
]

HISTORY_FIELDS = [
    "scenario",
    "instance_id",
    "method",
    "iteration",
    "grad_norm",
    "objective",
    "objective_gap_to_ldlt",
    "relative_weight_error_to_ldlt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="quick runs in seconds; full uses larger instances for the report.",
    )
    parser.add_argument("--output-dir", default="results", help="Directory for CSVs/plots.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute gradient tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override max iterations.")
    parser.add_argument(
        "--record-every",
        type=int,
        default=5,
        help="Record one convergence-history point every k iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = suite_config(args.suite, args.max_iter)
    summary_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    for scenario, instance_id, instance in build_instances(args.suite, args.seed):
        print(f"Running {scenario}/{instance_id} ...", flush=True)
        suite_summary, suite_history = run_algorithm_suite(
            scenario=scenario,
            instance_id=instance_id,
            instance=instance,
            tol=args.tol,
            max_iter=config["max_iter"],
            record_every=args.record_every,
            seed=args.seed,
        )
        summary_rows.extend(suite_summary)
        history_rows.extend(suite_history)

    summary_path = output_dir / "summary.csv"
    history_path = output_dir / "convergence_history.csv"
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(history_path, HISTORY_FIELDS, history_rows)

    plot_convergence(history_rows, figures_dir)
    plot_conditioning(summary_rows, figures_dir)
    plot_scaling(summary_rows, figures_dir)

    metadata = {
        "suite": args.suite,
        "seed": args.seed,
        "tol": args.tol,
        "max_iter": config["max_iter"],
        "record_every": args.record_every,
        "notes": (
            "LDLT, Heavy Ball, and Nesterov are scratch implementations. "
            "NumPy solve is used only as an off-the-shelf correctness check."
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nWrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote plots to {figures_dir}")


def suite_config(suite: str, max_iter_override: int | None) -> dict[str, int]:
    if max_iter_override is not None:
        return {"max_iter": max_iter_override}
    if suite == "full":
        return {"max_iter": 20_000}
    return {"max_iter": 8_000}


def build_instances(
    suite: str, seed: int
) -> Iterable[tuple[str, str, ELMInstance]]:
    """Yield reproducible ELM instances for validation and experiments."""

    if suite == "full":
        validation = dict(n_train=1200, n_test=400, n_features=30, hidden_width=160)
        conditioning_lambdas = [1e-1, 1e-2, 1e-3, 1e-4]
        scaling_widths = [60, 120, 240, 360]
        scaling = dict(n_train=1800, n_test=600, n_features=35)
    else:
        validation = dict(n_train=600, n_test=200, n_features=20, hidden_width=80)
        conditioning_lambdas = [1e-1, 1e-2, 1e-3]
        scaling_widths = [40, 80, 120]
        scaling = dict(n_train=800, n_test=250, n_features=25)

    yield (
        "validation",
        "validation_main",
        create_elm_classification_instance(
            **validation,
            n_classes=3,
            lambda_reg=1e-3,
            activation="tanh",
            seed=seed,
        ),
    )

    for lambda_reg in conditioning_lambdas:
        yield (
            "conditioning",
            f"lambda_{lambda_reg:.0e}",
            create_elm_classification_instance(
                n_train=validation["n_train"],
                n_test=validation["n_test"],
                n_features=validation["n_features"],
                n_classes=3,
                hidden_width=validation["hidden_width"],
                lambda_reg=lambda_reg,
                activation="tanh",
                seed=seed + 1,
            ),
        )

    for width in scaling_widths:
        yield (
            "scaling",
            f"hidden_{width}",
            create_elm_classification_instance(
                **scaling,
                n_classes=3,
                hidden_width=width,
                lambda_reg=1e-3,
                activation="tanh",
                seed=seed + 2,
            ),
        )


def run_algorithm_suite(
    *,
    scenario: str,
    instance_id: str,
    instance: ELMInstance,
    tol: float,
    max_iter: int,
    record_every: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    q = instance.q
    c = instance.c

    spectral = estimate_spectral_bounds(
        q,
        instance.lambda_reg,
        seed=seed,
        l_safety_factor=1.01,
    )

    objective_fn = lambda w: objective_value(
        w, instance.h_train_aug, instance.y_train, instance.lambda_reg
    )

    ldlt_result = ldlt_solve_weights(q, c)
    w_star = ldlt_result.weights
    f_star = objective_fn(w_star)

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
        reference_weights=w_star,
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
        reference_weights=w_star,
        record_every=record_every,
    )

    # Off-the-shelf solve is not one of the project algorithms. It is included
    # only as an external correctness check, as recommended by the guidelines.
    numpy_reference = numpy_solve_reference(q, c)

    results = [ldlt_result, hb_result, nag_result, numpy_reference]
    summary = [
        summarize_result(
            scenario=scenario,
            instance_id=instance_id,
            instance=instance,
            result=result,
            spectral=spectral,
            w_star=w_star,
            f_star=f_star,
            objective_fn=objective_fn,
        )
        for result in results
    ]

    history = []
    for result in [hb_result, nag_result]:
        history.extend(
            history_rows_for_result(
                scenario=scenario,
                instance_id=instance_id,
                result=result,
                f_star=f_star,
            )
        )

    return summary, history


def numpy_solve_reference(q: np.ndarray, c: np.ndarray) -> OptimizationResult:
    """Optional library baseline used only for validation."""

    from time import perf_counter

    start = perf_counter()
    weights = np.linalg.solve(q, c.T).T
    elapsed = perf_counter() - start
    grad_norm = float(np.sqrt(np.sum((weights @ q - c) ** 2)))
    return OptimizationResult(
        method="NumPy solve reference",
        weights=weights,
        iterations=0,
        converged=True,
        elapsed_seconds=elapsed,
        final_gradient_norm=grad_norm,
    )


def summarize_result(
    *,
    scenario: str,
    instance_id: str,
    instance: ELMInstance,
    result: OptimizationResult,
    spectral,
    w_star: np.ndarray,
    f_star: float,
    objective_fn,
) -> dict[str, object]:
    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug
    objective = objective_fn(result.weights)
    grad_norm = float(np.sqrt(np.sum(gradient(result.weights, instance.q, instance.c) ** 2)))

    return {
        "scenario": scenario,
        "instance_id": instance_id,
        "method": result.method,
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "hidden_width": instance.hidden_width,
        "n_variables_per_output": instance.n_variables_per_output,
        "lambda_reg": instance.lambda_reg,
        "activation": instance.activation,
        "mu_bound": spectral.mu,
        "l_smooth_estimate": spectral.l_smooth,
        "condition_estimate": spectral.condition_estimate,
        "power_iterations": spectral.power_iterations,
        "raw_largest_eigenvalue_estimate": spectral.raw_largest_eigenvalue_estimate,
        "iterations": result.iterations,
        "converged": result.converged,
        "elapsed_seconds": result.elapsed_seconds,
        "final_gradient_norm": grad_norm,
        "objective": objective,
        "objective_gap_to_ldlt": max(0.0, objective - f_star),
        "relative_weight_error_to_ldlt": relative_error(result.weights, w_star),
        "train_mse": mean_squared_error(train_scores, instance.y_train),
        "test_mse": mean_squared_error(test_scores, instance.y_test),
        "train_accuracy": classification_accuracy(train_scores, instance.train_labels),
        "test_accuracy": classification_accuracy(test_scores, instance.test_labels),
        "alpha": "" if result.alpha is None else result.alpha,
        "beta": "" if result.beta is None else result.beta,
    }


def history_rows_for_result(
    *,
    scenario: str,
    instance_id: str,
    result: OptimizationResult,
    f_star: float,
) -> list[dict[str, object]]:
    rows = []
    for iteration, grad_norm, objective, rel_error in zip(
        result.history["iteration"],
        result.history["grad_norm"],
        result.history["objective"],
        result.history["relative_error"],
        strict=True,
    ):
        rows.append(
            {
                "scenario": scenario,
                "instance_id": instance_id,
                "method": result.method,
                "iteration": int(iteration),
                "grad_norm": grad_norm,
                "objective": objective,
                "objective_gap_to_ldlt": max(0.0, objective - f_star),
                "relative_weight_error_to_ldlt": rel_error,
            }
        )
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(history_rows: list[dict[str, object]], figures_dir: Path) -> None:
    validation_rows = [
        row for row in history_rows if row["scenario"] == "validation"
    ]
    if not validation_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in sorted({row["method"] for row in validation_rows}):
        method_rows = [row for row in validation_rows if row["method"] == method]
        iterations = [row["iteration"] for row in method_rows]
        grad_norms = [max(float(row["grad_norm"]), 1e-300) for row in method_rows]
        gaps = [
            max(float(row["objective_gap_to_ldlt"]), 1e-300) for row in method_rows
        ]
        axes[0].semilogy(iterations, grad_norms, label=method)
        axes[1].semilogy(iterations, gaps, label=method)

    axes[0].set_title("Gradient Norm")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("||grad f(W)||_F")
    axes[1].set_title("Objective Gap to LDLT")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("f(W) - f(W*)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "convergence_validation.png", dpi=160)
    plt.close(fig)


def plot_conditioning(summary_rows: list[dict[str, object]], figures_dir: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if row["scenario"] == "conditioning" and row["method"] in {"Heavy Ball", "Nesterov"}
    ]
    if not rows:
        return

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for method in sorted({row["method"] for row in rows}):
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["lambda_reg"]),
        )
        lambdas = [float(row["lambda_reg"]) for row in method_rows]
        iterations = [int(row["iterations"]) for row in method_rows]
        axis.plot(lambdas, iterations, marker="o", label=method)
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_title("Effect of Regularization on Iterations")
    axis.set_xlabel("lambda")
    axis.set_ylabel("Iterations to tolerance")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "conditioning_iterations.png", dpi=160)
    plt.close(fig)


def plot_scaling(summary_rows: list[dict[str, object]], figures_dir: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if row["scenario"] == "scaling"
        and row["method"] in {"LDLT", "Heavy Ball", "Nesterov"}
    ]
    if not rows:
        return

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for method in sorted({row["method"] for row in rows}):
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: int(row["n_variables_per_output"]),
        )
        n_variables = [int(row["n_variables_per_output"]) for row in method_rows]
        times = [float(row["elapsed_seconds"]) for row in method_rows]
        axis.plot(n_variables, times, marker="o", label=method)
    axis.set_title("Runtime Scaling with Hidden Width")
    axis.set_xlabel("Variables per output row (hidden_width + 1)")
    axis.set_ylabel("Seconds")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "scaling_runtime.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
