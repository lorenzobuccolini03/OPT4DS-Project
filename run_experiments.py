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
from pathlib import Path
from time import perf_counter
import numpy as np
try:
    from scipy.linalg import ldl as scipy_ldl
    from scipy.linalg import solve_triangular as scipy_solve_triangular
except ImportError:
    scipy_ldl = None
    scipy_solve_triangular = None
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
    objective_value,
    relative_error,
)
BETA_COMPARISON_FIELDS = [
    "dataset",
    "algorithm",
    "beta_rule",
    "epsilon",
    "hidden_width",
    "output_weight_count",
    "converged",
    "iterations",
    "computational_time_seconds",
    "final_gradient_norm",
    "objective_gap",
    "relative_error_to_reference",
    "train_accuracy",
    "test_accuracy",
    "lambda_reg",
    "estimated_L",
    "mu",
    "condition_number",
    "alpha",
    "beta",
]
BETA_CONVERGENCE_FIELDS = [
    "dataset",
    "algorithm",
    "beta_rule",
    "epsilon",
    "hidden_width",
    "iteration",
    "gradient_norm",
    "relative_objective_gap",
    "relative_error_to_reference",
]
SYNTHETIC_SWEEP_SUMMARY_FIELDS = [
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
SYNTHETIC_SWEEP_RUN_FIELDS = [
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
DIMENSION_SCALING_FIELDS = [
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
DIMENSION_CONVERGENCE_FIELDS = [
    "algorithm",
    "iteration",
    "gradient_norm",
    "relative_gap",
    "output_weight_count",
    "epsilon",
]
REAL_OPTIMIZER_FIELDS = [
    "algorithm",
    "dataset",
    "epsilon",
    "initial_gradient_norm",
    "final_gradient_norm",
    "absolute_gap",
    "relative_gap",
    "computational_time_seconds",
    "iterations",
    "hidden_layer_size",
    "alpha",
    "beta",
]
LDLT_COMPARISON_FIELDS = [
    "dataset",
    "algorithm",
    "hidden_layer_size",
    "output_weight_count",
    "q_dimension",
    "minimum_value",
    "computational_time_seconds",
    "residual_norm",
    "relative_residual_norm",
    "relative_difference_to_our_ldlt",
]
SPARSE_DIMENSION_FIELDS = [
    "algorithm",
    "sparsity",
    "hidden_nodes",
    "computational_time_seconds",
    "epsilon",
    "converged",
    "iterations",
    "final_gradient_norm",
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
]


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


def choose_max_iter(max_iter_override):
    if max_iter_override is not None:
        return max_iter_override
    return 3000


def choose_dimension_scaling_max_iter(max_iter_override):
    """Default iteration budget for the Digits dimension experiment."""

    if max_iter_override is not None:
        return max_iter_override
    return 5000


def choose_synthetic_sweep_max_iter(max_iter_override):
    """Default budget for the new sparsity/rho synthetic sweeps."""

    if max_iter_override is not None:
        return max_iter_override
    return 5000


def choose_beta_comparison_max_iter(max_iter_override):
    """Iteration budget for the stricter 1e-6 beta-comparison plots."""

    if max_iter_override is not None:
        return max_iter_override
    return 20000


def format_decimal_for_name(value):
    text = format(value, ".1f")
    return text.replace(".", "_")


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


def force_label_dimension(train_labels, test_labels, n_classes):
    """Ensure that one_hot creates exactly n_classes output rows."""

    fixed_train_labels = np.array(train_labels, dtype=int, copy=True)
    fixed_test_labels = np.array(test_labels, dtype=int, copy=True)

    if fixed_train_labels.size > 0:
        fixed_train_labels[0] = n_classes - 1
    if fixed_test_labels.size > 0:
        fixed_test_labels[0] = n_classes - 1

    return fixed_train_labels, fixed_test_labels


def append_beta_result(rows, histories, context, result, epsilon):
    rows.append(make_beta_summary_row(context, result, epsilon))
    histories.extend(make_beta_history_rows(context, result, epsilon))


def make_beta_summary_row(context, result, epsilon):
    """Create one summary row for the fixed/variable beta experiment."""

    scenario = context["scenario"]
    instance = context["instance"]
    spectral = context["spectral"]
    objective = context["objective_fn"](result.weights)
    objective_gap = max(0.0, objective - context["reference_objective"])
    final_gradient = gradient(result.weights, instance.q, instance.c)
    final_gradient_norm = float(np.sqrt(np.sum(final_gradient * final_gradient)))

    train_scores = result.weights @ instance.h_train_aug
    test_scores = result.weights @ instance.h_test_aug

    return {
        "dataset": scenario["dataset_name"],
        "algorithm": result.method,
        "beta_rule": beta_rule_for_method(result.method),
        "epsilon": epsilon,
        "hidden_width": instance.hidden_width,
        "output_weight_count": output_weight_count(instance),
        "converged": result.converged,
        "iterations": result.iterations,
        "computational_time_seconds": result.elapsed_seconds,
        "final_gradient_norm": final_gradient_norm,
        "objective_gap": objective_gap,
        "relative_error_to_reference": relative_error(
            result.weights,
            context["reference_weights"],
        ),
        "train_accuracy": classification_accuracy(
            train_scores,
            instance.train_labels,
        ),
        "test_accuracy": classification_accuracy(
            test_scores,
            instance.test_labels,
        ),
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral.l_smooth,
        "mu": spectral.mu,
        "condition_number": spectral.condition_estimate,
        "alpha": value_or_empty(result.alpha),
        "beta": value_or_empty(result.beta),
    }


def make_beta_history_rows(context, result, epsilon):
    """Convert the recorded convergence history into CSV rows."""

    rows = []
    history = result.history
    scenario = context["scenario"]
    denominator = max(1.0, abs(context["reference_objective"]))

    for index in range(len(history["iteration"])):
        objective_gap = max(
            0.0,
            history["objective"][index] - context["reference_objective"],
        )
        rows.append(
            {
                "dataset": scenario["dataset_name"],
                "algorithm": result.method,
                "beta_rule": beta_rule_for_method(result.method),
                "epsilon": epsilon,
                "hidden_width": context["instance"].hidden_width,
                "iteration": int(history["iteration"][index]),
                "gradient_norm": history["grad_norm"][index],
                "relative_objective_gap": objective_gap / denominator,
                "relative_error_to_reference": history["relative_error"][index],
            }
        )

    return rows


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


def nesterov_beta_hidden_sizes():
    """Hidden sizes used in the compact 2x2 beta plots."""

    return [100, 500]


def nesterov_beta_epsilons():
    """Tolerance values used in the beta comparison."""

    return [1e-3, 1e-6]


def dimension_scaling_sizes():
    """Output-weight sizes for the Digits solver comparison."""

    configs = []
    total_weights = [10000, 25000, 50000, 100000]

    for value in total_weights:
        configs.append(
            {
                "target_output_weights": value,
            }
        )

    return configs


def run_digits_dimension_scaling(seed, epsilon, max_iter, record_every):
    """Compare our three methods on Digits as the ELM size increases."""

    check_sklearn_available()
    rows = []
    history_rows = []
    configs = dimension_scaling_sizes()

    for index in range(len(configs)):
        config = configs[index]
        target_weights = config["target_output_weights"]
        print(
            "Running Digits dimension scaling: output weights="
            + str(target_weights)
        )

        instance = create_real_comparison_instance(
            "digits",
            target_weights,
            seed + index,
        )

        # This experiment intentionally uses the full primal matrix Q.
        # No dual formulation is used, because the goal is to show the
        # computational bottleneck of our LDLT factorization.
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
                make_dimension_scaling_row(
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
                make_dimension_history_rows(
                    context,
                    result,
                    output_weight_count(instance),
                    epsilon,
                )
            )

    return rows, history_rows


def make_dimension_scaling_row(result, instance, epsilon):
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


def synthetic_sweep_values():
    values = []
    for index in range(1, 10):
        values.append(index / 10.0)
    return values


def synthetic_time_plot_levels():
    """Five readable levels used in the runtime bar plots."""

    return [0.1, 0.3, 0.5, 0.7, 0.9]


def synthetic_sweep_config():
    """Fixed synthetic size for the new rho/sparsity experiments."""

    return {
        "n_train": 1000,
        "n_test": 350,
        "n_features": 100,
        "n_classes": 100,
        "target_output_weights": 50000,
    }


def create_conditioning_effect_instance(rho, config, seed):
    """Create the fixed-size linear ELM used for the rho experiment."""

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
    """Generate features with covariance (1-rho)I + rho 11^T."""

    rng = np.random.default_rng(seed)
    rho = min(max(float(rho), 0.0), 0.99)

    covariance = (1.0 - rho) * np.eye(n_features)
    covariance = covariance + rho * np.ones((n_features, n_features))
    covariance = covariance + 1e-12 * np.eye(n_features)

    # Cholesky is used only to sample correlated data. It is not used to
    # solve the ELM optimization system.
    covariance_factor = np.linalg.cholesky(covariance)
    centers = 0.4 * rng.normal(size=(n_features, n_classes))

    train_labels = rng.integers(0, n_classes, size=n_train)
    test_labels = rng.integers(0, n_classes, size=n_test)
    train_noise = covariance_factor @ rng.normal(size=(n_features, n_train))
    test_noise = covariance_factor @ rng.normal(size=(n_features, n_test))

    x_train = centers[:, train_labels] + train_noise
    x_test = centers[:, test_labels] + test_noise
    x_train, x_test = standardize_train_test(x_train, x_test)
    return x_train, train_labels, x_test, test_labels


def sparse_hidden_sizes():
    """Hidden-layer sizes used in the sparse runtime experiment."""

    return [500, 1000, 2000, 4000, 8000, 16000]


def sparse_scaling_levels():
    """Sparsity levels used in the hidden-size scaling table."""

    return [0.1, 0.5, 0.9]


def choose_sparse_scaling_max_iter(max_iter_override):
    """Iteration budget for HB and Nesterov in the sparse size table."""

    if max_iter_override is not None:
        return max_iter_override
    return 10000


def create_sparse_scaling_instance(hidden_width, sparsity, seed):
    """Create the sparse synthetic ELM used in the hidden-size table.

    This follows the same synthetic sparse construction used in the other
    experiments: correlated features are generated first, then a random
    feature mask introduces zeros. The hidden layer is random and fixed. The
    optimized variable is only the output matrix.

    We do not build Q here. For large hidden layers Q can be very large, so it
    is constructed only when the LDLT timing is measured.
    """

    n_train = 1000
    n_test = 350
    n_features = 100
    n_classes = 4
    lambda_reg = 5e-3
    activation = "relu"
    hidden_scale = 0.9
    correlation = 0.25
    class_sep = 1.8
    noise = 1.0
    feature_scales = np.linspace(1.0, 0.3, n_features)

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

    x_train, x_test = apply_sparse_feature_mask(
        x_train,
        x_test,
        zero_probability=sparsity,
        seed=seed + 1000,
    )

    y_train = one_hot(train_labels, n_classes)
    y_test = one_hot(test_labels, n_classes)

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
    h_test = build_hidden_matrix(
        x_test,
        hidden_weights,
        hidden_bias,
        activation,
    )
    h_train_aug = augment_hidden_matrix(h_train)
    h_test_aug = augment_hidden_matrix(h_test)

    c = (y_train @ h_train_aug.T) / n_train

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
        None,
        c,
        lambda_reg,
        activation,
    )


def estimate_spectral_bounds_from_hidden(instance):
    """Estimate L and mu without materializing the large primal Q matrix."""

    h_aug = instance.h_train_aug
    gram = (h_aug.T @ h_aug) / instance.n_train
    eigenvalues = np.linalg.eigvalsh(gram)

    raw_l = float(np.max(eigenvalues))
    l_smooth = (raw_l + instance.lambda_reg) * 1.01
    mu = instance.lambda_reg
    condition_number = l_smooth / mu

    return {
        "mu": mu,
        "l_smooth": l_smooth,
        "condition_number": condition_number,
    }


def run_sparse_dimension_scaling(seed, epsilon, max_iter):
    """Measure runtime while hidden nodes double in sparse synthetic ELMs."""

    rows = []
    hidden_values = sparse_hidden_sizes()
    sparsity_values = sparse_scaling_levels()

    for sparsity_index in range(len(sparsity_values)):
        sparsity = sparsity_values[sparsity_index]

        for hidden_index in range(len(hidden_values)):
            hidden_width = hidden_values[hidden_index]
            local_seed = seed + 6000 + 100 * sparsity_index + hidden_index
            print(
                "Running sparse hidden scaling: sparsity="
                + format(sparsity, ".1f")
                + ", hidden nodes L="
                + str(hidden_width)
            )

            instance = create_sparse_scaling_instance(
                hidden_width,
                sparsity,
                local_seed,
            )
            spectral = estimate_spectral_bounds_from_hidden(instance)

            w0 = np.zeros_like(instance.c)
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

            q, c = build_primal_elm_system(instance)
            ldlt_result = ldlt_solve_weights(q, c)
            q_dimension = q.shape[0]
            del q
            del c

            results = [hb_result, nag_result, ldlt_result]
            for result in results:
                rows.append(
                    make_sparse_scaling_row(
                        result,
                        instance,
                        sparsity,
                        hidden_width,
                        q_dimension,
                        epsilon,
                        spectral,
                    )
                )

    return rows


def make_sparse_scaling_row(
    result,
    instance,
    sparsity,
    hidden_width,
    q_dimension,
    epsilon,
    spectral,
):
    """One CSV row for the sparse hidden-size runtime table."""

    return {
        "algorithm": result.method,
        "sparsity": sparsity,
        "hidden_nodes": hidden_width,
        "computational_time_seconds": result.elapsed_seconds,
        "epsilon": epsilon,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_gradient_norm": result.final_gradient_norm,
        "q_dimension": q_dimension,
        "n_train": instance.n_train,
        "n_test": instance.n_test,
        "n_features": instance.n_features,
        "n_classes": instance.n_classes,
        "lambda_reg": instance.lambda_reg,
        "estimated_L": spectral["l_smooth"],
        "mu": spectral["mu"],
        "condition_number": spectral["condition_number"],
        "alpha": value_or_empty(result.alpha),
        "beta": value_or_empty(result.beta),
    }


def run_synthetic_sweep(
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
    config = synthetic_sweep_config()
    values = synthetic_sweep_values()

    if analysis_type == "sparsity":
        kind = "sparse"
        parameter_name = "sparsity"
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
            make_ldlt_sweep_summary(
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
                row = make_synthetic_run_row(
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
                make_iterative_sweep_summary(
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
                make_synthetic_convergence_rows(
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

    max_iter_values = [
        max_iter,
        max_iter * 2,
        max_iter * 4,
        max_iter * 8,
    ]

    last_results = None
    search_plan = []
    for plot_max_iter in max_iter_values:
        search_plan.append((shared_plot_seed, plot_max_iter))

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


def make_synthetic_run_row(
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


def make_iterative_sweep_summary(
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


def make_ldlt_sweep_summary(
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


def make_synthetic_convergence_rows(
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


def build_nesterov_beta_case_groups(seed, hidden_sizes=None):
    """Create the Digits cases shown in Figures 7 and 8 of the report."""

    if hidden_sizes is None:
        hidden_sizes = nesterov_beta_hidden_sizes()

    scenarios = []
    for index in range(len(hidden_sizes)):
        hidden_width = hidden_sizes[index]
        scenarios.append(
            create_digits_beta_scenario(
                hidden_width,
                seed + index,
            )
        )

    return [
        (
            "Digits real dataset",
            "digits_nesterov_beta.png",
            scenarios,
        )
    ]


def create_digits_beta_scenario(hidden_width, seed):
    """Build one Digits ELM for the fixed-beta validation."""

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
        hidden_width=hidden_width,
        lambda_reg=1e-2,
        activation="tanh",
        hidden_scale=0.8,
        seed=seed,
        standardize_data=True,
    )

    return {
        "dataset_name": "digits",
        "scenario_type": "real_dataset_hidden_" + str(hidden_width),
        "correlation_strength": "",
        "zero_probability": "",
        "description": "Digits fixed-beta versus variable-beta comparison.",
        "instance": instance,
    }


def prepare_analysis_context(scenario, seed):
    """Prepare spectral constants and exact references for one beta case."""

    instance = scenario["instance"]
    spectral = estimate_spectral_bounds(
        instance.q,
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

    # NumPy is used only to define the trusted reference for the plots.
    reference_weights = np.linalg.solve(instance.q, instance.c.T).T
    return {
        "scenario": scenario,
        "instance": instance,
        "spectral": spectral,
        "objective_fn": objective_fn,
        "reference_weights": reference_weights,
        "reference_objective": objective_fn(reference_weights),
    }


def run_nesterov_beta_pair(
    scenario,
    seed,
    epsilon,
    max_iter,
    record_every,
):
    """Run only the two Nesterov variants needed for this comparison."""

    context = prepare_analysis_context(scenario, seed)
    instance = context["instance"]
    spectral = context["spectral"]
    w0 = np.zeros_like(instance.c)

    fixed_result = nesterov_accelerated_gradient(
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
    variable_result = nesterov_variable_beta(
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
    return context, fixed_result, variable_result


def run_nesterov_beta_comparison(
    seed,
    max_iter,
    record_every,
    figures_dir,
):
    """Create the 2x2 beta-comparison plots used in the report."""

    rows = []
    histories = []
    epsilons = nesterov_beta_epsilons()
    hidden_sizes = nesterov_beta_hidden_sizes()

    for epsilon_index in range(len(epsilons)):
        epsilon = epsilons[epsilon_index]
        case_groups = build_nesterov_beta_case_groups(
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

                append_beta_result(rows, histories, context, fixed_result, epsilon)
                append_beta_result(rows, histories, context, variable_result, epsilon)

                plot_items.append(
                    {
                        "context": context,
                        "results": [fixed_result, variable_result],
                    }
                )

            output_name = beta_figure_name(filename, epsilon)
            plot_nesterov_beta_grid(
                plot_items,
                epsilon,
                figures_dir / output_name,
                case_title,
            )

    return rows, histories


def beta_figure_name(filename, epsilon):
    if filename.endswith(".png"):
        base_name = filename[:-4]
    else:
        base_name = filename

    epsilon_text = format_epsilon_label(epsilon)
    return base_name + "_" + epsilon_text + ".png"


def beta_plot_label(method):
    if method == "Nesterov":
        return "Nesterov fixed beta"
    return method


def plot_nesterov_beta_grid(plot_items, epsilon, path, title):
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
        + ", ".join(str(value) for value in nesterov_beta_hidden_sizes())
        + ", epsilon="
        + format_epsilon_label(epsilon)
    )
    fig.suptitle(title + "\n" + subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def run_real_optimizer_comparison(
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
                        make_real_optimizer_row(
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


def make_real_optimizer_row(
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
        "iterations": result.iterations,
        "hidden_layer_size": instance.hidden_width,
        "alpha": alpha,
        "beta": beta,
    }
    return row


def plot_real_time_by_hidden_size(rows, figures_dir, epsilon):
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
                label=short_optimizer_label(method),
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

    fig.savefig(figures_dir / "real_datasets_time_by_hidden_size.png", dpi=160)
    plt.close(fig)


def short_optimizer_label(method):
    if method == "PyTorch SGD momentum":
        return "PyTorch HB"
    if method == "PyTorch SGD Nesterov":
        return "PyTorch Nesterov"
    return method


def plot_real_time_by_epsilon(rows, figures_dir):
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
                label=short_optimizer_label(method),
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
        figures_dir / "real_datasets_time_by_tolerance.png",
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
        "algorithm": "LDLT",
        "hidden_layer_size": instance.hidden_width,
        "output_weight_count": output_weight_count(instance),
        "q_dimension": q.shape[0],
        "minimum_value": minimum_value,
        "computational_time_seconds": elapsed,
        "residual_norm": residual_norm,
        "relative_residual_norm": relative_residual,
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
        "algorithm": "SciPy LDLT",
        "hidden_layer_size": instance.hidden_width,
        "output_weight_count": output_weight_count(instance),
        "q_dimension": q.shape[0],
        "minimum_value": minimum_value,
        "computational_time_seconds": elapsed,
        "residual_norm": residual_norm,
        "relative_residual_norm": relative_residual,
        "relative_difference_to_our_ldlt": relative_error(weights, our_weights),
    }
    return row, weights


def relative_residual_norm(residual, c):
    residual_norm = float(np.sqrt(np.sum(residual * residual)))
    c_norm = float(np.sqrt(np.sum(c * c)))
    if c_norm <= 1e-15:
        return residual_norm
    return residual_norm / c_norm


def remove_old_ldlt_outputs(figures_dir):
    path = figures_dir / "ldlt_solver_times.png"
    if path.exists():
        path.unlink()


def plot_ldlt_solver_times(rows, figures_dir):
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
    fig.savefig(figures_dir / "ldlt_solver_times.png", dpi=160)
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
    l_smooth = (raw_l + instance.lambda_reg) * 1.01
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


def make_dimension_history_rows(
    context,
    result,
    output_weights,
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
            "algorithm": result.method,
            "iteration": int(iterations[index]),
            "gradient_norm": grad_norms[index],
            "relative_gap": relative_gap,
            "output_weight_count": output_weights,
            "epsilon": epsilon,
        }
        rows.append(row)

    return rows


def remove_old_dimension_scaling_outputs(output_dir, figures_dir):
    """Clear the Digits dimension outputs before a fresh run."""

    csv_names = [
        "digits_dimension_scaling.csv",
        "digits_dimension_convergence.csv",
    ]
    figure_names = [
        "digits_dimension_scaling.png",
        "digits_dimension_scaling_bars.png",
        "digits_dimension_convergence.png",
    ]

    for csv_name in csv_names:
        path = output_dir / csv_name
        if path.exists():
            path.unlink()

    for figure_name in figure_names:
        path = figures_dir / figure_name
        if path.exists():
            path.unlink()


def marker_for_time_method(method):
    markers = {
        "LDLT": "o",
        "Heavy Ball": "s",
        "Nesterov": "^",
        "SciPy LDLT": "v",
    }
    if method in markers:
        return markers[method]
    return "o"


def plot_dimension_scaling_times(rows, figures_dir):
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
        "Digits Solver Runtime, epsilon="
        + format_epsilon_label(epsilon)
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "digits_dimension_scaling.png",
        dpi=160,
    )
    plt.close(fig)


def plot_dimension_scaling_bars(rows, figures_dir):
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
        "Digits Solver Runtime, epsilon="
        + format_epsilon_label(epsilon)
    )
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "digits_dimension_scaling_bars.png",
        dpi=160,
    )
    plt.close(fig)


def plot_digits_dimension_convergence(
    rows,
    epsilon,
    path,
):
    """Plot iterative convergence for each Digits model size."""

    if len(rows) == 0:
        return

    dimensions = unique_numeric_values(rows, "output_weight_count")
    methods = ["Heavy Ball", "Nesterov"]
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    for dim_index in range(len(dimensions)):
        dimension = dimensions[dim_index]
        color = colors(dim_index)

        for method in methods:
            selected = []
            for row in rows:
                same_method = row["algorithm"] == method
                same_dimension = (
                    float(row["output_weight_count"]) == float(dimension)
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
            axes[0].scatter(
                [iterations[-1]],
                [relative_gaps[-1]],
                color=color,
                marker=marker,
                s=34,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
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
            axes[1].scatter(
                [iterations[-1]],
                [gradient_norms[-1]],
                color=color,
                marker=marker,
                s=34,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

    axes[0].set_title("Relative Gap")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("(f(W)-f*) / max(1, |f*|)")

    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||grad f(W)||_F")
    axes[1].axhline(
        epsilon,
        color="black",
        linestyle=":",
        linewidth=1.4,
        label="epsilon",
    )

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)

    dimension_text = []
    for dimension in dimensions:
        dimension_text.append(str(int(dimension)))

    fig.suptitle(
        "Heavy Ball and Nesterov on Digits\n"
        + "epsilon="
        + format_epsilon_label(epsilon)
        + ", output weights="
        + ", ".join(dimension_text)
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_synthetic_convergence(
    rows,
    analysis_type,
    shared_plot_seed,
    figures_dir,
):
    """Create one convergence plot for each rho/sparsity value."""

    if len(rows) == 0:
        return

    if analysis_type == "sparsity":
        parameter_label = "sparsity"
    else:
        parameter_label = "rho"

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

        if analysis_type == "sparsity":
            percentage = int(round(100.0 * value))
            filename = "sparsity_" + str(percentage) + "_percent_convergence.png"
        else:
            filename = "rho_" + format(value, ".1f") + "_convergence.png"
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def plot_seed_for_selected_rows(rows, fallback_seed):
    for row in rows:
        value = row.get("plot_initialization_seed", "")
        if value != "":
            return value
    return fallback_seed


def plot_max_iter_for_selected_rows(rows):
    for row in rows:
        value = row.get("plot_max_iter", "")
        if value != "":
            return value

    largest_iteration = 0
    for row in rows:
        largest_iteration = max(largest_iteration, int(row["iteration"]))
    return largest_iteration


def plot_status_for_selected_rows(rows):
    status_by_method = {}
    for row in rows:
        if "plot_converged" in row:
            status_by_method[row["method"]] = row["plot_converged"]

    if len(status_by_method) == 0:
        return "plot convergence status unavailable"

    for value in status_by_method.values():
        if value is not True:
            return "at least one plot run did not converge"
    return "both plot runs converged"


def plot_synthetic_times(rows, analysis_type, path):
    """Bar plot of mean runtime for selected sparsity/rho values."""

    if len(rows) == 0:
        return

    methods = ["LDLT", "Heavy Ball", "Nesterov"]
    selected_values = synthetic_time_plot_levels()
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


def remove_old_synthetic_outputs(output_dir, figures_dir):
    """Clear the rho and sparsity outputs before a fresh run."""

    output_patterns = [
        "synthetic_sweeps_summary.csv",
        "synthetic_sweeps_runs.csv",
    ]
    figure_patterns = [
        "rho_*_convergence.png",
        "sparsity_*_convergence.png",
        "rho_time_comparison.png",
        "sparsity_time_comparison.png",
    ]

    remove_paths_matching(output_dir, output_patterns)
    remove_paths_matching(figures_dir, figure_patterns)


def remove_old_nesterov_beta_outputs(output_dir, figures_dir):
    """Clear the Nesterov beta outputs before a fresh run."""

    output_patterns = [
        "nesterov_beta_comparison.csv",
        "nesterov_beta_convergence.csv",
    ]
    figure_patterns = [
        "digits_nesterov_beta_*.png",
    ]

    remove_paths_matching(output_dir, output_patterns)
    remove_paths_matching(figures_dir, figure_patterns)


def remove_paths_matching(base_dir, patterns):
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file():
                path.unlink()


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


def new_local_history():
    return {
        "iteration": [],
        "grad_norm": [],
        "objective": [],
        "relative_error": [],
    }


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

    if (
        len(history["iteration"]) == 0
        or int(history["iteration"][-1]) != iterations
    ):
        record_local_history(
            history,
            iterations,
            final_evaluation_point,
            final_grad_norm,
            objective_fn,
            reference_weights,
        )

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


def write_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def metric_floor(field):
    return 1e-16


def unique_numeric_values(rows, field):
    values = []
    for row in rows:
        value = float(row[field])
        if value not in values:
            values.append(value)
    values.sort()
    return values


def format_epsilon_label(value):
    return format(float(value), ".0e")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help=(
            "Tolerance for the Digits dimension scaling and sparse hidden-size "
            "scaling experiments."
        ),
    )
    parser.add_argument(
        "--synthetic-sweep-tol",
        type=float,
        default=1e-6,
        help="Tolerance for the 100-initialization rho and sparsity sweeps.",
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=50)
    parser.add_argument(
        "--only-real-optimizer-comparison",
        action="store_true",
        help=(
            "Run only the Wine/Digits comparison between our HB/Nesterov "
            "and PyTorch HB/Nesterov using the same random initial weights."
        ),
    )
    parser.add_argument(
        "--only-ldlt-scipy-comparison",
        action="store_true",
        help="Run only the LDLT comparison between our implementation and SciPy LDLT.",
    )
    parser.add_argument(
        "--only-digits-dimension-scaling",
        action="store_true",
        help=(
            "Run only the dimensionality comparison between our LDLT, Heavy "
            "Ball, and Nesterov implementations."
        ),
    )
    parser.add_argument(
        "--only-synthetic-sweeps",
        action="store_true",
        help=(
            "Run only the new synthetic sparsity and rho sweeps with our "
            "hand-written LDLT, Heavy Ball, and Nesterov methods."
        ),
    )
    parser.add_argument(
        "--only-sparse-dimension-scaling",
        action="store_true",
        help=(
            "Run only the synthetic sparse runtime table while hidden nodes "
            "double from 500 to 16000."
        ),
    )
    parser.add_argument(
        "--only-nesterov-beta-comparison",
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
        help="Number of random initializations for HB/Nesterov in the synthetic sparsity/rho sweep.",
    )
    parser.add_argument(
        "--real-output-weights",
        default="10000,50000,100000",
        help="Comma-separated target counts for the trainable output weights in the real-data comparisons.",
    )
    parser.add_argument(
        "--real-epsilons",
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

    # Determine what to run (run all by default if no flags specified)
    run_all = not (
        args.only_real_optimizer_comparison
        or args.only_ldlt_scipy_comparison
        or args.only_digits_dimension_scaling
        or args.only_synthetic_sweeps
        or args.only_sparse_dimension_scaling
        or args.only_nesterov_beta_comparison
    )

    # 1. Real PyTorch same initialization comparison
    if run_all or args.only_real_optimizer_comparison:
        print("\n--- Comparing iterative optimizers on Wine and Digits ---")
        rows = run_real_optimizer_comparison(
            output_weight_values=parse_int_list(
                args.real_output_weights,
            ),
            epsilon_values=parse_float_list(
                args.real_epsilons,
            ),
            max_iter=choose_max_iter(args.max_iter),
            record_every=args.record_every,
            seed=args.seed,
        )
        output_path = output_dir / "real_dataset_optimizer_comparison.csv"
        write_csv(output_path, REAL_OPTIMIZER_FIELDS, rows)
        plot_real_time_by_hidden_size(
            rows,
            figures_dir,
            epsilon=1e-3,
        )
        plot_real_time_by_epsilon(
            rows,
            figures_dir,
        )
        print("Wrote " + str(output_path))

    # 2. LDLT vs SciPy LDLT comparison
    if run_all or args.only_ldlt_scipy_comparison:
        print("\n--- Running LDLT vs SciPy LDLT comparison ---")
        rows = run_ldlt_scipy_comparison(
            output_weight_values=parse_int_list(
                args.real_output_weights,
            ),
            seed=args.seed,
        )
        output_path = output_dir / "ldlt_solver_comparison.csv"
        write_csv(output_path, LDLT_COMPARISON_FIELDS, rows)
        remove_old_ldlt_outputs(figures_dir)
        plot_ldlt_solver_times(rows, figures_dir)
        print("Wrote " + str(output_path))

    # 3. Handwritten dimension scaling comparison
    if run_all or args.only_digits_dimension_scaling:
        print("\n--- Running Digits dimension scaling ---")
        epsilon = args.tol
        scaling_max_iter = choose_dimension_scaling_max_iter(args.max_iter)
        rows, convergence_rows = run_digits_dimension_scaling(
            seed=args.seed,
            epsilon=epsilon,
            max_iter=scaling_max_iter,
            record_every=args.record_every,
        )

        remove_old_dimension_scaling_outputs(output_dir, figures_dir)

        output_path = output_dir / "digits_dimension_scaling.csv"
        convergence_path = output_dir / "digits_dimension_convergence.csv"
        write_csv(output_path, DIMENSION_SCALING_FIELDS, rows)
        write_csv(
            convergence_path,
            DIMENSION_CONVERGENCE_FIELDS,
            convergence_rows,
        )
        plot_dimension_scaling_times(rows, figures_dir)
        plot_dimension_scaling_bars(rows, figures_dir)
        plot_digits_dimension_convergence(
            convergence_rows,
            epsilon,
            figures_dir / "digits_dimension_convergence.png",
        )

        print("Wrote " + str(output_path))
        print("Wrote " + str(convergence_path))

    # 4. Nesterov Beta Focused comparison
    if run_all or args.only_nesterov_beta_comparison:
        print("\n--- Comparing fixed and variable Nesterov beta ---")
        max_iter = choose_beta_comparison_max_iter(args.max_iter)
        remove_old_nesterov_beta_outputs(output_dir, figures_dir)
        rows, history_rows = run_nesterov_beta_comparison(
            args.seed,
            max_iter,
            args.record_every,
            figures_dir,
        )

        summary_path = output_dir / "nesterov_beta_comparison.csv"
        history_path = output_dir / "nesterov_beta_convergence.csv"
        write_csv(summary_path, BETA_COMPARISON_FIELDS, rows)
        write_csv(history_path, BETA_CONVERGENCE_FIELDS, history_rows)

        print("Wrote " + str(summary_path))
        print("Wrote " + str(history_path))

    # 5. Synthetic parameter sweeps
    if run_all or args.only_synthetic_sweeps:
        print("\n--- Running synthetic rho and sparsity sweeps ---")
        epsilon = args.synthetic_sweep_tol
        sweep_max_iter = choose_synthetic_sweep_max_iter(args.max_iter)
        initialization_trials = args.synthetic_init_trials
        shared_plot_seed = args.seed + 910000
        remove_old_synthetic_outputs(output_dir, figures_dir)

        sparsity_aggregate, sparsity_trials, sparsity_plot_history = run_synthetic_sweep(
            analysis_type="sparsity",
            seed=args.seed + 3000,
            epsilon=epsilon,
            max_iter=sweep_max_iter,
            record_every=args.record_every,
            initialization_trials=initialization_trials,
            shared_plot_seed=shared_plot_seed,
        )
        rho_aggregate, rho_trials, rho_plot_history = run_synthetic_sweep(
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

        aggregate_path = output_dir / "synthetic_sweeps_summary.csv"
        trial_path = output_dir / "synthetic_sweeps_runs.csv"

        write_csv(
            aggregate_path,
            SYNTHETIC_SWEEP_SUMMARY_FIELDS,
            aggregate_rows,
        )
        write_csv(trial_path, SYNTHETIC_SWEEP_RUN_FIELDS, trial_rows)

        plot_synthetic_convergence(
            sparsity_plot_history,
            "sparsity",
            shared_plot_seed,
            figures_dir,
        )
        plot_synthetic_times(
            aggregate_rows,
            "sparsity",
            figures_dir / "sparsity_time_comparison.png",
        )
        plot_synthetic_convergence(
            rho_plot_history,
            "rho",
            shared_plot_seed,
            figures_dir,
        )
        plot_synthetic_times(
            aggregate_rows,
            "rho",
            figures_dir / "rho_time_comparison.png",
        )

        print("Wrote " + str(aggregate_path))
        print("Wrote " + str(trial_path))

    # 6. Synthetic sparse hidden scaling comparison
    if run_all or args.only_sparse_dimension_scaling:
        print("\n--- Running sparse dimension scaling ---")
        max_iter = choose_sparse_scaling_max_iter(args.max_iter)
        rows = run_sparse_dimension_scaling(
            seed=args.seed,
            epsilon=args.tol,
            max_iter=max_iter,
        )
        output_path = output_dir / "sparse_dimension_scaling.csv"
        write_csv(output_path, SPARSE_DIMENSION_FIELDS, rows)
        print("Wrote " + str(output_path))


if __name__ == "__main__":
    main()
