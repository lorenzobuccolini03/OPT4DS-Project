"""Optimization algorithms for regularized Extreme Learning Machines.

The package separates the ELM problem construction from the numerical
algorithms, as requested in the project guidelines.  The three required
methods are implemented in :mod:`elm_optimization.algorithms`.
"""

from .algorithms import (
    OptimizationResult,
    SpectralBounds,
    estimate_spectral_bounds,
    heavy_ball,
    ldlt_factorize,
    ldlt_solve_weights,
    nesterov_accelerated_gradient,
    power_method_largest_eigenvalue,
)
from .elm import (
    ELMInstance,
    create_elm_classification_instance,
    formulate_elm_system,
    predict_scores,
)
from .metrics import (
    classification_accuracy,
    gradient,
    mean_squared_error,
    objective_value,
    relative_error,
)

__all__ = [
    "ELMInstance",
    "OptimizationResult",
    "SpectralBounds",
    "classification_accuracy",
    "create_elm_classification_instance",
    "estimate_spectral_bounds",
    "formulate_elm_system",
    "gradient",
    "heavy_ball",
    "ldlt_factorize",
    "ldlt_solve_weights",
    "mean_squared_error",
    "nesterov_accelerated_gradient",
    "objective_value",
    "power_method_largest_eigenvalue",
    "predict_scores",
    "relative_error",
]
