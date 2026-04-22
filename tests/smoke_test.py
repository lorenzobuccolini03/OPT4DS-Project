"""Small executable smoke test for the ELM optimization package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elm_optimization.algorithms import (
    estimate_spectral_bounds,
    heavy_ball,
    ldlt_solve_weights,
    nesterov_accelerated_gradient,
)
from elm_optimization.elm import create_elm_classification_instance
from elm_optimization.metrics import objective_value, relative_error


def main() -> None:
    instance = create_elm_classification_instance(
        n_train=120,
        n_test=40,
        n_features=8,
        n_classes=3,
        hidden_width=20,
        lambda_reg=1e-2,
        seed=123,
    )

    ldlt = ldlt_solve_weights(instance.q, instance.c)
    numpy_reference = np.linalg.solve(instance.q, instance.c.T).T
    assert relative_error(ldlt.weights, numpy_reference) < 1e-9
    assert ldlt.final_gradient_norm < 1e-9

    spectral = estimate_spectral_bounds(instance.q, instance.lambda_reg, seed=123)
    objective_fn = lambda w: objective_value(
        w, instance.h_train_aug, instance.y_train, instance.lambda_reg
    )
    w0 = np.zeros_like(instance.c)

    hb = heavy_ball(
        instance.q,
        instance.c,
        w0,
        mu=spectral.mu,
        l_smooth=spectral.l_smooth,
        tol=1e-5,
        max_iter=5000,
        objective_fn=objective_fn,
        reference_weights=ldlt.weights,
    )
    nag = nesterov_accelerated_gradient(
        instance.q,
        instance.c,
        w0,
        mu=spectral.mu,
        l_smooth=spectral.l_smooth,
        tol=1e-5,
        max_iter=5000,
        objective_fn=objective_fn,
        reference_weights=ldlt.weights,
    )

    assert hb.converged, f"Heavy Ball did not converge: {hb.final_gradient_norm}"
    assert nag.converged, f"Nesterov did not converge: {nag.final_gradient_norm}"
    assert relative_error(hb.weights, ldlt.weights) < 1e-3
    assert relative_error(nag.weights, ldlt.weights) < 1e-3

    print("Smoke test passed.")
    print(f"LDLT grad norm: {ldlt.final_gradient_norm:.3e}")
    print(f"Heavy Ball iterations: {hb.iterations}")
    print(f"Nesterov iterations: {nag.iterations}")


if __name__ == "__main__":
    main()
