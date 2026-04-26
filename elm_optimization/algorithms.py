"""Scratch implementations of the three project algorithms.

This file implements:

1. LDLT factorization and triangular solves for the direct method.
2. Heavy Ball with the optimal quadratic parameters from the theory.
3. Strongly-convex Nesterov Accelerated Gradient with constant parameters.

NumPy is used for array storage and basic dense arithmetic, but no built-in
linear-system solver, Cholesky routine, or optimization solver is used by the
project algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field # for convenient data classes to store results and parameters
from time import perf_counter # for measuring elapsed time of optimization methods
from typing import Callable # for type annotations of objective functions

import numpy as np

from .metrics import frobenius_norm # for computing the Frobenius norm of gradients and errors

ObjectiveFunction = Callable[[np.ndarray], float] # take weights as input and return a scalar value


@dataclass(frozen=True) # for storing spectral parameters needed by the first-order algorithms
class SpectralBounds:
    """Spectral parameters used by the first-order algorithms."""

    mu: float # the strong convexity parameter (lower bound on eigenvalues)
    l_smooth: float # the Lipschitz constant (upper bound on eigenvalues)
    condition_estimate: float # the estimated condition number (l_smooth / mu)
    power_iterations: int # the number of iterations taken by the Power Method to estimate the largest eigenvalue
    raw_largest_eigenvalue_estimate: float # the raw estimate of the largest eigenvalue from the Power Method before safety inflation


@dataclass
class OptimizationResult:
    """Result returned by an optimization method."""

    method: str # name of the optimization method used (e.g., "LDLT", "Heavy Ball", "Nesterov")
    weights: np.ndarray # the final weights found by the optimization method
    iterations: int # the number of iterations taken by the optimization method (0 for direct methods)
    converged: bool # whether the optimization method converged within the given tolerance and iteration limit
    elapsed_seconds: float # the time taken by the optimization method in seconds
    final_gradient_norm: float # the Frobenius norm of the final gradient at the solution
    alpha: float | None = None # the step size used by the first-order method (None for direct methods)
    beta: float | None = None # the momentum parameter used by the first-order method (None for direct methods)
    history: dict[str, list[float]] = field(default_factory=dict) # a history of optimization metrics recorded during the iterations (empty for direct methods)

# (A3) Cholesky-like factorization and solves for the direct method
def ldlt_factorize(q: np.ndarray, pivot_tol: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    """Compute Q = L diag(d) L.T for a dense Symmetric Positive Definite matrix.

    This is the square-root-free factorization described in Algorithm 1 of the
    theoretical report.  The matrix L has unit diagonal, while d stores the
    diagonal entries of D.  The implementation follows the column recursion:

        d_j = q_jj - sum_{k<j} L_jk^2 d_k
        L_ij = (q_ij - sum_{k<j} L_ik L_jk d_k) / d_j

    A vectorized matrix-vector product is used inside each column to keep the
    code efficient without delegating the factorization to a library.
    """

    q = np.asarray(q, dtype=float) # ensure q is a NumPy array of type float
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix.")
    if not np.allclose(q, q.T, rtol=1e-10, atol=1e-12):
        raise ValueError("q must be symmetric.")

    n = q.shape[0] # the size of the matrix
    l_factor = np.eye(n, dtype=float) # initialize L as the identity matrix (unit lower triangular)
    d = np.zeros(n, dtype=float) # initialize d as a zero vector to store the diagonal entries of D

    # The main loop:
    for j in range(n):
        if j == 0:
            diag_correction = 0.0
            v = np.empty(0, dtype=float) # empty vector for the first column, since there are no previous columns to consider
        else:
            # v_k = L_jk * d_k (vector initialized to store the products needed for the diagonal correction and column residual)
            v = l_factor[j, :j] * d[:j]
            diag_correction = float(l_factor[j, :j] @ v) # vectorized dot product to compute the diagonal correction for the pivot

        pivot = float(q[j, j] - diag_correction) # diagonal entry corrected by the contributions from previous columns
        if pivot <= pivot_tol:
            raise np.linalg.LinAlgError(
                "LDLT pivot is not positive. The matrix is not numerically SPD"
                f"at column {j}: pivot={pivot:.3e}."
            )
        d[j] = pivot 

        if j + 1 < n:
            if j == 0:
                column_residual = q[j + 1 :, j].copy() # case where v is empty
            else:
                column_residual = q[j + 1 :, j] - l_factor[j + 1 :, :j] @ v # dot product to compute the residual for the current column below the pivot
            l_factor[j + 1 :, j] = column_residual / d[j]

    return l_factor, d

# 1) Forward substitution to solve L Z = rhs for unit lower triangular L
def forward_substitution_unit_lower(l_factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve L X = rhs for unit lower triangular L."""

    rhs_2d, was_vector = _as_2d_rhs(rhs) # ensure rhs is 2D for consistent indexing, and remember if it was originally a vector
    n = l_factor.shape[0]
    x = np.zeros_like(rhs_2d, dtype=float)

    # The forward substitution loop:
    for i in range(n):
        x[i, :] = rhs_2d[i, :] - l_factor[i, :i] @ x[:i, :] # assign the solution for the current row by subtracting the contributions from previous rows
    return x[:, 0] if was_vector else x

# 2) Diagonal scaling
def diagonal_solve(d: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve diag(d) X = rhs by elementwise division."""

    rhs_2d, was_vector = _as_2d_rhs(rhs)
    x = rhs_2d / d[:, None]
    return x[:, 0] if was_vector else x

# 3) Backward substitution
def backward_substitution_unit_upper_from_lower(
    l_factor: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve L.T X = rhs using the stored lower triangular L."""

    rhs_2d, was_vector = _as_2d_rhs(rhs)
    n = l_factor.shape[0]
    x = np.zeros_like(rhs_2d, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i, :] = rhs_2d[i, :] - l_factor[i + 1 :, i] @ x[i + 1 :, :]

    return x[:, 0] if was_vector else x


def solve_with_ldlt(l_factor: np.ndarray, d: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Q X = rhs after Q = L diag(d) L.T has been computed."""

    z = forward_substitution_unit_lower(l_factor, rhs)
    v = diagonal_solve(d, z)
    return backward_substitution_unit_upper_from_lower(l_factor, v)


def ldlt_solve_weights(
    q: np.ndarray,
    c: np.ndarray,
    *,
    pivot_tol: float = 1e-14,
) -> OptimizationResult:
    """Solve the ELM optimality system Q W.T = C.T by scratch LDLT."""

    start = perf_counter()
    l_factor, d = ldlt_factorize(q, pivot_tol=pivot_tol)
    solution_transposed = solve_with_ldlt(l_factor, d, c.T)
    weights = solution_transposed.T
    elapsed = perf_counter() - start
    final_grad = weights @ q - c
    return OptimizationResult(
        method="LDLT",
        weights=weights,
        iterations=0,
        converged=True,
        elapsed_seconds=elapsed,
        final_gradient_norm=frobenius_norm(final_grad),
    )


def power_method_largest_eigenvalue(
    q: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 5000,
    seed: int = 0,
) -> tuple[float, int]:
    """Approximate lambda_max(Q) with the Power Method.

    This follows the computational discussion in the theory section: estimating
    the dominant eigenvalue costs only repeated matrix-vector products and
    avoids an O(n^3) eigendecomposition before the first-order algorithms.
    """

    rng = np.random.default_rng(seed)
    n = q.shape[0]
    v = rng.normal(size=n)
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        v[0] = 1.0
    else:
        v = v / v_norm

    previous = 0.0
    rayleigh = 0.0
    for iteration in range(1, max_iter + 1):
        qv = q @ v
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0.0:
            raise np.linalg.LinAlgError("Power method encountered a zero vector.")
        v = qv / qv_norm
        qv = q @ v
        rayleigh = float(v @ qv)

        if abs(rayleigh - previous) <= tol * max(1.0, abs(rayleigh)):
            return rayleigh, iteration
        previous = rayleigh

    return rayleigh, max_iter


def estimate_spectral_bounds(
    q: np.ndarray,
    lambda_reg: float,
    *,
    power_tol: float = 1e-8,
    power_max_iter: int = 5000,
    seed: int = 0,
    l_safety_factor: float = 1.01,
) -> SpectralBounds:
    """Estimate the spectral bounds required by HB and NAG.

    The theory explains that mu can be safely set to lambda_reg, because
    Q = H H.T / N + lambda I implies lambda_min(Q) >= lambda.  The largest
    eigenvalue is estimated by the Power Method and then slightly inflated so
    that the first-order steps do not accidentally use an underestimated
    Lipschitz constant.
    """

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be strictly positive.")
    raw_l, power_iterations = power_method_largest_eigenvalue(
        q, tol=power_tol, max_iter=power_max_iter, seed=seed
    )
    l_smooth = max(raw_l, lambda_reg) * l_safety_factor
    mu = lambda_reg
    return SpectralBounds(
        mu=mu,
        l_smooth=l_smooth,
        condition_estimate=l_smooth / mu,
        power_iterations=power_iterations,
        raw_largest_eigenvalue_estimate=raw_l,
    )


def heavy_ball(
    q: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    *,
    mu: float,
    l_smooth: float,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    objective_fn: ObjectiveFunction | None = None,
    reference_weights: np.ndarray | None = None,
    record_every: int = 1,
) -> OptimizationResult:
    """Run Polyak's Heavy Ball method on the ELM quadratic objective."""

    _validate_spectral_bounds(mu, l_smooth)
    sqrt_l = np.sqrt(l_smooth)
    sqrt_mu = np.sqrt(mu)
    alpha = 4.0 / (sqrt_l + sqrt_mu) ** 2
    beta = ((sqrt_l - sqrt_mu) / (sqrt_l + sqrt_mu)) ** 2

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    history = _new_history()
    start = perf_counter()
    converged = False
    final_grad_norm = np.inf
    iterations = 0

    for iteration in range(max_iter + 1):
        grad = weights @ q - c
        final_grad_norm = frobenius_norm(grad)
        if iteration % record_every == 0:
            _record_history(
                history,
                iteration=iteration,
                weights=weights,
                grad_norm=final_grad_norm,
                objective_fn=objective_fn,
                reference_weights=reference_weights,
            )
        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break
        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = weights - alpha * grad + beta * (weights - previous_weights)
        previous_weights, weights = weights, next_weights

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
        history=history,
    )


def nesterov_accelerated_gradient(
    q: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    *,
    mu: float,
    l_smooth: float,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    objective_fn: ObjectiveFunction | None = None,
    reference_weights: np.ndarray | None = None,
    record_every: int = 1,
) -> OptimizationResult:
    """Run the strongly-convex Nesterov Accelerated Gradient scheme.

    This is the constant-step variant in the theory:

        Y_t = W_t + beta (W_t - W_{t-1})
        W_{t+1} = Y_t - (1/L) grad f(Y_t)

    The gradient is evaluated at the look-ahead point Y_t, not at W_t.
    """

    _validate_spectral_bounds(mu, l_smooth)
    alpha = 1.0 / l_smooth
    beta = (np.sqrt(l_smooth) - np.sqrt(mu)) / (
        np.sqrt(l_smooth) + np.sqrt(mu)
    )

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    history = _new_history()
    start = perf_counter()
    converged = False
    final_grad_norm = np.inf
    iterations = 0
    final_evaluation_point = weights.copy()

    for iteration in range(max_iter + 1):
        evaluation_point = weights + beta * (weights - previous_weights)
        grad = evaluation_point @ q - c
        final_grad_norm = frobenius_norm(grad)
        final_evaluation_point = evaluation_point

        if iteration % record_every == 0:
            _record_history(
                history,
                iteration=iteration,
                weights=evaluation_point,
                grad_norm=final_grad_norm,
                objective_fn=objective_fn,
                reference_weights=reference_weights,
            )
        if final_grad_norm <= tol:
            converged = True
            iterations = iteration
            break
        if iteration == max_iter:
            iterations = iteration
            break

        next_weights = evaluation_point - alpha * grad
        previous_weights, weights = weights, next_weights

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
        history=history,
    )

# Appendix: Helper functions for the algorithms

# 1. Ensure the right-hand side is 2D for consistent indexing, and remember if it was originally a vector
def _as_2d_rhs(rhs: np.ndarray) -> tuple[np.ndarray, bool]: 
    rhs_array = np.asarray(rhs, dtype=float)
    was_vector = rhs_array.ndim == 1
    if was_vector:
        rhs_array = rhs_array[:, None]
    if rhs_array.ndim != 2:
        raise ValueError("rhs must be a vector or a matrix.")
    return rhs_array, was_vector

# 2. Validate the spectral bounds for the first-order algorithms
def _validate_spectral_bounds(mu: float, l_smooth: float) -> None:
    if mu <= 0.0:
        raise ValueError("mu must be strictly positive.")
    if l_smooth <= 0.0:
        raise ValueError("l_smooth must be strictly positive.")
    if l_smooth < mu:
        raise ValueError("l_smooth must be greater than or equal to mu.")

# 3. Create a new history dictionary for recording optimization metrics
def _new_history() -> dict[str, list[float]]:
    return {
        "iteration": [],
        "grad_norm": [],
        "objective": [],
        "relative_error": [],
    }

# 4. Record the current optimization metrics in the history dictionary
def _record_history(
    history: dict[str, list[float]],
    *,
    iteration: int,
    weights: np.ndarray,
    grad_norm: float,
    objective_fn: ObjectiveFunction | None,
    reference_weights: np.ndarray | None,
) -> None:
    history["iteration"].append(float(iteration))
    history["grad_norm"].append(float(grad_norm))
    if objective_fn is None:
        history["objective"].append(float("nan"))
    else:
        history["objective"].append(float(objective_fn(weights)))

    if reference_weights is None:
        history["relative_error"].append(float("nan"))
    else:
        numerator = frobenius_norm(weights - reference_weights)
        denominator = max(1.0, frobenius_norm(reference_weights))
        history["relative_error"].append(numerator / denominator)
