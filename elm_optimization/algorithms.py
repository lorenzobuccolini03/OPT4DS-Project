"""Optimization algorithms used in the ELM project.

The algorithms are implemented directly:

1. LDLT factorization for the direct solution.
2. Heavy Ball method.
3. Nesterov Accelerated Gradient.

NumPy is used for matrix arithmetic, but the project algorithms do not call a
built-in optimizer, Cholesky factorization, or linear-system solver.
"""

from time import perf_counter

import numpy as np

from .metrics import frobenius_norm


class SpectralBounds:
    """Small container for the spectral constants used by HB and Nesterov."""

    def __init__(
        self,
        mu,
        l_smooth,
        condition_estimate,
        power_iterations,
        raw_largest_eigenvalue_estimate,
    ):
        self.mu = mu
        self.l_smooth = l_smooth
        self.condition_estimate = condition_estimate
        self.power_iterations = power_iterations
        self.raw_largest_eigenvalue_estimate = raw_largest_eigenvalue_estimate


class OptimizationResult:
    """Small container for the output of an optimization method."""

    def __init__(
        self,
        method,
        weights,
        iterations,
        converged,
        elapsed_seconds,
        final_gradient_norm,
        alpha=None,
        beta=None,
        history=None,
    ):
        self.method = method
        self.weights = weights
        self.iterations = iterations
        self.converged = converged
        self.elapsed_seconds = elapsed_seconds
        self.final_gradient_norm = final_gradient_norm
        self.alpha = alpha
        self.beta = beta

        if history is None:
            self.history = {}
        else:
            self.history = history


def ldlt_factorize(q, pivot_tol=1e-14):
    """Compute Q = L diag(d) L.T for a symmetric positive definite matrix."""

    q = np.asarray(q, dtype=float)

    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix.")
    if not np.allclose(q, q.T, rtol=1e-10, atol=1e-12):
        raise ValueError("q must be symmetric.")

    n = q.shape[0]
    l_factor = np.eye(n, dtype=float)
    d = np.zeros(n, dtype=float)

    for j in range(n):
        # Compute v_k = L_jk d_k for k < j.
        if j == 0:
            v = np.array([], dtype=float)
            diagonal_correction = 0.0
        else:
            v = l_factor[j, :j] * d[:j]
            diagonal_correction = float(l_factor[j, :j] @ v)

        # Formula from LDLT:
        # d_j = q_jj - sum_{k<j} L_jk^2 d_k
        pivot = float(q[j, j] - diagonal_correction)
        if pivot <= pivot_tol:
            message = (
                "LDLT pivot is not positive at column "
                + str(j)
                + ": pivot="
                + str(pivot)
            )
            raise np.linalg.LinAlgError(message)
        d[j] = pivot

        # Formula for the entries below the diagonal:
        # L_ij = (q_ij - sum_{k<j} L_ik L_jk d_k) / d_j
        for i in range(j + 1, n):
            if j == 0:
                correction = 0.0
            else:
                correction = float(l_factor[i, :j] @ v)
            l_factor[i, j] = (q[i, j] - correction) / d[j]

    return l_factor, d


def forward_substitution_unit_lower(l_factor, rhs):
    """Solve L X = rhs where L has ones on the diagonal."""

    rhs_2d, was_vector = _as_2d_rhs(rhs)
    n = l_factor.shape[0]
    x = np.zeros_like(rhs_2d, dtype=float)

    for i in range(n):
        previous_terms = l_factor[i, :i] @ x[:i, :]
        x[i, :] = rhs_2d[i, :] - previous_terms

    if was_vector:
        return x[:, 0]
    return x


def diagonal_solve(d, rhs):
    """Solve diag(d) X = rhs."""

    rhs_2d, was_vector = _as_2d_rhs(rhs)
    x = rhs_2d / d[:, None]

    if was_vector:
        return x[:, 0]
    return x


def backward_substitution_unit_upper_from_lower(l_factor, rhs):
    """Solve L.T X = rhs using the stored lower triangular matrix L."""

    rhs_2d, was_vector = _as_2d_rhs(rhs)
    n = l_factor.shape[0]
    x = np.zeros_like(rhs_2d, dtype=float)

    for i in range(n - 1, -1, -1):
        next_terms = l_factor[i + 1 :, i] @ x[i + 1 :, :]
        x[i, :] = rhs_2d[i, :] - next_terms

    if was_vector:
        return x[:, 0]
    return x


def solve_with_ldlt(l_factor, d, rhs):
    """Solve Q X = rhs after Q = L diag(d) L.T has been factorized."""

    z = forward_substitution_unit_lower(l_factor, rhs)
    v = diagonal_solve(d, z)
    x = backward_substitution_unit_upper_from_lower(l_factor, v)
    return x


def ldlt_solve_weights(q, c, pivot_tol=1e-14):
    """Solve Q W.T = C.T with the scratch LDLT factorization."""

    start = perf_counter()

    l_factor, d = ldlt_factorize(q, pivot_tol)
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


def power_method_largest_eigenvalue(q, tol=1e-8, max_iter=5000, seed=0):
    """Approximate the largest eigenvalue of Q with the Power Method."""

    rng = np.random.default_rng(seed)
    n = q.shape[0]

    v = rng.normal(size=n)
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        v[0] = 1.0
    else:
        v = v / v_norm

    previous_rayleigh = 0.0
    rayleigh = 0.0

    for iteration in range(1, max_iter + 1):
        qv = q @ v
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0.0:
            raise np.linalg.LinAlgError("Power method produced a zero vector.")

        v = qv / qv_norm

        # Rayleigh quotient: v.T Q v
        qv = q @ v
        rayleigh = float(v @ qv)

        difference = abs(rayleigh - previous_rayleigh)
        tolerance_value = tol * max(1.0, abs(rayleigh))
        if difference <= tolerance_value:
            return rayleigh, iteration

        previous_rayleigh = rayleigh

    return rayleigh, max_iter


def estimate_spectral_bounds(
    q,
    lambda_reg,
    power_tol=1e-8,
    power_max_iter=5000,
    seed=0,
    l_safety_factor=1.01,
):
    """Estimate mu and L for the first-order methods.

    For this ELM problem, Q = H H.T / N + lambda I, so lambda is a safe lower
    bound for the smallest eigenvalue. The largest eigenvalue is estimated by
    the Power Method.
    """

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be strictly positive.")

    raw_l, power_iterations = power_method_largest_eigenvalue(
        q,
        tol=power_tol,
        max_iter=power_max_iter,
        seed=seed,
    )

    mu = lambda_reg
    l_smooth = max(raw_l, lambda_reg) * l_safety_factor
    condition_estimate = l_smooth / mu

    return SpectralBounds(
        mu,
        l_smooth,
        condition_estimate,
        power_iterations,
        raw_l,
    )


def heavy_ball(
    q,
    c,
    w0,
    mu,
    l_smooth,
    tol=1e-6,
    max_iter=10000,
    objective_fn=None,
    reference_weights=None,
    record_every=1,
):
    """Run the Heavy Ball method on the ELM quadratic objective."""

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
        # Exact gradient: grad f(W) = W Q - C
        grad = weights @ q - c
        final_grad_norm = frobenius_norm(grad)

        if iteration % record_every == 0:
            _record_history(
                history,
                iteration,
                weights,
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

        momentum_step = beta * (weights - previous_weights)
        next_weights = weights - alpha * grad + momentum_step

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
        history=history,
    )


def nesterov_accelerated_gradient(
    q,
    c,
    w0,
    mu,
    l_smooth,
    tol=1e-6,
    max_iter=10000,
    objective_fn=None,
    reference_weights=None,
    record_every=1,
):
    """Run Nesterov Accelerated Gradient for strongly convex functions."""

    _validate_spectral_bounds(mu, l_smooth)

    alpha = 1.0 / l_smooth
    beta = (np.sqrt(l_smooth) - np.sqrt(mu)) / (
        np.sqrt(l_smooth) + np.sqrt(mu)
    )

    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    final_evaluation_point = weights.copy()
    history = _new_history()

    start = perf_counter()
    converged = False
    final_grad_norm = np.inf
    iterations = 0

    for iteration in range(max_iter + 1):
        # First take the look-ahead step, then evaluate the gradient there.
        evaluation_point = weights + beta * (weights - previous_weights)
        grad = evaluation_point @ q - c
        final_grad_norm = frobenius_norm(grad)
        final_evaluation_point = evaluation_point

        if iteration % record_every == 0:
            _record_history(
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
        history=history,
    )


def _as_2d_rhs(rhs):
    """Make a vector look like a matrix with one column."""

    rhs_array = np.asarray(rhs, dtype=float)
    was_vector = rhs_array.ndim == 1

    if was_vector:
        rhs_array = rhs_array[:, None]

    if rhs_array.ndim != 2:
        raise ValueError("rhs must be a vector or a matrix.")

    return rhs_array, was_vector


def _validate_spectral_bounds(mu, l_smooth):
    if mu <= 0.0:
        raise ValueError("mu must be strictly positive.")
    if l_smooth <= 0.0:
        raise ValueError("l_smooth must be strictly positive.")
    if l_smooth < mu:
        raise ValueError("l_smooth must be greater than or equal to mu.")


def _new_history():
    history = {
        "iteration": [],
        "grad_norm": [],
        "objective": [],
        "relative_error": [],
    }
    return history


def _record_history(
    history,
    iteration,
    weights,
    grad_norm,
    objective_fn,
    reference_weights,
):
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
