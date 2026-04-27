# import time for timing the algorithms
from time import perf_counter

# Import numpy for numerical computations
import numpy as np

# Import the frobenius_norm function from the metrics module in the same package
from .metrics import frobenius_norm

################################################################################
# 0) ALGORITHMS FOR ELM OPTIMIZATION
# 1. SpectralBounds used to hold the spectral constants
# 2. OptimizationResult used to hold the output of the optimization methods

# 1) LDLT factorization and solve functions:
# 1. ldlt_factorize to compute the LDLT factorization of a symmetric positive definite matrix
# 2. forward_substitution_unit_lower to solve L X = rhs where L has ones on the diagonal
# 3. diagonal_solve to solve diag(d) X = rhs
# 4. backward_substitution_unit_upper_from_lower to solve L^T X = rhs using the stored lower triangular matrix L
# 5. solve_with_ldlt to solve Q X = rhs after Q = L diag(d) L^T has been factorized
# 6. ldlt_solve_weights to solve Q W.T = C.T with the scratch LDLT factorization

# 2) Spectral bounds estimation for the first-order methods:
# 1. power_method_largest_eigenvalue to approximate the largest eigenvalue of Q with the Power Method
# 2. estimate_spectral_bounds to estimate mu and L for the first-order method

# 3) First-order optimization algorithms for the ELM quadratic objective:
# 1. heavy_ball to run the Heavy Ball method on the ELM quadratic objective
# 2. nesterov_accelerated_gradient to run Nesterov Accelerated Gradient for strongly convex functions

# 3) Helper functions for the algorithms:
# 1. _as_2d_arr to make a vector look like a matrix with one column
# 2. _validate_spectral_bounds to check the spectral bounds before running the first-order
# 3. _new_history to create a new history dictionary for recording optimization history
# 4. _record_history to record the optimization history for plotting and analysis
################################################################################

####################################
# 0) ALGORITHMS FOR ELM OPTIMIZATION
####################################
# 1. SpectralBounds used to hold the spectral constants
class SpectralBounds:
    """Small container for the spectral constants used by HB and Nesterov."""

    def __init__(
        self,
        mu, # the strong convexity constant (minimum eigenvalue)
        l_smooth, # l-smooth constant (maximum eigenvalue)
        condition_estimate, # the condition number estimate (L / mu)
        power_iterations, # the number of iterations taken by the Power Method to estimate the largest eigenvalue
        raw_largest_eigenvalue_estimate, # the raw largest eigenvalue estimate from the Power Method BEFORE applying the safety factor
    ):
        self.mu = mu
        self.l_smooth = l_smooth
        self.condition_estimate = condition_estimate
        self.power_iterations = power_iterations
        self.raw_largest_eigenvalue_estimate = raw_largest_eigenvalue_estimate

# 2. OptimizationResult used to hold the output of the optimization methods
class OptimizationResult:
    """Small container for the output of an optimization method."""

    def __init__(
        self,
        method, # optimization method name (e.g., "Heavy Ball", "Nesterov", "LDLT")
        weights, # the final weights found by the optimization method (W_2*)
        iterations, # the number of iterations taken by the optimization method (0 for LDLT)
        converged, # whether the optimization method converged within the maximum iterations
        elapsed_seconds, # the time taken by the optimization method
        final_gradient_norm, # the norm of the final gradient
        alpha=None, # the step size for the optimization method
        beta=None, # the momentum parameter for the optimization method
        history=None, # the optimization history for plotting and analysis as a dictionary
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

###########################################
# 1) LDLT factorization and solve functions
###########################################
# 1. ldlt_factorize to compute the LDLT factorization of a symmetric positive definite matrix
def ldlt_factorize(q, pivot_tol=1e-14):
    """Compute Q = L diag(d) L^T for a symmetric positive definite matrix."""

    # 3 checks on the input matrix Q:
    # 1. Convert to a numpy array of type float.
    q = np.asarray(q, dtype=float)

    # 2. Check that Q is square and symmetric.
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix.")
    if not np.allclose(q, q.T, rtol=1e-10, atol=1e-12): # Q=Q.T within numerical tolerance
        raise ValueError("q must be symmetric.")

    # 3. Check that Q is positive definite by ensuring that the pivots are positive during the factorization.
    n = q.shape[0]
    l_factor = np.eye(n, dtype=float) # I matrix nxn
    d = np.zeros(n, dtype=float) # D vector n-th dimensional

    for j in range(n):
        # Compute v_k = L_jk d_k for k < j (pseudocode: line 6)
        if j == 0:
            v = np.array([], dtype=float) # empty vector for the first column since there are no previous columns
            diagonal_correction = 0.0 
        else:
            v = l_factor[j, :j] * d[:j] # L_jk * d_k to get the contributions from the previous columns
            diagonal_correction = float(l_factor[j, :j] @ v) # L_jk * v_k

        # d_jj = q_jj - sum_{k<j} L_jk^2 d_k (pseudocode: line 9)
        pivot = float(q[j, j] - diagonal_correction)
        # check that the pivot is positive to ensure 1/djj is defined
        if pivot <= pivot_tol:
            message = (
                "LDLT pivot is not positive at column "
                + str(j)
                + ": pivot="
                + str(pivot)
            )
            raise np.linalg.LinAlgError(message)
        d[j] = pivot # d_jj

        # Formula for the entries below the diagonal (pseudocode: lines 11-12):
        # L_ij = (q_ij - sum_{k<j} L_ik v_k) / d_j
        for i in range(j + 1, n):
            if j == 0:
                correction = 0.0
            else:
                correction = float(l_factor[i, :j] @ v) # L_ik * v_k
            l_factor[i, j] = (q[i, j] - correction) / d[j]

    return l_factor, d # L and D (such that Q = L * diag(d) * L^T)

# 2. forward_substitution_unit_lower to solve L*X = rhs where L has ones on the diagonal
def forward_substitution_unit_lower(l_factor, c_transposed):
    """Solve L X = rhs where L has ones on the diagonal."""

    # Make sure c_transposed is a matrix (2D array) and remember if it was originally a vector.
    c_transposed_2d, was_vector = _as_2d_arr(c_transposed)
    n = l_factor.shape[0]
    z = np.zeros_like(c_transposed_2d, dtype=float) # z is a L*m matrix where m is the number of columns in c_transposed

    for i in range(n):
        # backsolve: Lz = c^T where L is lower triangular with ones on the diagonal
        # z_i = c_i - sum_{k < i} L_ik * z_k
        previous_terms = l_factor[i, :i] @ z[:i, :] 
        z[i, :] = c_transposed_2d[i, :] - previous_terms  

    if was_vector:
        return z[:, 0]
    return z

# 3. diagonal_solve to solve diag(d) V = z
def diagonal_solve(d, z):
    """Solve diag(d) V = Z."""

    # Make sure z is a matrix (2D array) and remember if it was originally a vector.
    z_2d, was_vector = _as_2d_arr(z) 

    # Solve diag(d) * V = Z by elementwise division since diag(d) is a diagonal matrix.
    V = z_2d / d[:, None] 

    if was_vector:
        return V[:, 0]
    return V

# 4. backward_substitution_unit_upper_from_lower to solve L^T w_2 = V using the stored lower triangular matrix L.
def backward_substitution_unit_upper_from_lower(l_factor, V):
    """Solve L^T W_2*^T = V using the stored lower triangular matrix L."""

    rhs_2d, was_vector = _as_2d_arr(V)
    n = l_factor.shape[0]
    w_min = np.zeros_like(rhs_2d, dtype=float)

    # Now L is upper triangular with ones on the diagonal, so we can backsolve starting from the last row.
    for i in range(n - 1, -1, -1):
        next_terms = l_factor[i + 1 :, i] @ w_min[i + 1 :, :]
        w_min[i, :] = rhs_2d[i, :] - next_terms

    if was_vector:
        return w_min[:, 0]
    return w_min

# 5. solve_with_ldlt to solve Q Z = c_transposed after Q = L diag(d) L^T has been factorized.
def solve_with_ldlt(l_factor, d, c_transposed):
    """Solve Q Z = c_transposed after Q = L diag(d) L^T has been factorized."""

    # Solve L Z = C^T (forward substitution)
    z = forward_substitution_unit_lower(l_factor, c_transposed)
    # Solve diag(d) V = Z (diagonal solve)
    v = diagonal_solve(d, z)
    # Solve L^T (W_2*)^T = V (backward substitution using the stored lower triangular matrix L)
    w_min = backward_substitution_unit_upper_from_lower(l_factor, v)
    return w_min

# 6. ldlt_solve_weights to solve Q W^T = C^T with the scratch LDLT factorization.
def ldlt_solve_weights(q, c, pivot_tol=1e-14):
    """Solve Q W^T = C^T with the LDLT factorization."""

    start = perf_counter() # start timing the LDLT solve

    # 1: LDLT factorization of Q: Q = L diag(d) L^T
    l_factor, d = ldlt_factorize(q, pivot_tol)
    # 2: Solve Q W^T = C^T
    solution_transposed = solve_with_ldlt(l_factor, d, c.T) # (W_2*)^T is returned
    weights = solution_transposed.T # transpose back to get W_2*

    elapsed = perf_counter() - start
    final_grad = weights @ q - c # compute the final gradient to report its norm in the result (should be close to zero)

    return OptimizationResult(
        method="LDLT",
        weights=weights,
        iterations=0,
        converged=True,
        elapsed_seconds=elapsed,
        final_gradient_norm=frobenius_norm(final_grad),
    )

##################
# 2) POWER METHOD
##################

# 1.1 power_method_largest_eigenvalue to approximate the largest eigenvalue of Q with the Power Method.
def power_method_largest_eigenvalue(q, tol=1e-8, max_iter=5000, seed=0):
    """Approximate the largest eigenvalue of Q with the Power Method."""

    
    rng = np.random.default_rng(seed)
    n = q.shape[0] # v should have the same dimension as the number of rows/columns of Q

    v = rng.normal(size=n)

    # Normalize the initial vector to have unit norm to improve convergence.
    v_norm = np.linalg.norm(v) # norm of v
    if v_norm == 0.0:
        v[0] = 1.0
    else:
        v = v / v_norm

    previous_rayleigh = 0.0 # the Rayleigh quotient from the previous iteration, initialized to zero for the first iteration
    rayleigh = 0.0 # the current Rayleigh quotient, initialized to zero for the first iteration

    # Power Method iterations
    for iteration in range(1, max_iter + 1):
        qv = q @ v
        qv_norm = np.linalg.norm(qv)
        if qv_norm == 0.0:
            raise np.linalg.LinAlgError("Power method produced a zero vector.")

        v = qv / qv_norm

        # Rayleigh quotient: v^T * Q * v
        qv = q @ v
        rayleigh = float(v @ qv) # v is not transposed since it is a 1D array (not a 2D array with one column)

        difference = abs(rayleigh - previous_rayleigh) # change of rayleigh quotient from the previous iteration
        tolerance_value = tol * max(1.0, abs(rayleigh))

        # if the change is smaller than the tolerance, we consider that we have converged to the largest eigenvalue 
        if difference <= tolerance_value:
            return rayleigh, iteration # converged to the largest eigenvalue within the specified tolerance

        previous_rayleigh = rayleigh

    return rayleigh, max_iter

# 1.2 estimate_spectral_bounds to estimate mu and L for the first-order methods
def estimate_spectral_bounds(
    q,
    lambda_reg,
    power_tol=1e-8,
    power_max_iter=5000,
    seed=0,
    l_safety_factor=1.01,
):
    """Estimate mu and L for the first-order methods.

    For this ELM problem, Q = H H^T / N + lambda I, so lambda is a safe lower
    bound for the smallest eigenvalue. The largest eigenvalue is estimated by
    the Power Method.
    """

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be strictly positive.")

    # compute the (approximated) largest eigenvalues
    raw_l, power_iterations = power_method_largest_eigenvalue(
        q,
        tol=power_tol,
        max_iter=power_max_iter,
        seed=seed,
    )

    mu = lambda_reg
    # safety factor is applied in order to ensure that the estimated L is not smaller than the true largest eigenvalue
    l_smooth = max(raw_l, lambda_reg) * l_safety_factor 
    condition_estimate = l_smooth / mu

    return SpectralBounds(
        mu, 
        l_smooth,
        condition_estimate,
        power_iterations,
        raw_l,
    )



#########################################################################
# 3) First-order optimization algorithms for the ELM quadratic objective
#########################################################################

####################
# HEAVY BALL METHOD
####################

def heavy_ball(
    q,
    c,
    w0,
    mu,
    l_smooth,
    tol=1e-6,
    max_iter=10000,
    objective_fn=None,
    reference_weights=None, # real W_2* obtained by LDLT or through a built-in function
    record_every=1, # the frequency of recording the optimization history (e.g., every 10 iterations, every 100 iterations, etc.)
):
    """Run the Heavy Ball method on the ELM quadratic objective."""

    _validate_spectral_bounds(mu, l_smooth)

    # sqrt(L) and sqrt(mu) 
    sqrt_l = np.sqrt(l_smooth)
    sqrt_mu = np.sqrt(mu)

    # stepsize computation
    alpha = 4.0 / (sqrt_l + sqrt_mu) ** 2
    beta = ((sqrt_l - sqrt_mu) / (sqrt_l + sqrt_mu)) ** 2
    
    weights = np.array(w0, dtype=float, copy=True)
    previous_weights = weights.copy()
    history = _new_history()

    start = perf_counter() # start timing
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

##############################################################
# Nesterov Accelerated Gradient for strongly convex functions
##############################################################

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

    start = perf_counter() # start timing
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


#########################################
# 4) Helper functions for the algorithms
#########################################

# 1. This function is used to allow the forward/backward/diagonal solves to accept either vectors or matrices as the right-hand side.
def _as_2d_arr(vec):
    """Make a vector look like a matrix with one column."""

    vec_array = np.asarray(vec, dtype=float)
    was_vector = vec_array.ndim == 1

    if was_vector:
        vec_array = vec_array[:, None]

    if vec_array.ndim != 2:
        raise ValueError("input must be a vector or a matrix.")

    return vec_array, was_vector

# 2. This function is used to check the spectral bounds before running the first-order methods.
def _validate_spectral_bounds(mu, l_smooth):
    if mu <= 0.0:
        raise ValueError("mu must be strictly positive.")
    if l_smooth <= 0.0:
        raise ValueError("l_smooth must be strictly positive.")
    if l_smooth < mu:
        raise ValueError("l_smooth must be greater than or equal to mu.")

# 3. These functions are used to record the optimization history for plotting and analysis.
def _new_history():
    history = {
        "iteration": [],
        "grad_norm": [],
        "objective": [],
        "relative_error": [],
    }
    return history

# 4. This function is used to record the optimization history for plotting and analysis.
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
