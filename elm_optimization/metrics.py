"""Metric functions for the ELM experiments."""

import numpy as np

# compute the regularized ELM objective value, gradient, and various error metrics
def objective_value(weights, h_aug, y, lambda_reg):
    """Compute the regularized ELM objective."""

    residual = weights @ h_aug - y
    n_samples = h_aug.shape[1]

    loss = 0.5 * np.sum(residual * residual) / n_samples
    penalty = 0.5 * lambda_reg * np.sum(weights * weights)
    value = loss + penalty

    return float(value)

# compute the gradient of the regularized ELM objective with respect to the weights
def gradient(weights, q, c):
    """Compute grad f(W) = W Q - C."""

    return weights @ q - c

# compute the Frobenius norm of a matrix
def frobenius_norm(matrix):
    """Compute the Frobenius norm."""

    squared_norm = np.sum(matrix * matrix)
    return float(np.sqrt(squared_norm))

# compute the relative Frobenius error between a candidate solution and a reference solution
def relative_error(candidate, reference):
    """Compute relative Frobenius error."""

    numerator = frobenius_norm(candidate - reference)
    denominator = max(1.0, frobenius_norm(reference))
    return numerator / denominator

# compute the mean squared error between predicted scores and true labels
def mean_squared_error(scores, y):
    """Mean squared prediction error."""

    residual = scores - y
    return float(np.mean(residual * residual))

# compute the classification accuracy for one-hot targets
def classification_accuracy(scores, labels):
    """Classification accuracy for one-hot targets."""

    predicted_labels = np.argmax(scores, axis=0)
    accuracy = np.mean(predicted_labels == labels)
    return float(accuracy)
