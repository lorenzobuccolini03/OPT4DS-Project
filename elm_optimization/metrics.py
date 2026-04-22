"""Metrics and objective evaluations for the ELM experiments."""

from __future__ import annotations

import numpy as np


def objective_value(
    weights: np.ndarray,
    h_aug: np.ndarray,
    y: np.ndarray,
    lambda_reg: float,
) -> float:
    """Evaluate the regularized ELM objective from the theory."""

    residual = weights @ h_aug - y
    n_samples = h_aug.shape[1]
    loss = 0.5 * np.sum(residual * residual) / n_samples
    penalty = 0.5 * lambda_reg * np.sum(weights * weights)
    return float(loss + penalty)


def gradient(weights: np.ndarray, q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Evaluate grad f(W) = W Q - C."""

    return weights @ q - c


def frobenius_norm(matrix: np.ndarray) -> float:
    """Compute the Frobenius norm explicitly."""

    return float(np.sqrt(np.sum(matrix * matrix)))


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return ||candidate-reference||_F / max(1, ||reference||_F)."""

    numerator = frobenius_norm(candidate - reference)
    denominator = max(1.0, frobenius_norm(reference))
    return numerator / denominator


def mean_squared_error(scores: np.ndarray, y: np.ndarray) -> float:
    """Mean squared prediction error over all output entries."""

    residual = scores - y
    return float(np.mean(residual * residual))


def classification_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Classification accuracy for one-hot style targets."""

    predicted = np.argmax(scores, axis=0)
    return float(np.mean(predicted == labels))
