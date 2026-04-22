"""ELM problem construction.

This module implements the mathematical formulation from the theoretical
section of the report:

    min_W 1/(2N) ||W H - Y||_F^2 + lambda/2 ||W||_F^2

where H is the augmented hidden-layer matrix containing a final row of ones
for the output bias.  The hidden layer is randomly generated once and kept
fixed, so the optimization variable is only the output matrix W.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActivationName = Literal["tanh", "sigmoid", "relu", "linear"]


@dataclass(frozen=True)
class ELMInstance:
    """Container for one fixed ELM optimization problem.

    All algorithms must solve exactly the same instance: same data, same
    hidden layer, same augmented H, same regularization parameter, and
    therefore same Q and C matrices.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    train_labels: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_labels: np.ndarray
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    h_train_aug: np.ndarray
    h_test_aug: np.ndarray
    q: np.ndarray
    c: np.ndarray
    lambda_reg: float
    activation: ActivationName

    @property
    def n_train(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_test(self) -> int:
        return self.x_test.shape[1]

    @property
    def n_features(self) -> int:
        return self.x_train.shape[0]

    @property
    def n_classes(self) -> int:
        return self.y_train.shape[0]

    @property
    def hidden_width(self) -> int:
        return self.hidden_weights.shape[0]

    @property
    def n_variables_per_output(self) -> int:
        """Number of columns of W, equal to hidden_width + output bias."""

        return self.h_train_aug.shape[0]


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Return one-hot targets with shape (n_classes, n_samples)."""

    encoded = np.zeros((n_classes, labels.size), dtype=float)
    encoded[labels, np.arange(labels.size)] = 1.0
    return encoded


def standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features using training statistics only.

    The matrices follow the report notation: columns are samples and rows are
    features.  This keeps data preparation deterministic and avoids any
    dependency on machine-learning helper libraries.
    """

    mean = x_train.mean(axis=1, keepdims=True)
    std = x_train.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def generate_gaussian_classification_data(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    class_sep: float,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reproducible multi-class dataset.

    The data are generated on the fly, which is allowed by the guidelines.
    The separation and noise parameters make it possible to create realistic
    instances with nontrivial classification behavior while keeping every
    experiment fully reproducible.
    """

    rng = np.random.default_rng(seed)
    centers = class_sep * rng.normal(size=(n_features, n_classes))

    train_labels = rng.integers(0, n_classes, size=n_train)
    test_labels = rng.integers(0, n_classes, size=n_test)

    x_train = centers[:, train_labels] + noise * rng.normal(
        size=(n_features, n_train)
    )
    x_test = centers[:, test_labels] + noise * rng.normal(size=(n_features, n_test))

    x_train, x_test = standardize_train_test(x_train, x_test)
    return x_train, train_labels, x_test, test_labels


def activation_function(z: np.ndarray, activation: ActivationName) -> np.ndarray:
    """Apply the fixed hidden-layer activation elementwise."""

    if activation == "tanh":
        return np.tanh(z)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "linear":
        return z
    raise ValueError(f"Unsupported activation: {activation}")


def build_hidden_matrix(
    x: np.ndarray,
    hidden_weights: np.ndarray,
    hidden_bias: np.ndarray,
    activation: ActivationName,
) -> np.ndarray:
    """Compute H = sigma(W1 X + b1 1^T)."""

    affine = hidden_weights @ x + hidden_bias[:, None]
    return activation_function(affine, activation)


def augment_hidden_matrix(h: np.ndarray) -> np.ndarray:
    """Append the bias row of ones, producing the augmented H matrix."""

    ones = np.ones((1, h.shape[1]), dtype=h.dtype)
    return np.vstack([h, ones])


def formulate_elm_system(
    h_aug: np.ndarray, y: np.ndarray, lambda_reg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build the matrices Q and C from the report.

    Q = (H H^T) / N + lambda I
    C = (Y H^T) / N

    The optimum satisfies W* Q = C.  Equivalently, Q (W*)^T = C^T,
    which is the symmetric positive definite system solved by LDLT.
    """

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be strictly positive for Q to be SPD.")
    n_samples = h_aug.shape[1]
    n_variables = h_aug.shape[0]
    q = (h_aug @ h_aug.T) / n_samples
    q = q + lambda_reg * np.eye(n_variables)
    c = (y @ h_aug.T) / n_samples
    return q, c


def create_elm_classification_instance(
    *,
    n_train: int = 1000,
    n_test: int = 300,
    n_features: int = 20,
    n_classes: int = 3,
    hidden_width: int = 100,
    lambda_reg: float = 1e-3,
    activation: ActivationName = "tanh",
    class_sep: float = 2.0,
    noise: float = 1.0,
    hidden_scale: float = 1.0,
    seed: int = 0,
) -> ELMInstance:
    """Create a complete fixed ELM optimization instance.

    The random hidden layer is created once, then frozen.  This matches the
    ELM assumption in the theory: the optimization algorithms only train the
    output-layer matrix W.
    """

    x_train, train_labels, x_test, test_labels = generate_gaussian_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        n_classes=n_classes,
        class_sep=class_sep,
        noise=noise,
        seed=seed,
    )

    y_train = one_hot(train_labels, n_classes)
    y_test = one_hot(test_labels, n_classes)

    rng = np.random.default_rng(seed + 10_000)
    # The 1/sqrt(d) scaling keeps hidden activations in a numerically useful
    # range as the input dimension changes.
    hidden_weights = (
        hidden_scale
        * rng.normal(size=(hidden_width, n_features))
        / np.sqrt(float(n_features))
    )
    hidden_bias = rng.uniform(-1.0, 1.0, size=hidden_width)

    h_train = build_hidden_matrix(x_train, hidden_weights, hidden_bias, activation)
    h_test = build_hidden_matrix(x_test, hidden_weights, hidden_bias, activation)
    h_train_aug = augment_hidden_matrix(h_train)
    h_test_aug = augment_hidden_matrix(h_test)
    q, c = formulate_elm_system(h_train_aug, y_train, lambda_reg)

    return ELMInstance(
        x_train=x_train,
        y_train=y_train,
        train_labels=train_labels,
        x_test=x_test,
        y_test=y_test,
        test_labels=test_labels,
        hidden_weights=hidden_weights,
        hidden_bias=hidden_bias,
        h_train_aug=h_train_aug,
        h_test_aug=h_test_aug,
        q=q,
        c=c,
        lambda_reg=lambda_reg,
        activation=activation,
    )


def predict_scores(weights: np.ndarray, h_aug: np.ndarray) -> np.ndarray:
    """Compute class scores or regression outputs W H."""

    return weights @ h_aug
