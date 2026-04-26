"""Build the Extreme Learning Machine (ELM) optimization problem.

After the hidden layer is fixed, training the ELM means solving

    min_W 1/(2N) ||W H - Y||_F^2 + lambda/2 ||W||_F^2

Here H is the augmented hidden-layer matrix. Its last row is made of ones,
so the output bias is included in the matrix W.
"""

import numpy as np


class ELMInstance:
    """Simple object containing one fixed ELM problem."""

    def __init__(
        self,
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
        q,
        c,
        lambda_reg,
        activation,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.train_labels = train_labels
        self.x_test = x_test
        self.y_test = y_test
        self.test_labels = test_labels
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.h_train_aug = h_train_aug
        self.h_test_aug = h_test_aug
        self.q = q
        self.c = c
        self.lambda_reg = lambda_reg
        self.activation = activation

        # These values are useful in the tables produced by the experiments.
        self.n_train = x_train.shape[1]
        self.n_test = x_test.shape[1]
        self.n_features = x_train.shape[0]
        self.n_classes = y_train.shape[0]
        self.hidden_width = hidden_weights.shape[0]
        self.n_variables_per_output = h_train_aug.shape[0]


def one_hot(labels, n_classes):
    """Convert integer labels into a one-hot matrix."""

    encoded = np.zeros((n_classes, labels.size), dtype=float)
    encoded[labels, np.arange(labels.size)] = 1.0
    return encoded


def standardize_train_test(x_train, x_test):
    """Standardize the data using only training-set statistics."""

    mean = x_train.mean(axis=1, keepdims=True)
    std = x_train.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)

    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std
    return x_train_scaled, x_test_scaled


def generate_gaussian_classification_data(
    n_train,
    n_test,
    n_features,
    n_classes,
    class_sep,
    noise,
    seed,
):
    """Generate a reproducible synthetic classification dataset."""

    rng = np.random.default_rng(seed)

    centers = class_sep * rng.normal(size=(n_features, n_classes))
    train_labels = rng.integers(0, n_classes, size=n_train)
    test_labels = rng.integers(0, n_classes, size=n_test)

    train_noise = noise * rng.normal(size=(n_features, n_train))
    test_noise = noise * rng.normal(size=(n_features, n_test))
    x_train = centers[:, train_labels] + train_noise
    x_test = centers[:, test_labels] + test_noise

    x_train, x_test = standardize_train_test(x_train, x_test)
    return x_train, train_labels, x_test, test_labels


def generate_correlated_classification_data(
    n_train,
    n_test,
    n_features,
    n_classes,
    class_sep,
    noise,
    correlation_strength,
    feature_scales,
    seed,
):
    """Generate synthetic data with controlled feature scales and correlation.

    This function is used only to create more interesting ELM test cases.
    The optimization problem is still the same ELM problem built later from H,
    Q, and C.
    """

    rng = np.random.default_rng(seed)
    rho = min(max(correlation_strength, 0.0), 0.99)

    if feature_scales is None:
        scales = np.ones(n_features)
    else:
        scales = np.asarray(feature_scales, dtype=float)
        if scales.size != n_features:
            raise ValueError("feature_scales must have one value per feature.")

    centers = class_sep * rng.normal(size=(n_features, n_classes))
    centers = centers * scales[:, None]

    train_labels = rng.integers(0, n_classes, size=n_train)
    test_labels = rng.integers(0, n_classes, size=n_test)

    direction = rng.normal(size=(n_features, 1))
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        direction[0, 0] = 1.0
    else:
        direction = direction / direction_norm

    x_train = _sample_correlated_points(
        rng,
        centers,
        train_labels,
        noise,
        rho,
        scales,
        direction,
    )
    x_test = _sample_correlated_points(
        rng,
        centers,
        test_labels,
        noise,
        rho,
        scales,
        direction,
    )

    x_train, x_test = standardize_train_test(x_train, x_test)
    return x_train, train_labels, x_test, test_labels


def _sample_correlated_points(
    rng,
    centers,
    labels,
    noise,
    rho,
    scales,
    direction,
):
    """Sample points around class centers with an optional common direction."""

    n_features = centers.shape[0]
    n_samples = labels.size

    independent_noise = rng.normal(size=(n_features, n_samples))
    shared_values = rng.normal(size=(1, n_samples))
    shared_noise = direction @ shared_values

    mixed_noise = np.sqrt(1.0 - rho) * independent_noise
    mixed_noise = mixed_noise + np.sqrt(rho) * shared_noise
    mixed_noise = noise * mixed_noise * scales[:, None]

    x = centers[:, labels] + mixed_noise
    return x


def apply_sparse_feature_mask(x_train, x_test, zero_probability, seed):
    """Set a fixed random part of the feature matrix to zero.

    We keep ordinary dense NumPy arrays. The point of the test is only to see
    how the algorithms behave when the ELM hidden layer receives less dense
    information.
    """

    rng = np.random.default_rng(seed)
    keep_probability = 1.0 - zero_probability

    train_mask = rng.random(size=x_train.shape) < keep_probability
    test_mask = rng.random(size=x_test.shape) < keep_probability

    sparse_x_train = x_train * train_mask
    sparse_x_test = x_test * test_mask
    sparse_x_train, sparse_x_test = standardize_train_test(
        sparse_x_train,
        sparse_x_test,
    )

    return sparse_x_train, sparse_x_test


def activation_function(z, activation):
    """Apply the chosen hidden-layer activation function."""

    if activation == "tanh":
        return np.tanh(z)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "linear":
        return z

    raise ValueError("Unsupported activation: " + str(activation))


def build_hidden_matrix(x, hidden_weights, hidden_bias, activation):
    """Compute H = sigma(W1 X + b1 1^T)."""

    affine_part = hidden_weights @ x + hidden_bias[:, None]
    h = activation_function(affine_part, activation)
    return h


def augment_hidden_matrix(h):
    """Add the final row of ones used for the output bias."""

    ones = np.ones((1, h.shape[1]), dtype=h.dtype)
    h_aug = np.vstack([h, ones])
    return h_aug


def formulate_elm_system(h_aug, y, lambda_reg):
    """Build Q and C for the optimality equation W Q = C.

    Q = H H^T / N + lambda I
    C = Y H^T / N

    Since lambda is positive, Q is symmetric positive definite.
    """

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be strictly positive.")

    n_samples = h_aug.shape[1]
    n_variables = h_aug.shape[0]

    q = (h_aug @ h_aug.T) / n_samples
    q = q + lambda_reg * np.eye(n_variables)
    c = (y @ h_aug.T) / n_samples
    return q, c


def create_elm_classification_instance(
    n_train=1000,
    n_test=300,
    n_features=20,
    n_classes=3,
    hidden_width=100,
    lambda_reg=1e-3,
    activation="tanh",
    class_sep=2.0,
    noise=1.0,
    hidden_scale=1.0,
    seed=0,
):
    """Create data, random hidden layer, and the matrices Q and C."""

    data = generate_gaussian_classification_data(
        n_train,
        n_test,
        n_features,
        n_classes,
        class_sep,
        noise,
        seed,
    )
    x_train, train_labels, x_test, test_labels = data

    return create_elm_instance_from_arrays(
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


def create_elm_instance_from_arrays(
    x_train,
    train_labels,
    x_test,
    test_labels,
    hidden_width=100,
    lambda_reg=1e-3,
    activation="tanh",
    hidden_scale=1.0,
    seed=0,
    standardize_data=True,
):
    """Build the fixed ELM optimization problem from given data arrays."""

    if standardize_data:
        x_train, x_test = standardize_train_test(x_train, x_test)

    n_features = x_train.shape[0]
    n_classes = int(max(np.max(train_labels), np.max(test_labels))) + 1
    y_train = one_hot(train_labels, n_classes)
    y_test = one_hot(test_labels, n_classes)

    rng = np.random.default_rng(seed + 10000)

    # The factor 1/sqrt(d) keeps the hidden activations in a stable range.
    hidden_weights = rng.normal(size=(hidden_width, n_features))
    hidden_weights = hidden_scale * hidden_weights / np.sqrt(float(n_features))
    hidden_bias = rng.uniform(-1.0, 1.0, size=hidden_width)

    h_train = build_hidden_matrix(x_train, hidden_weights, hidden_bias, activation)
    h_test = build_hidden_matrix(x_test, hidden_weights, hidden_bias, activation)

    h_train_aug = augment_hidden_matrix(h_train)
    h_test_aug = augment_hidden_matrix(h_test)
    q, c = formulate_elm_system(h_train_aug, y_train, lambda_reg)

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
        q,
        c,
        lambda_reg,
        activation,
    )


def predict_scores(weights, h_aug):
    """Compute predictions W H."""

    scores = weights @ h_aug
    return scores
