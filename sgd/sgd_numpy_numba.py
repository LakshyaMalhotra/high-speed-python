import time

import numpy as np
from numba import njit
from sklearn.datasets import make_regression


@njit
def predict(features: np.ndarray, weights: np.ndarray, b: float) -> np.ndarray:
    """Compute vectorized ``W * X + b``."""
    return features @ weights + b


@njit
def calculate_gradients(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, ...]:
    """Vectorized gradient calculation of loss (mean squared error) with respect
    to weights and bias."""
    error = predict(features, weights, bias) - targets
    grad_w = 2 * features.T @ error
    grad_b = 2 * np.sum(error)
    return grad_w, grad_b


def normalize(features):
    """Normalize the features with min 0 and max 1."""
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    return (features - min_vals) / (max_vals - min_vals + 1e-8)


def train(
    features: np.ndarray,
    targets: np.ndarray,
    num_epochs: int,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
) -> None:
    """Helper function for training."""
    for epoch in range(num_epochs):
        grad_w, grad_b = calculate_gradients(features, targets, weights, bias)
        weights -= lr * grad_w
        bias -= lr * grad_b

        # Optionally print the progress
        if epoch % 100 == 0:
            loss = np.mean((predict(features, weights, bias) - targets) ** 2)
            print(
                "Epoch:",
                epoch,
                "Loss =",
                loss,
            )


def main(num_epochs: int):
    start = time.perf_counter()

    num_features = 100
    data = make_regression(
        n_samples=1000,
        n_features=num_features,
        bias=1.0,
        random_state=2024,
        coef=True,
        noise=1.0,
    )

    features, targets, _ = data
    targets = targets[:, np.newaxis]
    normalized_features = normalize(features)

    # initialize weights
    w = np.random.rand(num_features, 1)
    b = np.random.rand()
    lr = 1e-5

    train(normalized_features, targets, num_epochs, w, b, lr)

    print("Time taken:", (time.perf_counter() - start), "seconds")


if __name__ == "__main__":
    main(num_epochs=1000)
