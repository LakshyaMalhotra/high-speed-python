import time

import numpy as np
from sklearn.datasets import make_regression


def predict(
    features: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """Compute vectorized ``W * X + b``."""
    return features @ weights + bias


def calculate_gradients(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray, b: float
) -> tuple[np.ndarray, ...]:
    """Vectorized gradient calculation of loss (mean squared error) with respect
    to weights and bias."""
    error = predict(features, weights, b) - targets
    grad_w = 2 * features.T @ error
    grad_b = 2 * np.sum(error)
    return grad_w, grad_b


def normalize(features: np.ndarray) -> np.ndarray:
    """Normalize the features with min 0 and max 1."""
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    return (features - min_vals) / (max_vals - min_vals + 1e-8)


def main(num_epochs: int):
    start = time.perf_counter()
    num_features = 100
    data = make_regression(
        n_samples=10000,
        n_features=num_features,
        bias=1.0,
        random_state=2024,
        coef=True,
        noise=1.0,
    )
    features, targets, coefs = data
    targets = targets[:, np.newaxis]
    normalized_features = normalize(features)

    # initialize weights
    w = np.random.rand(num_features, 1)
    b = np.random.rand()
    lr = 1e-2

    for epoch in range(num_epochs):
        grad_w, grad_b = calculate_gradients(normalized_features, targets, w, b)
        w -= lr * grad_w / len(normalized_features)
        b -= lr * grad_b / len(normalized_features)

        # Optionally print the progress
        if epoch % 100 == 0:
            loss = np.mean((predict(normalized_features, w, b) - targets) ** 2)
            print(f"Epoch {epoch}: Loss = {loss:.5f}")

    print(f"Time taken: {(time.perf_counter() - start):.4f} seconds")


if __name__ == "__main__":
    main(num_epochs=1000)
