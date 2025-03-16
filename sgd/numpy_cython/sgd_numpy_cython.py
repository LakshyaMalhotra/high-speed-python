import time

import numpy as np
from sklearn.datasets import make_regression

import cython_np_predict, cython_np_grad, cython_np_normalize  # type:ignore


def main(num_epochs: int):
    """AOT compilation with Cython version 1. This version only compiles the
    core functions ahead of time. The format of ``main`` function is the same.
    """
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
    features, targets, _ = data
    targets = targets[:, np.newaxis]
    normalized_features = cython_np_normalize.normalize(features)

    # initialize weights
    w = np.random.rand(num_features, 1)
    b = np.random.rand()
    lr = 1e-2

    for epoch in range(num_epochs):
        grad_w, grad_b = cython_np_grad.calculate_gradients(
            normalized_features, targets, w, b
        )
        w -= lr * grad_w / len(normalized_features)
        b -= lr * grad_b / len(normalized_features)

        # Optionally print the progress
        if epoch % 100 == 0:
            loss = np.mean(
                (cython_np_predict.predict(normalized_features, w, b) - targets)
                ** 2
            )
            print(f"Epoch {epoch}: Loss = {loss:.5f}")

    print(f"Time taken: {(time.perf_counter() - start):.4f} seconds")


if __name__ == "__main__":
    main(num_epochs=1000)
