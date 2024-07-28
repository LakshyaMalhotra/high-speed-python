import time
import random

from sklearn.datasets import make_regression


def predict(features: list, weights: list, bias: float) -> float:
    """Compute ``W * X + b``."""
    return sum(fi * wi for fi, wi in zip(features, weights)) + bias


def calculate_gradients(
    features: list, targets: float, weights: list, bias: float
) -> tuple[list, float]:
    """Calculate gradients of loss (mean squared error) with respect to weights
    and bias."""
    error = predict(features, weights, bias) - targets
    grad_w = [2 * error * f for f in features]
    grad_b = 2 * error

    return grad_w, grad_b


def normalize(features: list) -> list:
    """Normalize the features with min 0 and max 1."""
    transposed = list(zip(*features))
    min_max = [(min(feature), max(feature)) for feature in transposed]

    normalized = []
    for row in features:
        normalized_row = []
        for i, feat in enumerate(row):
            normalized_feature = (feat - min_max[i][0]) / (
                min_max[i][1] - min_max[i][0]
            )
            normalized_row.append(normalized_feature)
        normalized.append(normalized_row)
    return normalized


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
    features, targets = data[0].tolist(), data[1].tolist()
    normalized_features = normalize(features)

    # randomly initialize weights and bias
    weights = [random.uniform(0, 1) for _ in range(data[0].shape[1])]
    bias = random.uniform(0, 1)
    lr = 1e-4

    for epoch in range(num_epochs):
        for feats, y in zip(normalized_features, targets):
            grad_w, grad_b = calculate_gradients(feats, y, weights, bias)
            weights = [wi - lr * gw for wi, gw in zip(weights, grad_w)]
            bias -= lr * grad_b

        # optionally print the progress
        if epoch % 100 == 0:
            loss = sum(
                [
                    (predict(feats, weights, bias) - y) ** 2
                    for feats, y in zip(normalized_features, targets)
                ]
            ) / len(data)
            print(
                f"Epoch {epoch}: Loss = {loss:.5f}"
                # f"weight = {w}, bias = {b:.5f}"
            )

    # print(f"Final parameters: weight = {w}, bias = {b:.5f}")
    print(f"Time taken: {(time.perf_counter() - start):.4f} seconds")


if __name__ == "__main__":
    main(num_epochs=1000)
