def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        # Forward pass
        z = np.dot(X, w) + b
        predictions = _sigmoid(z)

        dw = np.dot(X.T, predictions - y) / n_samples
        db = np.sum(predictions - y) / n_samples

        w -= lr * dw
        b -= lr * db

    return w, float(b)