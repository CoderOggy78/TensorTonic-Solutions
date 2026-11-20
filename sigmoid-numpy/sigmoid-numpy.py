import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    result = 1 / (1 + np.exp(-np.array(x)))

    if np.isscalar(x):
        return float(result)
    return result.astype(float)