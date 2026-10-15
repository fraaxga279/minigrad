import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        f_plus = f()
        x[idx] = original - eps
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad