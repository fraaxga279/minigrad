import numpy as np
from .base import Criterion

class MSELoss(Criterion):
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.mean((input - target) ** 2)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 * (input - target) / input.size
