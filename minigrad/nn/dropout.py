import numpy as np
from .module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
            return input * self.mask / (1 - self.p)
        else:
            return input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        if self.training:
            return grad_output * self.mask / (1 - self.p)
        return grad_output

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'