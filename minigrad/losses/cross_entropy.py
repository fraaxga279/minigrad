import numpy as np
from .base import Criterion
from minigrad.nn.activation import LogSoftmax

class CrossEntropyLoss(Criterion):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        B, C = input.shape
        log_probs = self.log_softmax(input)
        target_distribution = np.full(input.shape, self.label_smoothing / C, dtype=input.dtype)
        target_distribution[np.arange(B), target] += 1 - self.label_smoothing
        return -np.sum(target_distribution * log_probs) / B

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        B, C = input.shape
        log_probs = self.log_softmax(input)
        probs = np.exp(log_probs)
        target_distribution = np.full(input.shape, self.label_smoothing / C, dtype=input.dtype)
        target_distribution[np.arange(B), target] += 1 - self.label_smoothing
        return (probs - target_distribution) / B