import numpy as np
from abc import ABC, abstractmethod

class Criterion(ABC):
    def __init__(self):
        self.output = None

    @abstractmethod
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self.output = self.compute_output(input, target)
        return self.output

    def backward(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        grad_input = self.compute_grad_input(input, target)
        return grad_input

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
