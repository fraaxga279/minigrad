import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Module(ABC):
    def __init__(self):
        self.output = None
        self.training = True

    @abstractmethod
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        pass

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.output = self.compute_output(input)
        return self.output

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad_input = self.compute_grad_input(input, grad_output)
        self.update_grad_parameters(input, grad_output)
        return grad_input

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def zero_grad(self):
        pass

    def parameters(self) -> List[np.ndarray]:
        return []

    def parameters_grad(self) -> List[np.ndarray]:
        return []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
