import numpy as np
from .module import Module
from typing import List

class BatchNormalization(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        self.mean = None
        self.input_mean = None
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        if self.training:
            self.mean = input.mean(axis=0)
            self.var = input.var(axis=0)
            B = input.shape[0]
            self.norm_input = (input - self.mean) / np.sqrt(self.var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (B / (B - 1)) * self.var
        else:
            self.norm_input = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
        if self.affine:
            return self.norm_input * self.weight + self.bias
        else:
            return self.norm_input

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        B = input.shape[0]
        if self.affine:
            grad_x_hat = grad_output * self.weight
        else:
            grad_x_hat = grad_output
        if self.training:
            grad_var = np.sum(grad_x_hat * (input - self.mean), axis=0) * (-0.5) * (self.var + self.eps) ** (-1.5)
            grad_mean = (np.sum(grad_x_hat * ((-1) / np.sqrt(self.var + self.eps)), axis=0) 
            + grad_var * (np.sum(-2 * (input - self.mean), axis=0)) / B)
            grad_input = (grad_x_hat * (1/np.sqrt(self.var + self.eps)) + grad_var * 2 * (input - self.mean) / B
            + grad_mean / B)
            return grad_input
        else:
            return grad_x_hat / np.sqrt(self.running_var + self.eps)

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        if self.affine:
            self.grad_bias += np.sum(grad_output, axis=0)
            self.grad_weight += np.sum(grad_output * self.norm_input, axis=0)
        else:
            return

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'
