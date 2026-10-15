import numpy as np
from scipy import special
from .module import Module

class ReLU(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (input > 0)


class Sigmoid(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return special.expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        sigmoid = special.expit(input)
        return grad_output * sigmoid * (1 - sigmoid)


class GELU(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return input * special.ndtr(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        cdf = special.ndtr(input)
        pdf = (np.exp(-(input ** 2) / 2) / np.sqrt(2 * np.pi))
        grad_gelu = cdf + input * pdf
        return grad_output * grad_gelu

class Softmax(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return special.softmax(input, axis=-1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        output = special.softmax(input, axis=-1)
        return output * (grad_output - np.sum(grad_output * output, axis=-1, keepdims=True))


class LogSoftmax(Module):
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        return (input - special.logsumexp(input, axis=-1, keepdims=True))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        softmax = special.softmax(input, axis=-1)
        return (grad_output - softmax * np.sum(grad_output, axis=-1, keepdims=True))