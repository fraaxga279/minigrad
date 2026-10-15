import numpy as np
from .module import Module
from typing import List

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)
        self.inputs = None
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        self.inputs = [input]
        output = input
        for module in self.modules:
            output = module(output)
            self.inputs.append(output)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for i in range(len(self.modules) - 1, -1, -1):
            grad = self.modules[i].backward(self.inputs[i], grad)
        return grad

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str