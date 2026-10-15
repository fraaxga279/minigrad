import numpy as np
from .optimizer import Optimizer
from minigrad.nn.module import Module
class SGD(Optimizer):
    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
        for param, grad, m in zip(parameters, gradients, self.state['m']):
            grad = grad + self.weight_decay * param
            np.multiply(m, self.momentum, out=m)
            np.add(m, grad, out=m)
            if self.nesterov:
                update = grad + self.momentum * m
            elif self.momentum != 0:
                update = m
            else:
                update = grad
            param -= self.lr * update