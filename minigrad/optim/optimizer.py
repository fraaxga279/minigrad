from abc import ABC, abstractmethod
from minigrad.nn.module import Module

class Optimizer(ABC):
    def __init__(self, module: Module):
        self.module = module
        self.state = {}

    def zero_grad(self):
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        raise NotImplementedError
