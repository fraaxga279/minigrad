import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(
        self,
        module,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if "m" not in self.state:
            self.state["m"] = [
                np.zeros_like(param)
                for param in parameters
            ]
            self.state["v"] = [
                np.zeros_like(param)
                for param in parameters
            ]
            self.state["t"] = 0

        self.state["t"] += 1
        t = self.state["t"]
        for param, grad, m, v in zip(
            parameters,
            gradients,
            self.state["m"],
            self.state["v"]
        ):
            grad = grad + self.weight_decay * param
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            param -= self.lr * m_hat / (
                np.sqrt(v_hat) + self.eps
            )