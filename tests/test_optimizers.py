import numpy as np

from minigrad.nn import Linear
from minigrad.optim import SGD, Adam


def test_sgd_step():
    layer = Linear(2, 1, bias=False)
    layer.weight[:] = np.array([[1.0, 2.0]])
    layer.grad_weight[:] = np.array([[0.5, -1.0]])
    optimizer = SGD(layer, lr=0.1)
    optimizer.step()
    expected = np.array([
        [0.95, 2.1]
    ])
    assert np.allclose(
        layer.weight,
        expected
    )

def test_optimizer_zero_grad():
    layer = Linear(2, 1)
    layer.grad_weight[:] = 3.0
    layer.grad_bias[:] = 2.
    optimizer = SGD(layer)
    optimizer.zero_grad()
    assert np.all(layer.grad_weight == 0)
    assert np.all(layer.grad_bias == 0)

def test_adam_changes_weights():
    layer = Linear(2, 1, bias=False)
    initial_weight = layer.weight.copy()
    layer.grad_weight[:] = np.array([
        [0.5, -1.0]
    ])
    optimizer = Adam(layer, lr=0.01)
    optimizer.step()
    assert not np.allclose(
        layer.weight,
        initial_weight
    )

def test_adam_step_counter():
    layer = Linear(2, 1, bias=False)
    optimizer = Adam(layer)
    layer.grad_weight[:] = 1.0
    optimizer.step()
    assert optimizer.state["t"] == 1
    optimizer.step()
    assert optimizer.state["t"] == 2
    