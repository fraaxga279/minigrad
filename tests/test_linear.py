import numpy as np
from minigrad.nn import Linear

def test_linear_forward_shape():
    layer = Linear(3, 2)
    x = np.random.randn(4, 3)
    y = layer(x)
    assert y.shape == (4, 2)

def test_linear_backward_shape():
    layer = Linear(3, 2)
    x = np.random.randn(4, 3)
    grad_output = np.random.randn(4, 2)
    grad_input = layer.backward(x, grad_output)
    assert grad_input.shape == x.shape

def test_linear_weight_gradient():
    np.random.seed(42)
    layer = Linear(3, 2)
    x = np.random.randn(4, 3)
    grad_output = np.random.randn(4, 2)
    layer.zero_grad()
    layer.backward(x, grad_output)
    analytical_grad = layer.grad_weight.copy()
    eps = 1e-5
    numerical_grad = np.zeros_like(layer.weight)
    for i in range(layer.weight.shape[0]):
        for j in range(layer.weight.shape[1]):
            original = layer.weight[i, j]
            layer.weight[i, j] = original + eps
            y_plus = layer(x)
            loss_plus = np.sum(y_plus * grad_output)
            layer.weight[i, j] = original - eps
            y_minus = layer(x)
            loss_minus = np.sum(y_minus * grad_output)
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
            layer.weight[i, j] = original
    assert np.allclose(
        analytical_grad,
        numerical_grad,
        atol=1e-5
    )