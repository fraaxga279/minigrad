import numpy as np
from minigrad.nn import ReLU, Sigmoid, GELU, Softmax
from tests.utils import numerical_gradient

def test_relu_gradient():
    layer = ReLU()
    x = np.array([
        [-2.0, -1.0, 1.0],
        [2.0, 3.0, -4.0],
    ])
    grad_output = np.random.randn(*x.shape)
    analytical = layer.backward(x, grad_output)
    def f():
        return np.sum(layer(x) * grad_output)
    numerical = numerical_gradient(f, x)
    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5
    )

def test_sigmoid_gradient():
    layer = Sigmoid()
    x = np.array([
        [-2.0, -1.0, 1.0],
        [2.0, 3.0, -4.0],
    ])
    grad_output = np.random.randn(*x.shape)
    analytical = layer.backward(x, grad_output)
    def f():
        return np.sum(layer(x) * grad_output)
    numerical = numerical_gradient(f, x)
    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5
    )

def test_gelu_gradient():
    layer = GELU()
    x = np.array([
        [-2.0, -1.0, 1.0],
        [2.0, 3.0, -4.0],
    ])
    grad_output = np.random.randn(*x.shape)
    analytical = layer.backward(x, grad_output)
    def f():
        return np.sum(layer(x) * grad_output)
    numerical = numerical_gradient(f, x)
    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5
    )

def test_softmax_gradient():
    layer = Softmax()
    x = np.array([
        [1.0, 2.0, 3.0],
        [0.5, -1.0, 2.0],
    ])
    grad_output = np.random.randn(*x.shape)
    analytical = layer.backward(x, grad_output)
    def f():
        return np.sum(layer(x) * grad_output)
    numerical = numerical_gradient(f, x)
    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5
    )