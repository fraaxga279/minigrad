import numpy as np

from minigrad.losses import MSELoss, CrossEntropyLoss
from tests.utils import numerical_gradient

def test_mse_gradient():
    loss_fn = MSELoss()
    pred = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    target = np.array([
        [0.0, 1.0],
        [2.0, 2.0],
    ])
    analytical = loss_fn.backward(pred, target)
    def f():
        return loss_fn(pred, target)
    numerical = numerical_gradient(f, pred)
    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5
    )
    
def test_cross_entropy_gradient():
    loss_fn = CrossEntropyLoss()
    logits = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 0.5],
    ])
    target = np.array([2, 0])
    analytical = loss_fn.backward(logits, target)
    def f():
        return loss_fn(logits, target)
    numerical = numerical_gradient(f, logits)
    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5
    )