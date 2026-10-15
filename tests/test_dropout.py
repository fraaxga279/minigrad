import numpy as np

from minigrad.nn import Dropout

def test_dropout_eval_does_nothing():
    layer = Dropout(p=0.5)
    x = np.random.randn(10, 5)
    layer.eval()
    output = layer(x)
    assert np.allclose(output, x)

def test_dropout_preserves_shape():
    np.random.seed(42)
    layer = Dropout(p=0.5)
    x = np.ones((100, 10))
    layer.train()
    output = layer(x)
    assert output.shape == x.shape

def test_dropout_backward_shape():
    np.random.seed(42)
    layer = Dropout(p=0.5)
    x = np.ones((10, 5))
    layer.train()
    layer(x)
    grad_output = np.ones_like(x)
    grad_input = layer.backward(
        x,
        grad_output
    )
    assert grad_input.shape == x.shape