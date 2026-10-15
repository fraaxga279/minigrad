import numpy as np

from minigrad.nn import BatchNormalization

def test_batchnorm_shape():
    layer = BatchNormalization(3)
    x = np.random.randn(8, 3)
    output = layer(x)
    assert output.shape == x.shape

def test_batchnorm_normalizes():
    np.random.seed(42)
    layer = BatchNormalization(
        3,
        affine=False
    )
    x = np.random.randn(100, 3) * 5 + 10
    output = layer(x)
    assert np.allclose(
        output.mean(axis=0),
        0,
        atol=1e-6
    )
    assert np.allclose(
        output.var(axis=0),
        1,
        atol=1e-4
    )