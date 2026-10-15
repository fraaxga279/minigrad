import numpy as np
from minigrad.data import DataLoader

def test_dataloader_batches():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    loader = DataLoader(
        X,
        y,
        batch_size=2,
        shuffle=False,
    )
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (2, 2)
    assert batches[1][0].shape == (2, 2)
    assert batches[2][0].shape == (1, 2)

def test_dataloader_keeps_pairs():
    X = np.array([
        [10],
        [20],
        [30],
    ])
    y = np.array([1, 2, 3])
    loader = DataLoader(
        X,
        y,
        batch_size=1,
        shuffle=False,
    )
    for x_batch, y_batch in loader:
        assert x_batch[0, 0] // 10 == y_batch[0]