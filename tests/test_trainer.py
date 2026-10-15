import numpy as np

from minigrad.nn import Linear, ReLU, Sequential
from minigrad.losses import CrossEntropyLoss
from minigrad.optim import Adam
from minigrad.data import DataLoader
from minigrad.trainer import Trainer


def test_trainer_returns_history():
    np.random.seed(42)

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 2),
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(
        model,
        lr=0.01
    )
    loader = DataLoader(
        X,
        y,
        batch_size=4
    )
    trainer = Trainer(
        model,
        loss_fn,
        optimizer
    )
    history = trainer.fit(
        loader,
        epochs=100
    )
    assert "loss" in history
    assert len(history["loss"]) == 100
    assert (
        history["loss"][-1]
        <
        history["loss"][0]
    )