import numpy as np

from minigrad.nn import Linear, ReLU, Sequential
from minigrad.losses import CrossEntropyLoss
from minigrad.optim import Adam
from minigrad.data import DataLoader

def test_xor_training_reduces_loss():
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
    optimizer = Adam(model, lr=0.01)
    loader = DataLoader(
        X,
        y,
        batch_size=4,
        shuffle=True,
    )
    initial_logits = model(X)
    initial_loss = loss_fn(initial_logits, y)
    for _ in range(500):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            grad = loss_fn.backward(logits, y_batch)
            model.backward(x_batch, grad)
            optimizer.step()
    final_logits = model(X)
    final_loss = loss_fn(final_logits, y)
    assert final_loss < initial_loss

def test_xor_training_reaches_good_accuracy():
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
    optimizer = Adam(model, lr=0.01)
    loader = DataLoader(X, y, batch_size=4, shuffle=True)
    for _ in range(1000):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            grad = loss_fn.backward(logits, y_batch)
            model.backward(x_batch, grad)
            optimizer.step()
    pred = np.argmax(model(X), axis=1)
    accuracy = np.mean(pred == y)
    assert accuracy >= 0.75