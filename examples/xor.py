import numpy as np

from minigrad.nn import Linear, ReLU, Sequential
from minigrad.losses import CrossEntropyLoss
from minigrad.optim import Adam
from minigrad.data import DataLoader
from minigrad.trainer import Trainer


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
    lr=0.01,
)

loader = DataLoader(
    X,
    y,
    batch_size=4,
    shuffle=True,
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

history = trainer.fit(
    loader,
    epochs=1000,
)

model.eval()

logits = model(X)
pred = np.argmax(logits, axis=1)

accuracy = np.mean(pred == y)

print("loss:", history["loss"][0], "->", history["loss"][-1])
print("pred:", pred)
print("true:", y)
print("accuracy:", accuracy)