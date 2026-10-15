class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {
            "loss": []
        }
    def fit(self, loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            for x, y in loader:
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_fn(output, y)
                grad = self.loss_fn.backward(output, y)
                self.model.backward(x, grad)
                self.optimizer.step()
                epoch_loss += loss
                batches += 1
            epoch_loss /= batches
            self.history["loss"].append(epoch_loss)
        return self.history