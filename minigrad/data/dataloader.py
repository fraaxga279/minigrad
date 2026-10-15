import numpy as np

class DataLoader(object):
    def __init__(self, X, y, batch_size=1, shuffle=False):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0
        self.indices = np.arange(X.shape[0])

    def __len__(self) -> int:
        return (self.num_samples() + self.batch_size - 1) // self.batch_size

    def num_samples(self) -> int:
        return self.X.shape[0]

    def __iter__(self):
        self.batch_id = 0
        self.indices = np.arange(self.num_samples())
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.batch_id >= len(self):
            raise StopIteration
        start = self.batch_id * self.batch_size
        end = min(start + self.batch_size, self.num_samples())
        indices = self.indices[start:end]
        self.batch_id += 1
        return self.X[indices], self.y[indices]