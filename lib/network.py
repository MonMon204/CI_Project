import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, optimizer):
     for layer in reversed(self.layers):
         grad = layer.backward(grad)
         if hasattr(layer, "parameters"):
             optimizer.step(layer.parameters(), layer.gradients())


    def train(self, X, Y, loss_fn, loss_grad_fn, epochs, optimizer):
        m = X.shape[1]   # عدد السامبلز

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(m)
            X = X[:, idx]
            Y = Y[:, idx]

            for i in range(m):
                x_i = X[:, i:i+1]   # sample واحدة
                y_i = Y[:, i:i+1]

                out = self.forward(x_i)
                loss = loss_fn(y_i, out)
                grad = loss_grad_fn(y_i, out)

                self.backward(grad, optimizer)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss = {loss}")


    def save(self, path):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.append(layer.parameters())
        np.save(path, np.array(params, dtype=object), allow_pickle=True)

    def load(self, path):
        params = np.load(path, allow_pickle=True)
        idx = 0
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                W, B = params[idx]
                layer.W = W
                layer.b = B
                idx += 1



