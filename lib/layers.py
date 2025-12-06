import numpy as np

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.5
        self.b = np.zeros((out_features, 1))

    def forward(self, input):
        self.input = input
        return self.W @ self.input + self.b

    def backward(self, grad_output):
     dW = grad_output @ self.input.T
     db = np.sum(grad_output, axis=1, keepdims=True)
     d_input = self.W.T @ grad_output

     self.dW = dW
     self.db = db

     return d_input


    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]

