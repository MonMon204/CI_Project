import numpy as np
from .layers import Layer


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        grad = grad_output * (self.input > 0)
        return grad

class Tanh:
    def forward(self, x):
        self.input = x
        return np.tanh(x)

    def backward(self, grad_output):
        return grad_output * (1 - np.tanh(self.input)**2)


class Sigmoid:
    def forward(self, x):
        self.input = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad_output):
        sig = 1 / (1 + np.exp(-self.input))
        return grad_output * sig * (1 - sig)
