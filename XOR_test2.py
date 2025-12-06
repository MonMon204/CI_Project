import os
print("FILE RUNNING FROM:", os.path.abspath(__file__))

import numpy as np

from lib.network import Sequential
from lib.layers import Dense
from lib.activations import Tanh, Sigmoid
from lib.losses import mse_grad, mse_loss

# XOR dataset
x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)

x = (x > 0).astype(float)

y = np.array([[-1, 1, 1, -1]], dtype=float)

model = Sequential()
model.add(Dense(2, 8))
model.add(Tanh())
model.add(Dense(8, 1))
model.add(Tanh())

model.train(x, y, mse_loss, mse_grad, epochs=20000, lr=0.3)

print("Final predictions:")
print(model.forward(x))
print("True labels:")
print(y)
