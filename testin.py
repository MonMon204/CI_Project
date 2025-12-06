import numpy as np
from lib.optimizer import SGD

from lib.network import Sequential
from lib.layers import Dense
from lib.activations import Tanh, Sigmoid
from lib.losses import mse_grad, mse_loss

x1 = 20
x2 = 9

model = Sequential()
model.add(Dense(2, 8))
model.add(Tanh())
model.add(Dense(8, 1))
model.add(Tanh())

model.load("xor_weights.npy")
print("Model loaded!")
inp = np.array([[x1],[x2]], dtype=float)
out = model.forward(inp)
print("Raw output:", out)
print("XOR prediction:", (out > 0).astype(int))
