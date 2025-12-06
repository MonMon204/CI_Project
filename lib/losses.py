import numpy as np

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error Loss
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    """
    Gradient of MSE with respect to y_pred
    """
    return 2 * (y_pred - y_true) / y_true.size
