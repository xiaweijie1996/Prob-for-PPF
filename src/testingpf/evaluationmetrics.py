import numpy as np

# MSE
def mse_loss(output, target):
    return np.mean((output - target) ** 2)

# RMSE
def rmse_loss(output, target):
    return np.sqrt(mse_loss(output, target))