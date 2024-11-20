import numpy as np


# Define MAPE loss function
def MAPE(true, pred):
    return np.mean(np.abs((true - pred) / (true))) * 100


# Define SMAPE loss function
def SMAPE(true, pred):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 100


# Define NMAE loss function
def NMAE(true, pred, nominal):
    absolute_error = np.abs(true - pred)

    absolute_error /= nominal

    target_idx = np.where(true >= nominal * 0.1)

    return 100 * absolute_error.iloc[target_idx].mean()
