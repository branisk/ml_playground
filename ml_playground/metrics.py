import numpy as np


#  Regression Metrics
#  Return a value between 0 and 1, a larger value represents a greater fit
def r2_score(Y_pred, Y_actual):
    return 1 - (np.sum(Y_actual - Y_pred)**2 / np.sum(Y_actual - Y_pred))


#  Mean Squared Error
def mse(Y_pred, Y_actual):
    n = len(Y_pred)
    return (np.sum(Y_actual - Y_pred) ** 2) / n


#  Root Mean Squared Error (The Square root of MSE)
def rmse(Y_pred, Y_actual):
    n = len(Y_pred)
    return np.sqrt((np.sum(Y_actual - Y_pred) ** 2) / n)


#  Mean Absolute Error
def mae(Y_pred, Y_actual):
    n = len(Y_pred)
    return np.abs(np.sum(Y_actual - Y_pred)) / n
