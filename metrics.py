import numpy as np


regression_metrics = ['Equation', 'R2 Score', 'RMSE', 'MSE', 'MAE']
classification_metrics = ['Equation', 'Recall', 'Precision', 'F1 Score', 'Accuracy']

soft_margin_svc = {
    'summary': 'Find the optimal hyperplane which maximally separates two classes of data.',
    'mathjax': '\(\min \\frac{1}{2} ||w^{2}|| + C\sum{\max{(0,1-y_i(w^Tx_i+b))}}\)',
    'assumptions': '1. The Data is Partially Separable\n \
                    2. Misclassification is allowed within the margin to prevent overfitting, controlled by hyperparameter C',
    'info-href': 'https://en.wikipedia.org/wiki/Support_vector_machine',
    'info-text': 'https://en.wikipedia.org/wiki/Support_vector_machine'
}


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


#  Classification Metrics
#  Recall, the proportion of true positives among all positive samples
def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)


#  Precision, the proportion of true positives among predicted positive samples
def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)


#  F1 Score, the harmonic mean of precision and recall
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)


#  Accuracy, the proportion of correctly classified samples
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)