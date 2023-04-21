import numpy as np


regression_metrics = ['Equation', 'R2 Score', 'RMSE', 'MSE', 'MAE']
classification_metrics = ['Equation', 'Recall', 'Precision', 'F1 Score', 'Accuracy']

soft_margin_svc = {
    'summary': 'Find the optimal hyperplane which maximally separates two classes of data.',
    'mathjax': '\(\min \\frac{1}{2} ||w^{2}|| + C\sum{\max{(0,1-y_i(w^Tx_i+b))}}\)',
    'assumptions': '1. The dependent variable is 1 or -1\n \
                    2. The Data is Partially Separable\n \
                    3. Misclassification is allowed within the margin to prevent overfitting, controlled by hyperparameter C',
    'complexity': '\(Train: O(n^2)\\\Test: O(n*m) \\\Space: O(n*m)\)',
    'info-href': 'https://en.wikipedia.org/wiki/Support_vector_machine',
    'info-text': 'https://en.wikipedia.org/wiki/Support_vector_machine'
}

linear_regression = {
    'summary': 'Model the linear relationship between two variables by finding the line of best fit to the data',
    'mathjax': '\(\min\\beta: \\frac{1}{N} \sum{(y_i-(\\beta_0+\\beta_1x_i)^2}\)',
    'assumptions': '1. Linear relationship between variables\n \
                    2. Independent error terms\n \
                    3. Constant variance of errors\n \
                    4. Errors follow a normal distribution\n \
                    5. Independent variables are not highly correlated',
    'complexity': '\(Train: O(n*m^2+m^3)\\\Test: O(m) \\\Space: O(m)\)',
    'info-href': 'https://en.wikipedia.org/wiki/Linear_regression',
    'info-text': 'https://en.wikipedia.org/wiki/Linear_regression'
}

logistic_regression = {
    'summary': 'Model the probability of a binary outcome (0 or 1) based on one or more predictor variables',
    'mathjax': '\(\max\\beta: \sum{[y_i\cdot log(p_i)+(1-y_i)\cdot log(1-p_i)]}\) Where \(p_i=\\frac{1}{1+e^{-(\\beta_1x_i+\\beta_0)}} \)',
    'assumptions': '1. The dependent variable is 0 or 1\n \
                    2. Independent observations\n \
                    3. Linear relationship between predictors\n \
                    4. Large sample size\n \
                    5. Independent variables are not highly correlated',
    'complexity': '\(Train: O(n*m)\\\Test: O(m) \\\Space: O(m)\)',
    'info-href': 'https://en.wikipedia.org/wiki/Logistic_regression',
    'info-text': 'https://en.wikipedia.org/wiki/Logstic_regression'

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
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn)


#  Precision, the proportion of true positives among predicted positive samples
def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp)


#  F1 Score, the harmonic mean of precision and recall
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)


#  Accuracy, the proportion of correctly classified samples
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
