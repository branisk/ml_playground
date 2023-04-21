import plotly.express as px
import plotly.graph_objects as go

from metrics import *


#  Regression Methods
class LinearRegression:
    def __init__(
            self,
            n_features=1,
            max_iterations=1,
            optimizer="OLS"
    ):
        self.max_iter = max_iterations
        self.optimizer = optimizer
        self.weights = np.zeros(n_features + 1)  # Weight term
        self.results = [None] * 5

    def fit(self, X, Y):
        n = len(X)

        sx = np.sum(X)
        sy = np.sum(Y)
        sxx = np.dot(X.T, X)
        syy = np.dot(Y.T, Y)
        sxy = np.dot(X.T, Y)

        self.weights[1] = (n*sxy - sx*sy) / (n*sxx - sx*sx)
        self.weights[0] = (sy/n - self.weights[1]*sx/n)

    def plot_best_fit(self, X, X_test, Y_test):
        Y_pred = self.predict(X_test)
        residuals = Y_test.T[0] - Y_pred

        fig = px.scatter()
        x_vals = np.linspace(np.min(X), np.max(X), 100)
        y_vals = (x_vals * self.weights[1]) + self.weights[0]
        line = px.line(x=x_vals, y=y_vals, color_discrete_sequence=['red'])
        line.data[0]['showlegend'] = True
        line.data[0]['name'] = 'Best Fit'

        fig.add_trace(line.data[0])

        # Add predicted points and dotted lines
        for i in range(len(X_test)):
            x_test = X_test[i]
            y_test = Y_test[i]
            y_pred = Y_pred[i]
            residual = residuals[i]

            if i == 0:
                scatter = go.Scatter(x=[x_test, x_test], y=[y_test, y_pred], mode='lines', name='Residuals', legendgroup='1', line=dict(color='white', dash='dash'), showlegend=True, hovertemplate=f"Residual: {residual:.5f}<extra></extra>")
            else:
                scatter = go.Scatter(x=[x_test, x_test], y=[y_test, y_pred], mode='lines', legendgroup='1', line=dict(color='white', dash='dash'), showlegend=False, hovertemplate=f"Residual: {residual:.5f}<extra></extra>")

            fig.add_trace(scatter)

        fig.data[0]['showlegend'] = True
        fig.data[0]['name'] = 'Residuals'

        return fig

    def plot_residuals(self, X, Y):
        Y_pred = self.predict(X)
        residuals = Y.T[0] - Y_pred

        scatter = px.scatter(x=list(range(len(residuals))), y=residuals)

        return scatter

    def update_results(self, X, Y):
        Y_pred = self.predict(X)
        self.results = [
            f'{round(self.weights[1], 2)}x + {round(self.weights[0], 2)}',
            str(round(r2_score(Y_pred, Y)*100, 2)) + "%",
            round(rmse(Y_pred, Y), 5),
            round(mse(Y_pred, Y), 5),
            round(mae(Y_pred, Y), 5)
        ]

    def predict(self, X):
        return (X * self.weights[1]) + self.weights[0]


#  Classification Methods
class LogisticRegression:
    def __init__(
            self,
            n_features=2,
            learning_rate=0.01,
            regularization_term=0.1,
            max_iterations=1,
            regularization_type="Ridge (L2)",
            optimizer="Sub-Gradient Descent"
    ):
        self.eta = learning_rate
        self.C = regularization_term
        self.max_iter = max_iterations
        self.W = np.zeros(n_features)  # Weight term
        self.regularization_type = regularization_type  # L2=Ridge, L1=Lasso
        self.optimizer = optimizer
        self.results = [None] * 5

    def fit(self, X, Y):
        for i in range(self.max_iter):
            # Compute predictions and gradients
            self._sgd_step(X, Y)

        # Plot the decision boundary
        fig = px.scatter()
        x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y_vals = -(self.W[0] * x_vals + self.W[1]) / self.W[1]
        fig.add_trace(px.line(x=x_vals, y=y_vals).data[0])

        return fig

    def _sgd_step(self, X, Y):
        predictions = self.predict(X)
        difference = predictions.T - Y
        grad = np.dot(X.T, difference)
        grad = np.array([grad[0][0], grad[1][0]])

        self.W -= self.eta * grad

    def predict(self, x):
        return 1 / (1 + np.exp(x**2))  # Logistic Function

    def cross_entropy_loss(self, X, Y):
        predictions = np.array(np.sum([self.predict(x) for x in X]))
        regularization_term = self.C * np.dot(self.W[:-1], self.W[:-1])
        loss = -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions)) + regularization_term
        return loss


class SupportVectorClassifier:
    def __init__(self, n_features=2, kernel='linear', learning_rate=0.01,
                 regularization_term=0.1, max_iterations=1,
                 regularization_type="Ridge (L2)", optimizer="Sub-Gradient Descent"):
        self.kernel = kernel
        self.eta = learning_rate
        self.C = regularization_term
        self.max_iter = max_iterations
        self.W = np.zeros(n_features)  # Weight term
        self.b = 0  # "bias" or "intercept" term
        self.regularization_type = regularization_type  # L2=Ridge, L1=Lasso
        self.optimizer = optimizer
        self.results = [None] * 5

    #  X is 2 dimensional, and Y are our labels
    def fit(self, X, Y):
        for i in range(self.max_iter):
            if self.optimizer == "Sub-Gradient Descent":
                self._sgd_step(X, Y)
            elif self.optimizer == "Newton's Method":
                self._newton_step(X, Y)

    def _sgd_step(self, X, Y):
        w_grad = np.zeros_like(self.W)
        b_grad = 0

        # Calculate the Sub Gradients of the hinge loss function
        for x_i, y_i in zip(X, Y):
            fx_i = np.dot(self.W, x_i) + self.b
            t = y_i * fx_i

            if t < 1:
                w_grad -= y_i * x_i
                b_grad -= y_i

        #  Update the gradients based on the regularization type
        if self.regularization_type == "Ridge (L2)":
            w_grad += self.W + (self.C * w_grad)
        elif self.regularization_type == "Lasso (L1)":  # Many of these weights turn to 0
            w_grad += self.W + self.C * np.sign(w_grad)

        #  Update the weights
        self.W -= self.eta * w_grad
        self.b -= self.eta * b_grad

    def _newton_step(self, X, Y):
        w_grad = np.zeros_like(self.W)
        b_grad = 0
        hessian = np.zeros((len(self.W) + 1, len(self.W) + 1))

        for x_i, y_i in zip(X, Y):
            fx_i = np.dot(self.W, x_i) + self.b
            t = y_i * fx_i

            if t < 1:
                w_grad -= y_i * x_i
                b_grad -= y_i

                hessian[0, 0] += y_i ** 2
                for j in range(len(self.W)):
                    hessian[0, j + 1] += y_i * x_i[j]
                    hessian[j + 1, 0] += y_i * x_i[j]
                    for k in range(len(self.W)):
                        hessian[j + 1, k + 1] += x_i[j] * x_i[k]

        # Add regularization term to diagonal of Hessian
        hessian[1:, 1:] += 2 * self.C * np.eye(len(self.W))

        # Compute Newton direction
        if np.linalg.det(hessian) == 0:
            print("Can't improve, exiting early..")
            return
        newton_direction = np.linalg.solve(hessian, np.hstack([b_grad, w_grad]))

        # Update weights and bias
        self.b -= newton_direction[0]
        self.W -= newton_direction[1:]

    def plot_hyperplane(self, X):
        # Plot the hyperplane
        fig = px.scatter()
        x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y_vals = (-self.b - self.W[0] * x_vals) / self.W[1]
        fig.add_trace(px.line(x=x_vals, y=y_vals).data[0])

        # Add dotted decision boundary
        y_margin = 1 / self.W[1]
        y_upper = y_vals + y_margin
        y_lower = y_vals - y_margin
        fig.add_trace(go.Scatter(x=x_vals, y=y_upper, line=dict(dash="dash"), name="Upper margin"))
        fig.add_trace(go.Scatter(x=x_vals, y=y_lower, line=dict(dash="dash"), name="Lower margin"))

        return fig

    def hinge_loss(self, X, Y):
        distance_sum = 0

        for x_i, y_i in zip(X, Y):
            distance_sum += max(0, 1 - y_i * (np.dot(self.W, x_i) + self.b))

        if self.regularization_type == "Ridge (L2)":
            regularization_term = 0.5 * np.dot(np.transpose(self.W), self.W)
            error_term = self.C * distance_sum
        elif self.regularization_type == "Lasso (L1)":
            regularization_term = self.C * np.abs(self.W)
            error_term = distance_sum
        else:
            print("No regularization type set")
            return None

        loss = regularization_term + error_term

        return loss

    def update_results(self, X, Y):
        Y_pred = self.predict(X)

        self.results = [
            f'{np.round(self.W, 2)}x + {np.round(self.b, 2)}',
            str(round(recall(Y_pred, Y.T[0])*100, 2)) + "%",
            str(round(precision(Y_pred, Y.T[0])*100, 2)) + "%",
            str(round(f1_score(Y_pred, Y.T[0])*100, 2)) + "%",
            str(round(accuracy(Y_pred, Y.T[0])*100, 2)) + "%"
        ]

        return

    def predict(self, X):
        return np.array([1 if np.dot(self.W, x_i) + self.b >= 0 else -1 for x_i in X])


#  Clustering Algorithms
class KMeans:
    def __init__(
            self
    ):
        return
