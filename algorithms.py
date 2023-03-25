import numpy as np
import plotly.express as px


class SupportVectorClassifier:
    def __init__(
            self,
            linearity='linear',
            learning_rate=0.01,
            regularization_term=0.1,
            max_iterations=1,
            regularization_type="L2",
            optimizer="Sub-Gradient Descent"
    ):
        self.linearity = linearity
        self.eta = learning_rate
        self.C = regularization_term
        self.max_iter = max_iterations
        self.W = np.zeros(2)
        self.b = 0 # "bias" or "intercept" term
        self.w_grad = np.zeros_like(self.W)
        self.b_grad = 0
        self.regularization_type = regularization_type # L2 is Ridge, L1 is Lasso
        self.optimizer = optimizer

    def fit(self, X, Y):
        # Train our SVM using stochastic gradient descent
        for i in range(self.max_iter):
            if self.optimizer == "Sub-Gradient Descent":
                self._sgd_step(X, Y)
            elif self.optimizer == "Newton's Method":
                self._newton_step(X, Y)

        print("SVM is fit.")
        print(self.accuracy(X, Y))

        # Plot the hyperplane
        fig = px.scatter()
        x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        y_vals = (-self.b - self.W[0] * x_vals) / self.W[1]
        fig.add_trace(px.line(x=x_vals, y=y_vals).data[0])

        return fig

    def _sgd_step(self, X, Y):
        # Calculate the Sub Gradients of the hinge loss function
        for x_i, y_i in zip(X, Y):
            fx_i = np.dot(self.W, x_i) + self.b
            t = y_i * fx_i

            if t < 1:
                self.w_grad -= y_i * x_i
                self.b_grad -= y_i

        # Sub Gradient Descent
        if self.regularization_type == "L2":
            print("Using Ridge Regression")
            self.w_grad = self.W + (self.C * self.w_grad)
            self.b_grad = self.C * self.b_grad
        elif self.regularization_type == "L1": # Many of these weights turn to 0
            print("Using Lasso Regression")
            self.w_grad = self.W + self.C * np.sign(self.w_grad)
            self.b_grad = self.C * np.sign(self.b_grad)

        self.W -= self.eta * self.w_grad
        self.b -= self.eta * self.b_grad

        loss = self.hinge_loss(X,Y)
        print(f"SGD step: {loss}")

    def _newton_step(self, X, Y):
        # Calculate the gradient and Hessian of the hinge loss function
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
        newton_direction = np.linalg.solve(hessian, np.concatenate(([b_grad], w_grad)))

        # Update weights and bias
        self.b -= newton_direction[0]
        self.W -= newton_direction[1:]

        # Calculate loss after update
        loss = self.hinge_loss(X, Y)
        print(f"Newton step: {loss}")

    def hinge_loss(self, X, Y):
        distance_sum = 0

        for x_i, y_i in zip(X, Y):
            distance_sum += max(0, 1 - y_i * (np.dot(self.W, x_i) + self.b))

        if self.regularization_type == "L2":
            regularization_term = 0.5 * np.dot(np.transpose(self.W), self.W)
            error_term = self.C * distance_sum
        elif self.regularization_type == "L1":
            regularization_term = self.C * np.abs(self.W)
            error_term = distance_sum
        else:
            print("No regularization type set")
            return None

        loss = regularization_term + error_term

        return loss

    def predict(self, x):
        return 1 if np.dot(self.W, x) + self.b >= 0 else -1

    def accuracy(self, X, Y):
        successes = sum(1 for i in range(len(X)) if self.predict(X[i]) == Y[i])
        accuracy = successes / len(X) * 100
        print(f"Testing accuracy: {accuracy:.2f}%")

        return accuracy
