import plotly.express as px
import plotly.graph_objects as go

from metrics import *


#  Regression Methods
class LinearRegression:
    def __init__(
            self,
            n_features=1,
            max_iterations=1,
            method="OLS"
    ):
        self.max_iter = max_iterations
        self.method = method
        self.weights = np.zeros(n_features + 1)  # Weight term
        self.results = [None] * 5

    def fit(self, X, Y):
        '''
        Projection Matrix P = X * (X.T*X)**(-1) * X.T
        Annihilator Matrix M = I_n - P
        residuals = M * Y
        '''
        n = len(X)

        if self.method == "OLS":
            sx = np.sum(X)
            sy = np.sum(Y)
            sxx = np.dot(X.T, X)
            syy = np.dot(Y.T, Y)
            sxy = np.dot(X.T, Y)

            self.weights[1] = (n*sxy - sx*sy) / (n*sxx - sx*sx)
            self.weights[0] = (sy/n - self.weights[1]*sx/n)
        elif self.method == "MLE":
            print("MLE")
            # Add a column of ones to the X matrix to include the intercept term
            X_augmented = np.column_stack((np.ones(n), X))

            # MLE estimation for linear regression with Gaussian likelihood is equivalent to OLS
            self.weights = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ Y
        else:
            print("No method chosen.")

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

        scatter = px.scatter(x=list(range(len(residuals))), y=residuals, title='Residuals Plot')

        return scatter

    def plot_gaussian_likelihood(self, X, Y):
        Y_pred = self.predict(X)
        residuals = Y.T[0] - Y_pred

        # Estimate Gaussian likelihood parameters
        mu = np.mean(residuals)
        sigma = np.std(residuals)

        # Create a range of values for the likelihood function
        x_values = np.linspace(-3*sigma, 3*sigma, 100)

        # Calculate the Gaussian likelihood
        likelihood = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x_values - mu)**2 / (2 * sigma**2))

        scatter = px.scatter(x=x_values, y=likelihood)

        return scatter

    def update_results(self, X, Y):
        Y_pred = self.predict(X)
        self.results = [
            f'{round(self.weights[1], 2)}x + {round(self.weights[0], 2)}',
            str(round(r2_score(Y_pred, Y)*100, 2)) + "%",
            round(rmse(Y_pred, Y), 5),
            round(mse(Y_pred, Y), 5),
            round(mae(Y_pred, Y), 5),
            round(reduced_chi_squared(Y_pred, Y), 5),
            round(rse(Y_pred, Y), 5),
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
    def __init__(self, n_clusters=5, max_iterations=1, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.results = None

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        centroid_indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids = X[centroid_indices]

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X, Y):
        if self.centroids is None:
            self._initialize_centroids(X)
            prev_centroids = None
        else:
            prev_centroids = self.centroids

        for _ in range(self.max_iterations):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            if np.array_equal(new_centroids, prev_centroids):
                print("Cannot improve, stopping")
                break

            prev_centroids = new_centroids
            self.centroids = new_centroids

        return self

    def plot_centroids(self, X, Y):
        if self.centroids is not None:
            labels = self._assign_clusters(X)

            # Create a scatter plot for each cluster
            scatter_list = []
            for cluster in range(self.n_clusters):
                indices = np.where(labels == cluster)[0]
                scatter_list.append(go.Scatter(
                    x=X[indices, 0], y=X[indices, 1], mode='markers',
                    name=f"Cluster {cluster}",
                    marker=dict(size=6, color=px.colors.qualitative.Plotly[cluster])
                ))
                scatter_list.append(self.plot_ellipses(X[indices], cluster))

            # Add the centroids to the scatter list
            scatter1 = go.Scatter(x=self.centroids[:, 0], y=self.centroids[:, 1],
                                  mode='markers',
                                  name='Centroids',
                                  marker=dict(color='rgba(255, 0, 0, 0.8)',
                                              line=dict(color='rgba(255, 255, 255, 1)', width=2),
                                              size=10))
            scatter_list.append(scatter1)

            fig = go.Figure(data=scatter_list)

            # Update the layout for a more professional appearance
            fig.update_layout(
                title="K-means Clustering",
                title_x=0.5,
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                font=dict(size=14),
                plot_bgcolor="white",
                xaxis=dict(gridcolor='rgba(180, 180, 180, 0.3)'),
                yaxis=dict(gridcolor='rgba(180, 180, 180, 0.3)')
            )

            return fig
        else:
            print("Centroids have not been computed yet.")

    def update_results(self, X, Y):
        pass

    def predict(self, X):
        return self._assign_clusters(X)

    def fit_predict(self, X, Y):
        self.fit(X)
        return self.predict(X)

    def plot_ellipses(self, X, cluster):
        if len(X) < 2:
            return go.Scatter()

        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        sqrt_eigvals = np.sqrt(eigvals)

        eigvecs *= 2

        # Compute the angle of rotation from the eigenvectors
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        # Compute coordinates for the ellipse
        t = np.linspace(0, 2 * np.pi, 100)
        coord_array = np.array([sqrt_eigvals[0] * np.cos(t), sqrt_eigvals[1] * np.sin(t)])
        coords = np.dot(eigvecs, coord_array).T + np.mean(X, axis=0)

        return go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='lines', name=f'Ellipse {cluster}')


class OrthogonalProjection:
    def __init__(self, normal_vector=np.array([0,0,1])):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.mean = None
        self.results = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Project the centered points onto the 2D plane
        self.results = self.orthogonal_project(X_centered)

    def orthogonal_project(self, X):
        distance_to_plane = np.dot(X, self.normal_vector)
        return X - np.outer(distance_to_plane, self.normal_vector)

    def transform(self, X):
        X_centered = X - self.mean
        return self.orthogonal_project(X_centered)

    def distance_to_plane(self, X):
        X_centered = X - self.mean
        return np.abs(np.dot(X_centered, self.normal_vector))

    def plot(self, data):
        X_2D = self.transform(data) + self.mean  # Add the mean back to the projected points
        Z = self.distance_to_plane(data)

        # Create scatter plot of the projected 2D points
        scatter = go.Scatter3d(x=X_2D[:, 0], y=X_2D[:, 1], z=X_2D[:, 2],
                               mode='markers',
                               marker=dict(size=4, color='red'),
                               name='Projected Points')

        # Create line segments connecting the original 3D points to the projected 2D points
        lines = []
        for p1, p2 in zip(data, X_2D):
            lines.extend([go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                       mode='lines',
                                       line=dict(color='white', dash='dash'),
                                       showlegend=False)])

        fig = go.Figure(data=[scatter, *lines])
        return fig


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.results = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        eigenvectors = eigenvectors.T

        # Sort the eigenvectors based on the descending order of their corresponding eigenvalues
        sorted_components_idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[sorted_components_idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components) + self.mean

    def plot(self, data):
        X_transformed = self.transform(data)
        X_projected = self.inverse_transform(X_transformed)

        # Create scatter plot of the projected points
        scatter = go.Scatter3d(x=X_projected[:, 0], y=X_projected[:, 1], z=X_projected[:, 2],
                               mode='markers',
                               marker=dict(size=4, color='red'),
                               name='Projected Points')

        # Create line segments connecting the original 3D points to the projected points
        lines = []
        for p1, p2 in zip(data, X_projected):
            lines.extend([go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                                       mode='lines',
                                       line=dict(color='white', dash='dash'),
                                       showlegend=False)])

        fig = go.Figure(data=[scatter, *lines])
        return fig
