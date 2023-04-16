import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split


def gather_regression():
    X, Y = make_regression(n_samples=50, n_features=2, effective_rank=.05, random_state=42)
    Y = X[:, 1]
    X = X[:, 0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    fig = go.FigureWidget(px.scatter(x=X_train, y=Y_train))

    scatter2 = px.scatter(x=X_test, y=Y_test).update_traces(dict(marker_line_width=1, marker_line_color="white"))

    fig.add_traces(scatter2.data)

    fig.update_layout(
        title='Regression Dataset',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        template="plotly_dark",
    )

    data = np.vstack((X, Y))

    return fig, data


def gather_classification():
    X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
    Y = np.where(Y == 0, -1, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    fig = go.FigureWidget(px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=Y_train.astype(str)))

    scatter2 = px.scatter(x=X_test[:, 0], y=X_test[:, 1], color=Y_test.astype(str)).update_traces(dict(marker_line_width=1, marker_line_color="white"))

    fig.add_traces(scatter2.data)

    fig.update_layout(
        title='Classification Dataset',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        template="plotly_dark"
    )

    data = np.hstack((X, Y.reshape(-1, 1)))

    return fig, data


def gather_clustering():
    X, Y = make_blobs(n_samples=50, centers=4, random_state=0, cluster_std=0.30)
    Y = np.where(Y == 0, -1, Y)

    fig = go.FigureWidget(px.scatter(x=X[:, 0], y=X[:, 1], color=Y))

    fig.update_layout(
        title='Clustering Dataset',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        template="plotly_dark"
    )

    data = np.hstack((X, Y.reshape(-1, 1)))

    return fig, data
