import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.datasets import make_blobs, make_regression


def gather_regression():
    X, Y = make_regression(n_samples=100, n_features=2, effective_rank=.1, random_state=42)
    Y = X[:, 1]
    X = X[:, 0]

    fig = go.FigureWidget(px.scatter(x=X, y=Y))

    fig.update_layout(
        title='Simulation Dataset',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        template="plotly_dark"
    )

    data = np.vstack((X, Y))

    print(data)

    return fig, data


def gather_simulation():
    X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
    Y = np.where(Y == 0, -1, Y)

    fig = go.FigureWidget(px.scatter(x=X[:, 0], y=X[:, 1], color=Y))

    fig.update_layout(
        title='Simulation Dataset',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        template="plotly_dark"
    )

    data = np.hstack((X, Y.reshape(-1, 1)))

    return fig, data


def gather_iris():

    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data)
    data['label'] = iris.target
    data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'label']
    data.dropna(how="all", inplace=True) # remove any empty lines
    data.label.replace({0: 1, 1: 2, 2: 3}, inplace=True)

    data = data[data['label'] != 2]
    remap = {1: 1, 3: -1}

    data = data.replace({'label': remap})
    data = data.drop(columns=['petal_len', 'petal_wid'])

    fig = go.FigureWidget(px.scatter(data, x="sepal_len", y="sepal_wid", color="label"))

    fig.update_layout(
        title='Iris Dataset Simplified',
        xaxis=dict(title='Sepal Length'),
        yaxis=dict(title='Sepal Width'),
        template="plotly_dark"
    )

    return fig, np.array(data)

def plot_data():
    return