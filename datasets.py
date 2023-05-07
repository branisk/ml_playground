import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split


def gather_regression():
    X, Y = make_regression(n_samples=500, n_features=2, effective_rank=.04, random_state=42)
    Y = X[:, 1]
    X = X[:, 0]
    minx = min(X)
    miny = min(Y)
    X = (X-minx) * 100
    Y = (Y-miny) * 100
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    fig = go.FigureWidget()

    scatter1 = px.scatter(
        x=X_train,
        y=Y_train
    )
    scatter1.data[0]['showlegend'] = True
    scatter1.data[0]['name'] = 'Train'

    fig.add_traces(scatter1.data)

    scatter2 = px.scatter(
        x=X_test,
        y=Y_test,
        labels={'x': 'Test', 'y': 'Test'}
    ).update_traces(dict(marker_line_width=1, marker_line_color="white"))
    scatter2.data[0]['showlegend'] = True
    scatter2.data[0]['name'] = 'Test'

    fig.add_traces(scatter2.data)

    fig.update_layout(
        title='Regression Dataset (Linear)',
        xaxis=dict(title='Independent Variable', showgrid=False, zeroline=False),
        yaxis=dict(title='Dependent Variable', showgrid=False, zeroline=False),
        template="plotly_dark",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=20, t=50, b=20)
    )

    data = np.vstack((X, Y))

    return fig, data


def gather_classification():
    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.90)
    Y = np.where(Y == 0, -1, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scatter1 = px.scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        color= np.where(Y_train == 1, "-1 Train", "+1 Train").astype(str),
        color_discrete_map={
            "-1 Train": "orange",
            "+1 Train": "royalblue"
        }
    )

    fig = go.FigureWidget(scatter1)

    scatter2 = px.scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        color=np.where(Y_test == 1, "+1 Test", "-1 Test").astype(str),
        color_discrete_map={
            "+1 Test": "orange",
            "-1 Test": "blue"
        }
    ).update_traces(dict(marker_line_width=1, marker_line_color="white"))

    fig.add_traces(scatter2.data)

    fig.update_layout(
        title='Classification Dataset',
        xaxis=dict(title='X_0 Independent Variable', showgrid=False, zeroline=False),
        yaxis=dict(title='X_1 Independent Variable', showgrid=False, zeroline=False),
        template="plotly_dark",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=20, t=50, b=20),
    )

    data = np.hstack((X, Y.reshape(-1, 1)))

    return fig, data


def gather_clustering():
    X, Y = make_blobs(n_samples=200, centers=4, random_state=0, cluster_std=1)

    fig = go.FigureWidget(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                     marker=dict(color='grey', size=6)))

    fig.update_layout(
        title='Clustering Dataset',
        xaxis=dict(title='X_0 Independent Variable', showgrid=False, zeroline=False),
        yaxis=dict(title='X_1 Independent Variable', showgrid=False, zeroline=False),
        template="plotly_dark"
    )

    data = np.hstack((X, Y.reshape(-1, 1)))

    return fig, data

    
def gather_dimensionalityreduction():
    X, Y = make_regression(n_samples=50, n_features=3, effective_rank=.1, random_state=42)
    data = pd.DataFrame(X, columns=['X', 'Y', 'Z'])
    data = data - data.min()

    fig = go.FigureWidget()

    scatter1 = go.Scatter3d(
        x=data['X'],
        y=data['Y'],
        z=data['Z'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['Z'],
            colorscale='Plasma',
            opacity=0.8,
            showscale=False  # Hide the color scale legend
        ),
        name='3D'
    )

    fig.add_trace(scatter1)

    fig.update_layout(
        title='Dimensionality Reduction',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            bgcolor="rgba(0,0,0,0.1)",
            xaxis=dict(
                gridcolor='grey',
                showbackground=True
            ),
            yaxis=dict(
                gridcolor='grey',
                showbackground=True
            ),
            zaxis=dict(
                gridcolor='grey',
                showbackground=True
            )
        ),
        template="plotly_dark",
        margin=dict(l=0, r=20, t=50, b=20)
    )

    return fig, data
