import dash
from dash.dependencies import Input, Output

from algorithms import *
from app import app
from datasets import *

global fig, data, model

@app.callback(
    Output('graph', 'figure'),
    Input('dataset_dropdown', 'value'),
    Input('button', 'n_clicks'),
    prevent_initial_call=True
)
def update_graph(value, button):
    global fig, data, model
    triggered_id = dash.callback_context.triggered_id

    match triggered_id:
        case 'dataset_dropdown':
            if value == "Iris":
                fig, data = gather_iris()
                return fig
            elif value == "Simulation":
                fig, data = gather_simulation()
                return fig

        case 'button':
            X = data[:, :2]
            Y = data[:, 2:3]
            hyperplane = model.fit(X, Y)

            fw = go.FigureWidget(data=fig.data + hyperplane.data, layout=fig.layout)
            return fw


@app.callback(
    Output('algorithm_dropdown', 'options'),
    Output('algorithm_dropdown', 'style'),
    Output('button', 'style'),
    Input('type_dropdown', 'value'),
    prevent_initial_call=True
)
def update_algorithms(value):
    if value == "None":
        return [''], {''}, {''}
    if value == "Classification":
        return ['Support Vector Classifier', 'Logistic Regression'], {'display': 'block'}, {'display': 'inline-block'}
    elif value == "Regression":
        return ['Linear Regression'], {'display': 'block'}, {'display': 'inline-block'}
    elif value == "Clustering":
        return ['KNearestNeighbor'], {'display': 'block'}, {'display': 'inline-block'}


@app.callback(
    Output('optimizer_dropdown', 'options'),
    Output('regularization_dropdown', 'options'),
    Output('regularization_input', 'value'),
    Output('optimizer_dropdown', 'style'),
    Output('regularization_dropdown', 'style'),
    Output('regularization_input', 'style'),
    Input('algorithm_dropdown', 'value'),
    prevent_initial_call=True
)
def update_options(value):
    global model

    if not value:
        return [''], [''], None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    elif value == "Support Vector Classifier":
        model = SupportVectorClassifier()
        return ['Sub-Gradient Descent', "Newton's Method"],\
            ["Lasso (L1)", 'Ridge (L2)'], 0.01, {'display': 'block'}, \
            {'display': 'block'}, {'display': 'block'}
    elif value == "Logistic Regression":
        model = LogisticRegression()
        return ['', ""], [""], 0.01, {'display': 'block'}, \
            {'display': 'block'}, {'display': 'block'}


@app.callback(
    Output('none1', 'style'),
    Input('optimizer_dropdown', 'value'),
    prevent_initial_call=True
)
def update_optimizer(value):
    global model

    if value:
        model.optimizer = value

@app.callback(
    Output('none2', 'style'),
    Input('regularization_dropdown', 'value'),
    prevent_initial_call=True
)
def update_regularization_type(value):
    global model

    if value:
        model.regularization_type = value


@app.callback(
    Output('none3', 'style'),
    Input('regularization_input', 'value'),
    prevent_initial_call=True
)
def update_regularization_term(value):
    global model

    if value:
        model.C = float(value)
