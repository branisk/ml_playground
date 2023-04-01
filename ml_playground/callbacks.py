import dash
from dash.dependencies import Input, Output

from algorithms import *
from app import app
from datasets import *

global fig, data, model
fig, data, model = None, None, None


@app.callback(
    Output('graph', 'figure'),
    Input('dataset_dropdown', 'value'),
    Input('button', 'n_clicks'),
    prevent_initial_call=True
)
def update_results(value, button):
    global fig, data, model

    #  Cases based on which component triggered the callback
    #  Note: We are only allowed one callback per output, so we must combine the dataset_dropdown and training button
    match dash.callback_context.triggered_id:
        case 'dataset_dropdown':
            if value == "Classification":
                fig, data = gather_classification()
                return fig
            elif value == "Regression":
                fig, data = gather_regression()
                return fig
            elif value == "Clustering":
                return fig

        case 'button':
            if value != "Regression":
                X = data[:, :2]
                Y = data[:, 2:3]
            else:
                X = data[0]
                Y = data[1]

            fit_fig = model.fit(X, Y)
            fw = go.FigureWidget(data=fig.data + fit_fig.data, layout=fig.layout)

            return fw

@app.callback(
    Output('algorithm_dropdown', 'options'),
    Output('algorithm_dropdown', 'style'),
    Output('button', 'style'),
    Input('dataset_dropdown', 'value'),
    prevent_initial_call=True
)
def update_algorithms(value):
    if value == "None":
        return [''], {''}, {''}
    elif value == "Classification":
        return ['Support Vector Classifier', 'Logistic Regression'], {'display': 'block'}, {'display': 'inline-block'}
    elif value == "Regression":
        return ['Linear Regression'], {'display': 'block'}, {'display': 'inline-block'}
    elif value == "Clustering":
        return ['KNearestNeighbors'], {'display': 'block'}, {'display': 'inline-block'}


@app.callback(
    Output('table', 'data'),
    Input('dataset_dropdown', 'value'),
    prevent_initial_call=True
)
def update_table(value):
    global data

    if data is None:
        return None

    return [
        {'index': i+1, 'X': x, 'Y': y}
        for i, (x, y) in enumerate(zip(np.round(data[0], 2), np.round(data[1], 2)))
    ]


@app.callback(
    Output('optimizer_dropdown', 'options'),
    Output('regularization_dropdown', 'options'),
    Output('regularization_input', 'value'),
    Output('optimizer_dropdown', 'style'),
    Output('regularization_dropdown', 'style'),
    Output('regularization_input', 'style'),
    Output('regression_method_dropdown', 'style'),
    Output('regression_method_dropdown', 'options'),
    Input('algorithm_dropdown', 'value'),
    prevent_initial_call=True
)
def update_layout(value):
    global model

    if not value:
        return [], [], None, {}, {}, {}, {}, []

    elif value == "Support Vector Classifier":
        model = SupportVectorClassifier()
        return ['Sub-Gradient Descent', "Newton's Method"],\
            ["Lasso (L1)", 'Ridge (L2)'], 0.01, {'display': 'block'}, \
            {'display': 'block'}, {'display': 'block'}, {}, []

    elif value == "Logistic Regression":
        model = LogisticRegression()
        return ['Sub-Gradient Descent'], ['Lasso (L1)'], 0.01, {'display': 'block'}, \
            {'display': 'block'}, {'display': 'block'}, {}, []

    elif value == "Linear Regression":
        model = LinearRegression()
        return ['', ""], [""], None, {}, \
            {}, {}, {'display': 'block'}, ["OLS"]


@app.callback(
    Output('none', 'style'),
    Input('optimizer_dropdown', 'value'),
    Input('regularization_dropdown', 'value'),
    Input('regularization_input', 'value'),
    prevent_initial_call=True
)
def update_values(optimizer, regularization_type, regularization_value):
    global model

    if model is None:
        return

    #  Cases based on which component triggered the callback
    match dash.callback_context.triggered_id:
        case 'optimizer_dropdown':
            model.optimizer = optimizer
        case 'regularization_dropdown':
            model.regularization_type = regularization_type
        case 'regularization_value':
            model.C = float(regularization_value)
