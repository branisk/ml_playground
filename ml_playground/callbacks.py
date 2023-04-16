import dash
from dash.dependencies import Input, Output, State
from dash import dash_table

from algorithms import *
from app import app
from datasets import *
from metrics import *

global model
model = None


@app.callback(
    Output('graph', 'figure'),
    Output('fig-store', 'data'),
    Output('data-store', 'data'),
    Input('button', 'n_clicks'),
    Input('dataset_dropdown', 'value'),
    State('fig-store', 'data'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_results(button, dataset, fig, data):
    global model

    #  Cases based on which component triggered the callback
    match dash.callback_context.triggered_id:
        case 'dataset_dropdown':
            if dataset == "Classification":
                fig, data = gather_classification()
                return fig, fig.to_dict(), data.tolist()
            elif dataset == "Regression":
                fig, data = gather_regression()
                return fig, fig.to_dict(), data.tolist()
            elif dataset == "Clustering":
                fig, data = None, None
                return fig, fig, data

        case 'button':
            data = np.array(data)
            #  Regression is used for 2d datasets, where Classification has a 3rd column for 'label'
            if dataset != "Regression":
                X = data[:, :2]
                Y = data[:, 2:3]
            else:
                X = data[0]
                Y = data[1]

            fit_fig = model.fit(X, Y)
            fw = go.FigureWidget(data=fig['data'] + list(fit_fig['data']), layout=fig['layout'])

            return fw, fig, data

@app.callback(
    Output('algorithm_dropdown', 'options'),
    Output('algorithm_dropdown', 'style'),
    Output('button', 'style'),
    Input('dataset_dropdown', 'value'),
    prevent_initial_call=True
)
def update_algorithms(value):
    #  When the desired dataset is chosen, we need to un-hide the relevant ML algorithms
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
    Input('data-store', 'data'),
    Input('algorithm_dropdown', 'value'),
    prevent_initial_call=True
)
def update_data(dataset, data, algorithm):
    if dataset == "Classification":
        col1 = [row[0] for row in data]
        col2 = [row[1] for row in data]
        if model is not None and algorithm == "Logistic Regression":  # Check if the model is logistic regression
            col3 = [0 if row[2] == -1 else row[2] for row in data]  # Reshape the label column
        else:
            col3 = [row[2] for row in data]
        return [
            {'index': i + 1, 'X': x, 'Y': y, 'label': z}
            for i, (x, y, z) in enumerate(zip(np.round(col1, 2), np.round(col2, 2), col3))
        ]
    elif dataset == "Regression":
        return [
            {'index': i + 1, 'X': x, 'Y': y}
            for i, (x, y) in enumerate(zip(np.round(data[0], 2), np.round(data[1], 2)))
        ]
    elif dataset == "Clustering":
        return


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
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_layout(value, data):
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


@app.callback(
    Output('results-table', 'columns'),
    Output('results-table', 'data'),
    Input('dataset_dropdown', 'value'),
    Input('button', 'n_clicks'),
    prevent_initial_call=True
)
def update_results_layout(value, button):
    global model

    if not value:
        return [None] * 5

    if model is None:
        results = [None] * 5
    else:
        results = model.results

    if value == "Classification":
        columns = [{'id': 'metric', 'name': '', 'editable': False},
                   {'id': 'value', 'name': 'values', 'editable': False}]
        rows = [{'metric': metric, 'value': result} for metric, result in zip(classification_metrics, results)]
        return columns, rows

    elif value == "Regression":
        columns = [{'id': 'metric', 'name': '', 'editable': False},
                   {'id': 'value', 'name': 'values', 'editable': False}]
        rows = [{'metric': metric, 'value': result} for metric, result in zip(regression_metrics, results)]
        return columns, rows

    elif value == "Clustering":
        return [None] * 5


@app.callback(
    Output('summary-text', 'children'),
    Output('objective-text', 'children'),
    Output('assumptions-text', 'children'),
    Input('algorithm_dropdown', 'value'),
    prevent_initial_call=True
)
def update_info_layout(algorithm):
    if not algorithm:
        return 'None', 'None', 'None'
    elif algorithm == "Support Vector Classifier":
        return [values for values in soft_margin_svc.values()]
