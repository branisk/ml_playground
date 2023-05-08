import dash
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split

from algorithms import *
from app import app
from datasets import *
from metrics import *
from layouts import *

import pickle
import base64

def serialize_model(model):
    return base64.b64encode(pickle.dumps(model)).decode('utf-8')

def deserialize_model(model_data):
    return pickle.loads(base64.b64decode(model_data.encode('utf-8')))


@app.callback(
    Output('main-graph', 'figure'),
    Output('fig-store', 'data'),
    Output('data-store', 'data'),
    Output('residual-graph', 'figure'),
    Output('gaussian-likelihood-graph', 'figure'),
    Output('model-store', 'data'),
    Output('results-store', 'data'),
    Input('step_button', 'n_clicks'),
    Input('step_input', 'value'),
    Input('dataset_dropdown', 'value'),
    Input('algorithm_dropdown', 'value'),
    State('fig-store', 'data'),
    State('data-store', 'data'),
    State('model-store', 'data'),
    prevent_initial_call=True
)
def update_results(step_button, step_input, dataset, algorithm, fig, data, model_data):
    #  Cases based on which component triggered the callback
    match dash.callback_context.triggered_id:
        case 'dataset_dropdown' | 'algorithm_dropdown':
            if not algorithm: # The callback is for the dataset
                if dataset == "Classification":
                    fig, data = gather_classification()
                elif dataset == "Regression":
                    fig, data = gather_regression()
                elif dataset == "Clustering":
                    fig, data = gather_clustering()
                elif dataset == "Dimensionality Reduction":
                    fig, data = gather_dimensionalityreduction()
                return fig.to_dict(), fig.to_dict(), pd.DataFrame(data).values.tolist(), None, None, None, None
            else: # The callback is for the algorithm, so intiialize the algorithm
                if algorithm == "Support Vector Classifier":
                    model = SupportVectorClassifier()
                elif algorithm == "Logistic Regression":
                    model = LogisticRegression()
                elif algorithm == "Linear Regression":
                    model = LinearRegression()
                elif algorithm == "KMeans":
                    model = KMeans()
                elif algorithm == "PCA":
                    model = PCA()
                elif algorithm == "OrthogonalProjection":
                    model = OrthogonalProjection()
                return fig, fig, data, {}, {}, serialize_model(model), None

        case 'step_button':
            data = np.array(data)
            #  Regression is used for 2d datasets, where Classification has a 3rd column for 'label'
            if dataset != "Regression":
                X = data[:, :2]
                Y = data[:, 2:3]
            else:
                X = data[0]
                Y = data[1]

            # Split the data into train and test sets
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            if model_data and dataset != "Dimensionality Reduction":
                model = deserialize_model(model_data)

                model.max_iter = int(step_input)

                # Fit the model on the train set
                model.fit(X_train, Y_train)

                # Update the results using the test set
                model.update_results(X_test, Y_test)
            else:
                model = None

            if dataset == "Regression":
                fit_fig = model.plot_best_fit(X, X_test, Y_test)
                fw = go.FigureWidget(data=fig['data'] + list(fit_fig['data']), layout=fig['layout'])

                res = model.plot_residuals(X_test, Y_test)
                resfw = go.Figure(data=res['data'], layout=fig['layout'])
                resfw.layout.update(dict(title="Residuals <br><sup>The difference between the actual y value and the predicted y values</sup>",
                    xaxis_title="Index",
                    yaxis_title="Residual"

                ))
                lh = model.plot_gaussian_likelihood(X_test, Y_test)
                lhfw = go.Figure(data=lh['data'], layout=fig['layout'])
                lhfw.layout.update(dict(
                    title="Maximum Gaussian Likelihood <br><sup>The distribution of the residuals, assumed to be normal</sup>",
                    xaxis_title="Residual Values",
                    yaxis_title="Likelihood"

                ))
            elif dataset == "Classification":
                fit_fig = model.plot_hyperplane(X)
                fw = go.Figure(data=fig['data'] + list(fit_fig['data']), layout=fig['layout'])
                resfw = None
                lhfw = None
            if dataset == "Clustering": # Adding the clustering case
                fit_fig = model.plot(X, Y)
                fw = go.Figure(data=fig['data'] + list(fit_fig['data']), layout=fig['layout'])
                resfw = None
                lhfw = None
            elif dataset == "Dimensionality Reduction":
                model = deserialize_model(model_data)
                # Fit the model on the train set
                model.fit(data)
                fit_fig = model.plot(data)
                fw = go.Figure(data=fig['data'] + list(fit_fig['data']), layout=fig['layout'])
                resfw = None
                lhfw = None

            return fw, fig, data, resfw, lhfw, serialize_model(model), True

        case 'step_input':
            return [dash.no_update,] * 7

@app.callback(
    Output('algorithm_dropdown', 'options'),
    Output('algorithm_dropdown', 'style'),
    Output('step', 'style'),
    Output('reset_button', 'style'),
    Input('dataset_dropdown', 'value'),
    prevent_initial_call=True
)
def update_algorithms(value):
    #  When the desired dataset is chosen, we need to un-hide the relevant ML algorithms
    if value == "None":
        return [''], {''}, {''}, {''}
    elif value == "Classification":
        return ['Support Vector Classifier', 'Logistic Regression'], {'display': 'block'}, {'display': 'inline-block'}, {'display': 'inline-block'}
    elif value == "Regression":
        return ['Linear Regression'], {'display': 'block'}, {'display': 'inline-block'}, {'display': 'inline-block'}
    elif value == "Clustering":
        return ['KMeans'], {'display': 'block'}, {'display': 'inline-block'}, {'display': 'inline-block'}
    elif value == "Dimensionality Reduction":
        return ['OrthogonalProjection', 'PCA', 'SVD', 'LDA'], {'display': 'block'}, {'display': 'inline-block'}, {'display': 'inline-block'}


@app.callback(
    Output('train_table', 'data'),
    Output('test_table', 'data'),
    Input('dataset_dropdown', 'value'),
    Input('data-store', 'data'),
    Input('algorithm_dropdown', 'value'),
    State('model-store', 'data'),
    prevent_initial_call=True
)
def update_data(dataset, data, algorithm, model_data):
    if model_data:
        model = deserialize_model(model_data)
    else:
        model = None

    df = pd.DataFrame(data)

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
        ], None
    elif dataset == "Regression":
        return [
            {'index': i + 1, 'X': x, 'Y': y}
            for i, (x, y) in enumerate(zip(np.round(data[0], 2), np.round(data[1], 2)))
        ], None
    elif dataset == "Clustering":
        return df.round({0: 2, 1: 2, 2: 3}).rename(columns={0: 'X', 1: 'Y', 2: 'Z'}).reset_index().to_dict('records'), None
    elif dataset == "Dimensionality Reduction":
        return df.round({0: 2, 1: 2, 2: 3}).rename(columns={0: 'X', 1: 'Y', 2: 'Z'}).reset_index().to_dict('records'), None

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
    State('model-store', 'data'),
    prevent_initial_call=True
)
def update_layout(value, data, model_data):
    if not value:
        return [], [], None, {}, {}, {}, {}, []

    elif value == "Support Vector Classifier":
        return ['Sub-Gradient Descent', "Newton's Method"],\
            ["Lasso (L1)", 'Ridge (L2)'], 0.01, {'display': 'block'}, \
            {'display': 'block'}, {'display': 'block'}, {}, []

    elif value == "Logistic Regression":
        return ['Sub-Gradient Descent'], \
            ['Lasso (L1)'], 0.01, {'display': 'block'}, \
            {'display': 'block'}, {'display': 'block'}, {}, []

    elif value == "Linear Regression":
        return ['', ''], [''], None, {}, \
            {}, {}, {'display': 'block'}, ['OLS', 'MLE']

    elif value == "KMeans":
        return [], [], None, {}, {}, {}, {}, []

    elif value == "PCA" or "OrthogonalProjection":
        return [], [], None, {}, {}, {}, {}, []


@app.callback(
    Output('none', 'style'),
    Input('optimizer_dropdown', 'value'),
    Input('regularization_dropdown', 'value'),
    Input('regularization_input', 'value'),
    Input('regression_method_dropdown', 'value'),
    State('model-store', 'data'),
    prevent_initial_call=True
)
def update_hyperparameters(optimizer, regularization_type, regularization_value, method, model_data):
    if model_data:
        model = deserialize_model(model_data)
    else:
        return

    #  Cases based on which component triggered the callback
    match dash.callback_context.triggered_id:
        case 'optimizer_dropdown':
            model.optimizer = optimizer
        case 'regularization_dropdown':
            model.regularization_type = regularization_type
        case 'regularization_input':
            model.C = float(regularization_value)
        case 'regression_method_dropdown':
            model.method = method

@app.callback(
    Output('results-table', 'columns'),
    Output('results-table', 'data'),
    Input('dataset_dropdown', 'value'),
    Input('results-store', 'data'),
    State('model-store', 'data'),
    prevent_initial_call=True
)
def update_results_layout(value, results, model_data):
    if model_data:
        model = deserialize_model(model_data)
    else:
        model = None

    if not value or model is None or results is None:
        columns = [{'id': 'metric', 'name': 'metric', 'editable': False},
                   {'id': 'value', 'name': 'values', 'editable': False}]
        rows = []
        return columns, rows

    results = model.results

    if value == "Classification":
        columns = [{'id': 'metric', 'name': 'metric', 'editable': False},
                   {'id': 'value', 'name': 'values', 'editable': False}]
        rows = [{'metric': metric, 'value': result} for metric, result in zip(classification_metrics, results)]
        return columns, rows

    elif value == "Regression":
        columns = [{'id': 'metric', 'name': 'metric', 'editable': False},
                   {'id': 'value', 'name': 'values', 'editable': False}]
        rows = [{'metric': metric, 'value': result} for metric, result in zip(regression_metrics, results)]
        return columns, rows

    elif value == "Clustering":
        return [None] * 2

    elif value == "Dimensionality Reduction" or "OrthogonalProjection":
        return [None] * 2


@app.callback(
    Output('summary-text', 'children'),
    Output('objective-text', 'children'),
    Output('assumptions-text', 'children'),
    Output('complexity-text', 'children'),
    Output('info-text', 'children'),
    Output('info-text', 'href'),
    Input('algorithm_dropdown', 'value'),
    prevent_initial_call=True
)
def update_info_layout(algorithm):
    if not algorithm:
        return 'None', 'None', 'None', 'None', 'None', None
    elif algorithm == "Support Vector Classifier":
        return [values for values in soft_margin_svc.values()]
    elif algorithm == "Logistic Regression":
        return [values for values in logistic_regression.values()]
    elif algorithm == "Linear Regression":
        return [values for values in linear_regression.values()]
    elif algorithm == "PCA" or "OrthogonalProjection":
        return 'None', 'None', 'None', 'None', 'None', None


@app.callback(
    Output('reset_button', 'n_clicks'),
    Input('reset_button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_layout(n_clicks):
    print("RESETTING DASHBOARD")
    app.layout = base_layout
    return 0
