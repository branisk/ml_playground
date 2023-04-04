import plotly.express as px
from dash import html, dcc, dash_table, callback_context

base_layout = html.Div(className="container", children=[
    html.P(id='none'),  # Dummy component, used to update model
    dcc.Store(id='fig-store'),
    dcc.Store(id='data-store'),
    dcc.Store(id='model-store'),
    html.Div(className="Title center", children=[
        html.H1("Machine Learning from Scratch", className="title-text")
    ]),

    html.Div(className="Options center", style={"width":"80%"}, children=[
        dcc.Dropdown(['Classification', 'Regression', 'Clustering'], id='dataset_dropdown', placeholder='Dataset'),
        dcc.Dropdown([''], id='algorithm_dropdown', placeholder='Algorithm'),
        dcc.Dropdown([''], id='regression_method_dropdown', placeholder='Method'),
        dcc.Dropdown([''], id='optimizer_dropdown', placeholder='Optimizer'),
        html.Div([
            dcc.Dropdown([''], id='regularization_dropdown', placeholder='Regularization Type', className="inline"),
            dcc.Input('0.01', id="regularization_input", className="regularization_input"),
            ],
            className="input-group-append",
            id="regularization_group"
        ),
        html.Button('Step', id='button', style={'display': 'none'})
    ]),

    html.Div(className="Graphs", children=[
        dcc.Graph(id='graph', figure=px.scatter().update_layout(template="plotly_dark")),
    ]),

    html.Div(className="Code", children=[
        html.H4("Code", className="center"),
        html.Div(children=[
            html.H5("Loss Function:"),
            html.Pre(html.Code(
                "TODO"
            ))],
            className="code-holder"
        ),
    ]),

    html.Div(className="Data", children=[
        html.H4("Data", className="center"),
        dash_table.DataTable(
            id="table",
            columns=(
                [{'id': 'index', 'name': '', 'editable': False},
                 {'id': 'X', 'name': 'X', 'editable': False},
                 {'id': 'Y', 'name': 'Y', 'editable': False}]
            ),
            style_table={'height': '180px', 'overflowY': 'auto'}
        )
    ]),

    html.Div(className="Results", children=[
        html.H4("Results", id="results", className="center"),
        dash_table.DataTable(
            id="results-table",
            style_table={'width':'80%', 'padding':'10px'}
        )
    ]),

    html.Div(className="Logo", children=[
        html.H4("Logo", className="center")
    ]),
])