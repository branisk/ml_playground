import plotly.express as px
from dash import html, dcc, dash_table

base_layout = html.Div(className="container", children=[
    html.P(id='none1'),
    html.P(id='none2'),
    html.P(id='none3'),
    html.Div(className="Title center", children=[
        html.H1("Machine Learning from Scratch", className="title-text")
    ]),
    html.Div(className="Parameters center", style={"width":"80%"}, children=[
        dcc.Dropdown(['Simulation', 'Iris', 'Regression'], id='dataset_dropdown', placeholder='Dataset'),
        dcc.Dropdown(['Classification', 'Regression', 'Clustering'], id='type_dropdown', placeholder='Type'),
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
        html.Button('Step', id='button', style={'display':'none'})
    ]),
    html.Div(className="Graphs", children=[
        dcc.Graph(id='graph', figure=px.scatter().update_layout(template="plotly_dark")),
    ]),
    html.Div(className="Code", children=[
        html.H4("Code", className="center"),
        html.Div(children=[
            html.H5("Loss Function:"),
            html.Pre(html.Code(
                "def hinge_loss(self, X, Y):\n   distance_sum = 0\n   for x_i, y_i in zip(X, Y):\n      dist+=max(0, 1 - y_i * (dot(self.weight, x_i) + self.intercept))\n   reg = 0.5 * dot(self.weight.T, self.weight)\n   error_term = reg * dist\n   loss = reg + error\n   return loss",
                style={'font-size': '10px'},
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
            style_table={'height': '200px', 'overflowY': 'auto'}
        )
    ]),
    html.Div(className="Results", children=[
        html.H4("Results", className="center")
    ]),
    html.Div(className="Logo", children=[
        html.H4("Logo", className="center")
    ]),
])