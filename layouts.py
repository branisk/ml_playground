import plotly.express as px
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

base_layout = html.Div(className="container", children=[
    html.P(id='none'),  # Dummy component, used to update model
    dcc.Location(id="url"),
    dcc.Store(id='fig-store'),
    dcc.Store(id='data-store'),
    dcc.Store(id='model-store'),
    dcc.Store(id='results-store'),

    html.Div(className="Title center", children=[
        html.H1("Machine Learning from Scratch", className="title-text")
    ]),

    html.Div(className="Options center", style={"width": "80%"}, children=[
        dcc.Dropdown(['Classification', 'Regression', 'Clustering', 'Dimensionality Reduction'], id='dataset_dropdown', placeholder='Dataset', className="dropdown"),
        dcc.Dropdown([''], id='classification_dropdown', placeholder='Dataset', className="dropdown"),
        dcc.Dropdown([''], id='algorithm_dropdown', placeholder='Algorithm', className="dropdown"),
        dcc.Dropdown([''], id='regression_method_dropdown', placeholder='Method', className="dropdown"),
        dcc.Dropdown([''], id='optimizer_dropdown', placeholder='Optimizer', className="dropdown"),
        html.Div([
            dcc.Dropdown([''], id='regularization_dropdown', placeholder='Regularization Type', className="inline dropdown"),
            dcc.Input('0.01', id="regularization_input", className="regularization_input"),
            ],
            className="input-group-append",
            id="regularization_group"
        ),
        html.Div(id="step", children=[
            html.Button(id='step_button', className="inline", children="Step"),
            dcc.Input(id='step_input', value=1, className="inline", style={"width":"5em", "text-align":"right"}),
        ],
        className="input-group-append",
        style={'display':'none'}
        ),
        html.Br(),
        html.Button('Reset', id='reset_button', style={'display':'none'})
    ]),

    html.Div(className="Graphs", children=[
        dcc.Tabs(id="graph-tabs", children=[
            dcc.Tab(label="Primary Data", id="data-tab", className="tabs", children=[
                dcc.Graph(id='main-graph', figure=px.scatter().update_layout(template="plotly_dark", margin=dict(l=50, r=30, t=30, b=50), xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))),
            ]),
            dcc.Tab(label="Residuals", id="residuals-tab", className="tabs hidden", children=[
                dcc.Graph(id='residual-graph', figure=px.scatter().update_layout(template="plotly_dark")),
            ]),
            dcc.Tab(label="Distribution", id="residual-distribution-tab", className="tabs hidden", children=[
                dcc.Graph(id='residual-distribution-graph', figure=px.scatter().update_layout(template="plotly_dark")),
            ]),
        ]),
    ]),

    html.Div(className="Info", children=[
        html.Div(children=[
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.P("Please choose an algorithm.", id="summary-text", className='accordion-text')
                        ],
                        title="Summary",
                    ),
                    dbc.AccordionItem(
                        id="objective",
                        children=[
                            html.P("Please choose an algorithm.", id="objective-text", className='accordion-text')
                        ],
                        title="Objective",
                    ),
                    dbc.AccordionItem(
                        id="complexity",
                        children=[
                            html.P("Please choose an algorithm.", id="complexity-text", className='accordion-text')
                        ],
                        title="Complexity",
                    ),
                    dbc.AccordionItem(
                        [
                            html.P("Please choose an algorithm.", id="assumptions-text", className='accordion-text')
                        ],
                        title="Assumptions",
                    ),
                    dbc.AccordionItem(
                        [
                            html.A(id="info-text", className='accordion-text'),
                        ],
                        title="More Information",
                    )
                ],
                start_collapsed=True,
                id='info-accordion'  # Add this line
            ),
        ],
            className=""
        ),
    ]),

    html.Div(className="Data", children=[
        html.H4("Data", id="data-text", className="text-center"),
        dcc.Tabs(id="train-test-tabs", children=[
            dcc.Tab(label="Train", className="tabs", children=[
                dash_table.DataTable(
                    id="train_table",
                    style_table={'height': '200px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#343a40', 'fontWeight': 'bold', 'color': 'white'},
                    style_cell={'backgroundColor': '#343a40', 'color': 'white'},
                ),
            ]),
            dcc.Tab(label="Test", className="tabs", children=[
                dash_table.DataTable(
                    id="test_table",
                    style_table={'height': '120px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#343a40', 'fontWeight': 'bold', 'color': 'white'},
                    style_cell={'backgroundColor': '#343a40', 'color': 'white'},
                ),
            ]),
        ]),
    ]),

    html.Div(className="Results", children=[
        html.H4("Results", id="results-text", className=""),
        dash_table.DataTable(
            id="results-table",
            style_table={'width': '80%', 'max-width': '80%'},
            style_header={'backgroundColor': '#343a40', 'fontWeight': 'bold', 'color': 'white'},
            style_cell={'backgroundColor': '#343a40', 'color': 'white'},
            data=[{'metric': '', 'values': ''}]
        )
    ]),
])
