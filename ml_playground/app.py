import dash
import dash_bootstrap_components as dbc

dbc_css = "https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/lux/bootstrap.min.css"
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc_css],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
app.title = 'ML from Scratch'
