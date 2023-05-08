import dash
import dash_bootstrap_components as dbc

dbc_css = "https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/lux/bootstrap.min.css"
math_jax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
scripts = [{
        'type': 'text/javascript',
         'id': 'MathJax-script',
         'src': math_jax,
}]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc_css, math_jax],
    external_scripts=scripts,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)
app.title = 'ML from Scratch'
