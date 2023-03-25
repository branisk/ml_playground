from app import app
from layouts import (
    base_layout
)

app.layout = base_layout

app.run_server(debug=True, port=8051)
