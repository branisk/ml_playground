from layouts import (
    base_layout
)
from callbacks import *

app.layout = base_layout
app.run_server(debug=True, port=8051)
