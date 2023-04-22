from layouts import (
    base_layout
)
from callbacks import *

app.layout = base_layout
app.run_server(debug=False, port=8050)
