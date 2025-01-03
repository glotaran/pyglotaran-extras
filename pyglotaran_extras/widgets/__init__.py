from __future__ import annotations

try:
    import anywidget
    import networkx
    import traitlets

except ImportError:
    msg = (
        "Widget dependencies were not found, install `pyglotaran-extras` with the `widgets` extra "
        '(e.g. `uv pip install "pyglotaran-extras[widgets]"`).'
    )
    raise ImportError(msg) from None

from pyglotaran_extras.widgets.kineticschemevisualizer.visualizer import visualize_dataset_model
from pyglotaran_extras.widgets.kineticschemevisualizer.visualizer import visualize_megacomplex
