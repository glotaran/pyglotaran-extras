"""Kinetic Scheme Visualizer widget module."""

from __future__ import annotations

import pathlib
from typing import Any

import anywidget
import traitlets


class GraphWidget(anywidget.AnyWidget):
    """Widget for editing kinetic scheme graphs."""

    _esm: pathlib.Path = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css: pathlib.Path = pathlib.Path(__file__).parent / "static" / "widget.css"
    graph_data = traitlets.Dict().tag(sync=True)
    visualization_options = traitlets.Dict().tag(sync=True)
    cy_json = traitlets.Dict({}).tag(sync=True)
    height = traitlets.Int(default_value=600).tag(sync=True)

    def __init__(
        self,
        graph_data: dict[str, Any] = {},
        visualization_options: dict[str, Any] = {},
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.graph_data = graph_data
        self.visualization_options = visualization_options
