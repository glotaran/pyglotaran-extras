"""Named constants for the kinetic scheme visualizer."""

from __future__ import annotations

GROUND_STATE_PREFIX: str = "GS"
"""Prefix used when generating synthetic ground state node labels."""

PS_INVERSE_TO_NS_INVERSE: float = 1e3
"""Conversion factor from ps⁻¹ to ns⁻¹."""

# Default node dimensions (in data coordinates for matplotlib)
DEFAULT_NODE_WIDTH: float = 1.2
DEFAULT_NODE_HEIGHT: float = 0.6
DEFAULT_NODE_FACECOLOR: str = "#4A90D9"
DEFAULT_NODE_EDGECOLOR: str = "#2C3E50"
DEFAULT_FONTSIZE: int = 10

# Default edge styling
DEFAULT_EDGE_COLOR: str = "#555555"
DEFAULT_EDGE_LINEWIDTH: float = 1.5
DEFAULT_ARROWSTYLE: str = "->,head_length=0.4,head_width=0.2"
DEFAULT_RATE_FONTSIZE: int = 9

# Ground state bar styling
DEFAULT_GROUND_STATE_COLOR: str = "#222222"
DEFAULT_GROUND_STATE_LINEWIDTH: float = 4.0
DEFAULT_GROUND_STATE_PER_MC_LINEWIDTH: float = 2.5

# Layout defaults
DEFAULT_FIGSIZE: tuple[float, float] = (10.0, 8.0)
DEFAULT_HORIZONTAL_SPACING: float = 2.0
DEFAULT_VERTICAL_SPACING: float = 1.5
DEFAULT_GROUND_STATE_OFFSET: float = 1.2
DEFAULT_COMPONENT_GAP: float = 3.0
"""Horizontal gap between disconnected graph components in side-by-side layout."""

# W3C luminance threshold for text contrast
LUMINANCE_THRESHOLD: float = 0.179
