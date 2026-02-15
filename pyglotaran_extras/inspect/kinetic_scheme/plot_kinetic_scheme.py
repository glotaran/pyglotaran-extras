"""Layer 4: Matplotlib rendering and public API for kinetic scheme visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_ARROWSTYLE
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_COMPONENT_GAP
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_EDGE_COLOR
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_EDGE_LINEWIDTH
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_FIGSIZE
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_FONTSIZE
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_GROUND_STATE_COLOR
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_GROUND_STATE_LINEWIDTH
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_GROUND_STATE_OFFSET
from pyglotaran_extras.inspect.kinetic_scheme._constants import (
    DEFAULT_GROUND_STATE_PER_MC_LINEWIDTH,
)
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_HORIZONTAL_SPACING
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_NODE_EDGECOLOR
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_NODE_FACECOLOR
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_NODE_HEIGHT
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_NODE_WIDTH
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_RATE_FONTSIZE
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_VERTICAL_SPACING
from pyglotaran_extras.inspect.kinetic_scheme._constants import LUMINANCE_THRESHOLD
from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import extract_dataset_transitions
from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import extract_transitions
from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticGraph
from pyglotaran_extras.inspect.kinetic_scheme._layout import LayoutAlgorithm
from pyglotaran_extras.inspect.kinetic_scheme._layout import compute_layout

if TYPE_CHECKING:
    from glotaran.model import Model
    from glotaran.parameter import Parameters
    from matplotlib.axes import Axes

    from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticEdge
    from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticNode
    from pyglotaran_extras.inspect.kinetic_scheme._layout import NodePositions


class NodeStyleConfig(BaseModel):
    """Per-node style overrides for kinetic scheme visualization.

    Parameters
    ----------
    display_label : str | None
        Custom display name for the node. Defaults to None (use node label).
    width : float
        Node width in data coordinates. Defaults to 1.2.
    height : float
        Node height in data coordinates. Defaults to 0.6.
    facecolor : str | None
        Node fill color. Defaults to None (use global default).
    fontsize : int
        Font size for the node label. Defaults to 10.
    """

    model_config = ConfigDict(extra="forbid")

    display_label: str | None = None
    width: float = DEFAULT_NODE_WIDTH
    height: float = DEFAULT_NODE_HEIGHT
    facecolor: str | None = None
    fontsize: int = DEFAULT_FONTSIZE


class KineticSchemeConfig(BaseModel):
    """Configuration for kinetic scheme visualization.

    Parameters
    ----------
    node_styles : dict[str, NodeStyleConfig]
        Per-node style overrides keyed by node label.
    color_mapping : dict[str, list[str]]
        Maps color strings to lists of node labels that should use that color.
    node_facecolor : str
        Default node fill color.
    node_edgecolor : str
        Default node border color.
    node_width : float
        Default node width.
    node_height : float
        Default node height.
    edge_color : str
        Default edge color.
    edge_linewidth : float
        Default edge line width.
    rate_unit : Literal["ps", "ns"]
        Unit for displaying rate constants.
    rate_decimal_places : int | None
        Decimal places for rate constant display. None for smart rounding.
    show_rate_labels : bool
        Whether to show parameter label prefix on edges.
    show_ground_state : Literal[False, "shared", "per_megacomplex"]
        Ground state bar rendering mode.
    layout_algorithm : Literal["hierarchical", "spring", "manual"]
        Layout algorithm to use.
    horizontal_layout_preference : str | None
        Pipe-delimited preferred node ordering (e.g., ``"S2|S1|T1"``).
    manual_positions : dict[str, tuple[float, float]] | None
        User-supplied positions for manual layout.
    horizontal_spacing : float
        Horizontal distance between nodes.
    vertical_spacing : float
        Vertical distance between layers.
    ground_state_offset : float
        Vertical offset for ground state below parent node.
    component_gap : float
        Horizontal gap between disconnected graph components.
    figsize : tuple[float, float]
        Default figure size in inches.
    title : str | None
        Optional plot title.
    omit_parameters : set[str]
        Parameter labels to exclude from visualization.
    """

    model_config = ConfigDict(extra="forbid")

    # Node styling
    node_styles: dict[str, NodeStyleConfig] = Field(default_factory=dict)
    color_mapping: dict[str, list[str]] = Field(default_factory=dict)
    node_facecolor: str = DEFAULT_NODE_FACECOLOR
    node_edgecolor: str = DEFAULT_NODE_EDGECOLOR
    node_width: float = DEFAULT_NODE_WIDTH
    node_height: float = DEFAULT_NODE_HEIGHT

    # Edge styling
    edge_color: str = DEFAULT_EDGE_COLOR
    edge_linewidth: float = DEFAULT_EDGE_LINEWIDTH

    # Rate formatting
    rate_unit: Literal["ps", "ns"] = "ns"
    rate_decimal_places: int | None = None
    show_rate_labels: bool = False

    # Ground state rendering
    show_ground_state: Literal[False, "shared", "per_megacomplex"] = False

    # Layout
    layout_algorithm: Literal["hierarchical", "spring", "manual"] = "hierarchical"
    horizontal_layout_preference: str | None = None
    manual_positions: dict[str, tuple[float, float]] | None = None
    horizontal_spacing: float = DEFAULT_HORIZONTAL_SPACING
    vertical_spacing: float = DEFAULT_VERTICAL_SPACING
    ground_state_offset: float = DEFAULT_GROUND_STATE_OFFSET
    component_gap: float = DEFAULT_COMPONENT_GAP

    # Figure
    figsize: tuple[float, float] = DEFAULT_FIGSIZE
    title: str | None = None

    # Filtering
    omit_parameters: set[str] = Field(default_factory=set)


def show_kinetic_scheme(
    megacomplexes: str | list[str],
    model: Model,
    parameters: Parameters,
    *,
    ax: Axes | None = None,
    config: KineticSchemeConfig | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Show a kinetic decay scheme from pyglotaran megacomplexes.

    Parameters
    ----------
    megacomplexes : str | list[str]
        One or more megacomplex labels to visualize.
    model : Model
        A pyglotaran model containing the megacomplex definitions.
    parameters : Parameters
        Resolved parameters for the model.
    ax : Axes | None
        Matplotlib axes to plot on. If None, a new figure is created.
    config : KineticSchemeConfig | None
        Visualization configuration. If None, defaults are used.
    figsize : tuple[float, float]
        Figure size in inches. Only used when ``ax`` is None.
    title : str | None
        Optional title for the plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes containing the kinetic scheme plot.
    """
    config = config or KineticSchemeConfig()

    effective_figsize = config.figsize if config.figsize != DEFAULT_FIGSIZE else figsize
    effective_title = config.title if config.title is not None else title

    # Layer 1: Extract
    transitions = extract_transitions(
        megacomplexes, model, parameters, omit_parameters=config.omit_parameters
    )

    # Layer 2: Build graph
    graph = KineticGraph.from_transitions(transitions)

    # Layer 3: Layout
    algorithm = LayoutAlgorithm(config.layout_algorithm)
    positions = compute_layout(
        graph,
        algorithm,
        horizontal_spacing=config.horizontal_spacing,
        vertical_spacing=config.vertical_spacing,
        ground_state_offset=config.ground_state_offset,
        component_gap=config.component_gap,
        horizontal_layout_preference=config.horizontal_layout_preference,
        manual_positions=config.manual_positions,
    )

    # Layer 4: Render
    fig, ax_result = _render_kinetic_scheme(
        graph, positions, config, ax=ax, figsize=effective_figsize, title=effective_title
    )

    return fig, ax_result


def show_dataset_kinetic_scheme(
    dataset_name: str,
    model: Model,
    parameters: Parameters,
    *,
    exclude_megacomplexes: set[str] | None = None,
    ax: Axes | None = None,
    config: KineticSchemeConfig | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Show a kinetic decay scheme from all decay megacomplexes of a dataset.

    Parameters
    ----------
    dataset_name : str
        The dataset name to look up in the model.
    model : Model
        A pyglotaran model containing the dataset and megacomplex definitions.
    parameters : Parameters
        Resolved parameters for the model.
    exclude_megacomplexes : set[str] | None
        Megacomplex labels to exclude. Defaults to None.
    ax : Axes | None
        Matplotlib axes to plot on. If None, a new figure is created.
    config : KineticSchemeConfig | None
        Visualization configuration. If None, defaults are used.
    figsize : tuple[float, float]
        Figure size in inches. Only used when ``ax`` is None.
    title : str | None
        Optional title for the plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes containing the kinetic scheme plot.
    """
    config = config or KineticSchemeConfig()

    effective_figsize = config.figsize if config.figsize != DEFAULT_FIGSIZE else figsize
    effective_title = config.title if config.title is not None else title

    # Layer 1: Extract
    transitions = extract_dataset_transitions(
        dataset_name,
        model,
        parameters,
        exclude_megacomplexes=exclude_megacomplexes,
        omit_parameters=config.omit_parameters,
    )

    # Layer 2: Build graph
    graph = KineticGraph.from_transitions(transitions)

    # Layer 3: Layout
    algorithm = LayoutAlgorithm(config.layout_algorithm)
    positions = compute_layout(
        graph,
        algorithm,
        horizontal_spacing=config.horizontal_spacing,
        vertical_spacing=config.vertical_spacing,
        ground_state_offset=config.ground_state_offset,
        component_gap=config.component_gap,
        horizontal_layout_preference=config.horizontal_layout_preference,
        manual_positions=config.manual_positions,
    )

    # Layer 4: Render
    fig, ax_result = _render_kinetic_scheme(
        graph, positions, config, ax=ax, figsize=effective_figsize, title=effective_title
    )

    return fig, ax_result


# ---------------------------------------------------------------------------
# Internal rendering functions
# ---------------------------------------------------------------------------


def _render_kinetic_scheme(
    graph: KineticGraph,
    positions: NodePositions,
    config: KineticSchemeConfig,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Render the kinetic scheme onto a matplotlib axes.

    Parameters
    ----------
    graph : KineticGraph
        The kinetic graph to render.
    positions : NodePositions
        Node positions in data coordinates.
    config : KineticSchemeConfig
        Visualization configuration.
    ax : Axes | None
        Axes to render on. Creates new if None.
    figsize : tuple[float, float]
        Figure size if creating a new figure.
    title : str | None
        Plot title.

    Returns
    -------
    tuple[Figure, Axes]
        The Figure and Axes.
    """
    if ax is None:
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig_or_none = ax.get_figure()
        assert isinstance(fig_or_none, Figure), "Axes must belong to a Figure"
        fig = fig_or_none

    # Draw ground state bars (if enabled)
    if config.show_ground_state == "shared":
        _draw_shared_ground_state_bar(ax, graph, positions, config)
    elif config.show_ground_state == "per_megacomplex":
        _draw_per_megacomplex_ground_state_bars(ax, graph, positions, config)

    # Draw edges first (so nodes are on top)
    _draw_all_edges(ax, graph, positions, config)

    # Draw nodes
    for node in graph.compartment_nodes():
        if node.label in positions:
            _draw_node(ax, node, positions[node.label], config)

    # Configure axes
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Auto-fit margins
    ax.margins(0.15)

    return fig, ax


def _get_node_facecolor(node: KineticNode, config: KineticSchemeConfig) -> str:
    """Determine the fill color for a node.

    Parameters
    ----------
    node : KineticNode
        The node.
    config : KineticSchemeConfig
        Configuration containing color mappings.

    Returns
    -------
    str
        The resolved facecolor.
    """
    # Priority 1: Per-node style override
    if node.label in config.node_styles:
        style = config.node_styles[node.label]
        if style.facecolor is not None:
            return style.facecolor

    # Priority 2: Node's own color attribute
    if node.color is not None:
        return node.color

    # Priority 3: color_mapping lookup
    for color, labels in config.color_mapping.items():
        if node.label in labels:
            return color

    # Priority 4: global default
    return config.node_facecolor


def _get_node_dimensions(node_label: str, config: KineticSchemeConfig) -> tuple[float, float]:
    """Get width and height for a node.

    Parameters
    ----------
    node_label : str
        The node label.
    config : KineticSchemeConfig
        Configuration.

    Returns
    -------
    tuple[float, float]
        (width, height) in data coordinates.
    """
    if node_label in config.node_styles:
        style = config.node_styles[node_label]
        return style.width, style.height
    return config.node_width, config.node_height


def _get_display_label(node: KineticNode, config: KineticSchemeConfig) -> str:
    """Get the display label for a node.

    Parameters
    ----------
    node : KineticNode
        The node.
    config : KineticSchemeConfig
        Configuration.

    Returns
    -------
    str
        The label to display.
    """
    if node.label in config.node_styles:
        style = config.node_styles[node.label]
        if style.display_label is not None:
            return style.display_label
    if node.display_label is not None:
        return node.display_label
    return node.label


def _compute_text_color(facecolor: str) -> str:
    """Compute text color for contrast against the given background.

    Uses the W3C relative luminance formula on linearized sRGB values.

    Parameters
    ----------
    facecolor : str
        Background color string (any matplotlib-compatible format).

    Returns
    -------
    str
        ``"white"`` or ``"black"`` depending on background luminance.
    """
    rgba = mcolors.to_rgba(facecolor)
    # Linearize sRGB
    channels = []
    for c in rgba[:3]:
        if c <= 0.04045:
            channels.append(c / 12.92)
        else:
            channels.append(((c + 0.055) / 1.055) ** 2.4)

    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "black" if luminance > LUMINANCE_THRESHOLD else "white"


def _draw_node(
    ax: Axes,
    node: KineticNode,
    position: tuple[float, float],
    config: KineticSchemeConfig,
) -> None:
    """Draw a single node as a rounded rectangle with centered text.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    node : KineticNode
        The node to draw.
    position : tuple[float, float]
        Center position in data coordinates.
    config : KineticSchemeConfig
        Configuration.
    """
    width, height = _get_node_dimensions(node.label, config)
    facecolor = _get_node_facecolor(node, config)
    text_color = _compute_text_color(facecolor)
    display_label = _get_display_label(node, config)

    fontsize = DEFAULT_FONTSIZE
    if node.label in config.node_styles:
        fontsize = config.node_styles[node.label].fontsize

    x, y = position
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.05",
        facecolor=facecolor,
        edgecolor=config.node_edgecolor,
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        display_label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
        zorder=4,
    )


def _compute_arrow_endpoints(
    source_center: tuple[float, float],
    target_center: tuple[float, float],
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute arrow start/end points at node rectangle boundaries.

    Parameters
    ----------
    source_center : tuple[float, float]
        Center of source node.
    target_center : tuple[float, float]
        Center of target node.
    source_width : float
        Width of source node.
    source_height : float
        Height of source node.
    target_width : float
        Width of target node.
    target_height : float
        Height of target node.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        (start_point, end_point) on the boundaries of source and target.
    """
    sx, sy = source_center
    tx, ty = target_center

    dx = tx - sx
    dy = ty - sy

    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return source_center, target_center

    # Find intersection with source rectangle boundary
    start = _rect_edge_intersection(sx, sy, source_width, source_height, dx, dy)

    # Find intersection with target rectangle boundary (from target toward source)
    end = _rect_edge_intersection(tx, ty, target_width, target_height, -dx, -dy)

    return start, end


def _rect_edge_intersection(
    cx: float,
    cy: float,
    width: float,
    height: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Find where a ray from the center of a rectangle exits the rectangle.

    Parameters
    ----------
    cx : float
        Center x.
    cy : float
        Center y.
    width : float
        Rectangle width.
    height : float
        Rectangle height.
    dx : float
        Direction x component.
    dy : float
        Direction y component.

    Returns
    -------
    tuple[float, float]
        The point where the ray exits the rectangle boundary.
    """
    half_w = width / 2
    half_h = height / 2

    if abs(dx) < 1e-10:
        # Vertical line
        return (cx, cy + half_h) if dy > 0 else (cx, cy - half_h)
    if abs(dy) < 1e-10:
        # Horizontal line
        return (cx + half_w, cy) if dx > 0 else (cx - half_w, cy)

    # Find parameter t for intersection with each edge
    t_right = half_w / abs(dx)
    t_left = half_w / abs(dx)
    t_top = half_h / abs(dy)
    t_bottom = half_h / abs(dy)

    # The exit point is the one with the smallest positive t
    t_x = t_right if dx > 0 else t_left
    t_y = t_top if dy > 0 else t_bottom
    t = min(t_x, t_y)

    return (cx + dx * t, cy + dy * t)


def _draw_all_edges(
    ax: Axes,
    graph: KineticGraph,
    positions: NodePositions,
    config: KineticSchemeConfig,
) -> None:
    """Draw all edges in the graph.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    graph : KineticGraph
        The graph.
    positions : NodePositions
        Node positions.
    config : KineticSchemeConfig
        Configuration.
    """
    # Track edge pairs to handle parallel edges (back-transfer)
    edge_count: dict[tuple[str, str], int] = {}

    for edge in graph.edges:
        target_node = graph.nodes.get(edge.target)
        if target_node is None:
            continue

        # Skip edges to ground state nodes when ground state is shown
        # (arrows to the bar are drawn separately)
        if target_node.is_ground_state:
            if config.show_ground_state is not False:
                _draw_ground_state_arrow(ax, edge, positions, config)
            else:
                _draw_ground_state_decay_arrow(ax, edge, positions, config)
            continue

        if edge.source not in positions or edge.target not in positions:
            continue

        # Count parallel edges for curvature
        pair_key = (
            min(edge.source, edge.target),
            max(edge.source, edge.target),
        )
        edge_count.setdefault(pair_key, 0)
        edge_count[pair_key] += 1
        edge_index = edge_count[pair_key]

        _draw_transfer_edge(ax, edge, positions, config, edge_index)


def _draw_transfer_edge(
    ax: Axes,
    edge: KineticEdge,
    positions: NodePositions,
    config: KineticSchemeConfig,
    edge_index: int,
) -> None:
    """Draw a transfer edge between two compartment nodes.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    edge : KineticEdge
        The edge to draw.
    positions : NodePositions
        Node positions.
    config : KineticSchemeConfig
        Configuration.
    edge_index : int
        Index among parallel edges (1-based). Used for curvature.
    """
    source_w, source_h = _get_node_dimensions(edge.source, config)
    target_w, target_h = _get_node_dimensions(edge.target, config)

    start, end = _compute_arrow_endpoints(
        positions[edge.source],
        positions[edge.target],
        source_w,
        source_h,
        target_w,
        target_h,
    )

    connection_style = "arc3,rad=0.0"
    if edge_index > 1:
        rad = 0.25 * ((-1) ** edge_index) * ((edge_index + 1) // 2)
        connection_style = f"arc3,rad={rad:.2f}"

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=DEFAULT_ARROWSTYLE,
        connectionstyle=connection_style,
        color=config.edge_color,
        linewidth=config.edge_linewidth,
        zorder=2,
    )
    ax.add_patch(arrow)

    # Rate label at t=0.35 (closer to source) to reduce overlap when edges converge
    rate_text = edge.format_rate(
        unit=config.rate_unit,
        decimal_places=config.rate_decimal_places,
        show_label=config.show_rate_labels,
    )
    t = 0.35
    label_x = start[0] + t * (end[0] - start[0])
    label_y = start[1] + t * (end[1] - start[1])

    # Perpendicular offset to avoid overlapping the arrow line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx**2 + dy**2) ** 0.5, 0.01)
    offset_x = -dy / length * 0.25
    offset_y = dx / length * 0.25

    ax.text(
        label_x + offset_x,
        label_y + offset_y,
        rate_text,
        ha="center",
        va="center",
        fontsize=DEFAULT_RATE_FONTSIZE,
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.1",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )


def _draw_ground_state_decay_arrow(
    ax: Axes,
    edge: KineticEdge,
    positions: NodePositions,
    config: KineticSchemeConfig,
) -> None:
    """Draw a ground state decay arrow pointing downward into empty space.

    Used when ``show_ground_state`` is ``False``.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    edge : KineticEdge
        The ground state decay edge.
    positions : NodePositions
        Node positions.
    config : KineticSchemeConfig
        Configuration.
    """
    if edge.source not in positions:
        return

    _, source_h = _get_node_dimensions(edge.source, config)
    sx, sy = positions[edge.source]

    start = (sx, sy - source_h / 2)
    end = (sx, sy - source_h / 2 - config.ground_state_offset * 0.7)

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=DEFAULT_ARROWSTYLE,
        color=config.edge_color,
        linewidth=config.edge_linewidth,
        zorder=2,
    )
    ax.add_patch(arrow)

    # Rate label
    rate_text = edge.format_rate(
        unit=config.rate_unit,
        decimal_places=config.rate_decimal_places,
        show_label=config.show_rate_labels,
    )
    mid_y = (start[1] + end[1]) / 2
    ax.text(
        sx + 0.2,
        mid_y,
        rate_text,
        ha="left",
        va="center",
        fontsize=DEFAULT_RATE_FONTSIZE,
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.1",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )


def _draw_ground_state_arrow(
    ax: Axes,
    edge: KineticEdge,
    positions: NodePositions,
    config: KineticSchemeConfig,
) -> None:
    """Draw a ground state decay arrow pointing to the ground state bar.

    Used when ``show_ground_state`` is ``"shared"`` or ``"per_megacomplex"``.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    edge : KineticEdge
        The ground state decay edge.
    positions : NodePositions
        Node positions.
    config : KineticSchemeConfig
        Configuration.
    """
    if edge.source not in positions or edge.target not in positions:
        return

    _, source_h = _get_node_dimensions(edge.source, config)
    sx, sy = positions[edge.source]
    _, ty = positions[edge.target]

    start = (sx, sy - source_h / 2)
    end = (sx, ty + 0.05)  # Slightly above the bar

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=DEFAULT_ARROWSTYLE,
        color=config.edge_color,
        linewidth=config.edge_linewidth,
        zorder=2,
    )
    ax.add_patch(arrow)

    # Rate label
    rate_text = edge.format_rate(
        unit=config.rate_unit,
        decimal_places=config.rate_decimal_places,
        show_label=config.show_rate_labels,
    )
    mid_y = (start[1] + end[1]) / 2
    ax.text(
        sx + 0.2,
        mid_y,
        rate_text,
        ha="left",
        va="center",
        fontsize=DEFAULT_RATE_FONTSIZE,
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.1",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )


def _draw_shared_ground_state_bar(
    ax: Axes,
    graph: KineticGraph,
    positions: NodePositions,
    config: KineticSchemeConfig,
) -> None:
    """Draw a shared ground state bar at the bottom of the figure.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    graph : KineticGraph
        The graph.
    positions : NodePositions
        Node positions.
    config : KineticSchemeConfig
        Configuration.
    """
    compartment_positions = [
        positions[n.label] for n in graph.compartment_nodes() if n.label in positions
    ]
    if not compartment_positions:
        return

    x_coords = [p[0] for p in compartment_positions]
    y_coords = [p[1] for p in compartment_positions]

    margin = config.node_width * 0.8
    x_min = min(x_coords) - margin
    x_max = max(x_coords) + margin
    y_bar = min(y_coords) - config.ground_state_offset

    # Update GS node positions to be at the bar level
    for gs_node in graph.ground_state_nodes():
        parents = graph.predecessors(gs_node.label)
        if parents and parents[0] in positions:
            parent_x = positions[parents[0]][0]
            positions[gs_node.label] = (parent_x, y_bar)

    ax.plot(
        [x_min, x_max],
        [y_bar, y_bar],
        color=DEFAULT_GROUND_STATE_COLOR,
        linewidth=DEFAULT_GROUND_STATE_LINEWIDTH,
        solid_capstyle="butt",
        zorder=1,
    )


def _draw_per_megacomplex_ground_state_bars(
    ax: Axes,
    graph: KineticGraph,
    positions: NodePositions,
    config: KineticSchemeConfig,
) -> None:
    """Draw per-megacomplex ground state bars below each group.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes.
    graph : KineticGraph
        The graph.
    positions : NodePositions
        Node positions.
    config : KineticSchemeConfig
        Configuration.
    """
    # Group compartment nodes by megacomplex
    mc_groups: dict[str, list[str]] = {}
    for node in graph.compartment_nodes():
        if node.label not in positions:
            continue
        for mc_label in node.megacomplex_labels:
            mc_groups.setdefault(mc_label, []).append(node.label)

    margin = config.node_width * 0.5

    for mc_label, node_labels in mc_groups.items():
        mc_positions = [positions[label] for label in node_labels]
        x_coords = [p[0] for p in mc_positions]
        y_coords = [p[1] for p in mc_positions]

        x_min = min(x_coords) - margin
        x_max = max(x_coords) + margin
        y_bar = min(y_coords) - config.ground_state_offset

        # Update GS node positions for this megacomplex
        for gs_node in graph.ground_state_nodes():
            if mc_label in gs_node.megacomplex_labels:
                parents = graph.predecessors(gs_node.label)
                if parents and parents[0] in positions:
                    parent_x = positions[parents[0]][0]
                    positions[gs_node.label] = (parent_x, y_bar)

        ax.plot(
            [x_min, x_max],
            [y_bar, y_bar],
            color=DEFAULT_GROUND_STATE_COLOR,
            linewidth=DEFAULT_GROUND_STATE_PER_MC_LINEWIDTH,
            solid_capstyle="butt",
            zorder=1,
        )
