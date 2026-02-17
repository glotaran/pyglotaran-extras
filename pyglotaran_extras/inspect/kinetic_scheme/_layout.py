"""Layer 3: Layout algorithms for kinetic scheme graphs."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_COMPONENT_GAP
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_GROUND_STATE_OFFSET
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_HORIZONTAL_SPACING
from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_VERTICAL_SPACING

if TYPE_CHECKING:
    from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticGraph

NodePositions = dict[str, tuple[float, float]]
"""Mapping from node label to (x, y) position in data coordinates."""

_LOGGER = logging.getLogger(__name__)
_SAME_COLUMN_TOLERANCE = 0.3


class LayoutAlgorithm(str, Enum):
    """Available layout algorithms for kinetic scheme visualization.

    Attributes
    ----------
    HIERARCHICAL
        Layered layout based on topological ordering. Best for DAGs and
        simple cyclic graphs.
    SPRING
        Force-directed Fruchterman-Reingold layout. For complex cyclic
        schemes.
    MANUAL
        User-supplied compartment positions with derived GS positioning.
    """

    HIERARCHICAL = "hierarchical"
    SPRING = "spring"
    MANUAL = "manual"


def compute_layout(
    graph: KineticGraph,
    algorithm: LayoutAlgorithm = LayoutAlgorithm.HIERARCHICAL,
    *,
    horizontal_spacing: float = DEFAULT_HORIZONTAL_SPACING,
    vertical_spacing: float = DEFAULT_VERTICAL_SPACING,
    ground_state_offset: float = DEFAULT_GROUND_STATE_OFFSET,
    component_gap: float = DEFAULT_COMPONENT_GAP,
    horizontal_layout_preference: str | None = None,
    manual_positions: NodePositions | None = None,
) -> NodePositions:
    """Compute (x, y) positions for all nodes in the kinetic graph.

    Parameters
    ----------
    graph : KineticGraph
        The kinetic scheme graph to lay out.
    algorithm : LayoutAlgorithm
        The layout algorithm to use. Defaults to ``HIERARCHICAL``.
    horizontal_spacing : float
        Horizontal distance between adjacent nodes. Defaults to 0.0
        (sentinel: auto-computed as ``3 × DEFAULT_NODE_WIDTH``).
    vertical_spacing : float
        Vertical distance between layers. Defaults to 2.0.
    ground_state_offset : float
        Vertical offset for ground state nodes below their parent. Defaults to 1.2.
    component_gap : float
        Horizontal gap between disconnected graph components. Defaults to 3.0.
    horizontal_layout_preference : str | None
        Pipe-delimited string specifying preferred left-to-right node ordering
        (e.g., ``"S2|S1|T1"``). Defaults to None.
    manual_positions : NodePositions | None
        User-supplied positions for manual layout. Required when
        ``algorithm`` is ``MANUAL``.

    Returns
    -------
    NodePositions
        Mapping from node label to (x, y) position.

    Raises
    ------
    ValueError
        If manual layout is selected but positions are missing for some nodes.
    """
    # Resolve the sentinel value: 0 means "auto-compute from node width".
    # The default node width (1.2) gives 3 × 1.2 = 3.6.
    if horizontal_spacing <= 0:
        from pyglotaran_extras.inspect.kinetic_scheme._constants import DEFAULT_NODE_WIDTH

        horizontal_spacing = 3.0 * DEFAULT_NODE_WIDTH

    if algorithm == LayoutAlgorithm.MANUAL:
        positions = _manual_layout(graph, manual_positions)
    elif algorithm == LayoutAlgorithm.SPRING:
        positions = _spring_layout(
            graph,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
    else:
        positions = _hierarchical_layout(
            graph,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            component_gap=component_gap,
            horizontal_layout_preference=horizontal_layout_preference,
        )

    # Position ground state nodes below their parent compartment
    _position_ground_state_nodes(graph, positions, ground_state_offset)

    # Nudge compartment nodes that sit directly below a node with a ground
    # state decay arrow, to prevent the transfer edge from overlapping the
    # decay arrow.
    _avoid_ground_state_arrow_overlap(graph, positions, horizontal_spacing)

    return positions


def _find_connected_components(graph: KineticGraph) -> list[set[str]]:
    """Find connected components among compartment nodes using BFS.

    Treats edges as undirected (uses both successors and predecessors).
    Ground state nodes are excluded.

    Parameters
    ----------
    graph : KineticGraph
        The kinetic graph.

    Returns
    -------
    list[set[str]]
        Connected components sorted by the smallest label in each component.
    """
    compartment_labels = {n.label for n in graph.compartment_nodes()}
    visited: set[str] = set()
    components: list[set[str]] = []

    for start in sorted(compartment_labels):
        if start in visited:
            continue

        component: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)

            # Traverse both directions (undirected connectivity)
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
            for neighbor in neighbors:
                if neighbor in compartment_labels and neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return sorted(components, key=min)


def _hierarchical_layout(
    graph: KineticGraph,
    *,
    horizontal_spacing: float,
    vertical_spacing: float,
    component_gap: float,
    horizontal_layout_preference: str | None,
) -> NodePositions:
    """Layered layout for DAGs and cycle-broken graphs.

    Detects disconnected components and lays each out independently,
    placing them side-by-side with ``component_gap`` spacing.

    Parameters
    ----------
    graph : KineticGraph
        The graph to lay out.
    horizontal_spacing : float
        Horizontal distance between adjacent nodes.
    vertical_spacing : float
        Vertical distance between layers.
    component_gap : float
        Horizontal gap between disconnected components.
    horizontal_layout_preference : str | None
        Preferred left-to-right node ordering.

    Returns
    -------
    NodePositions
        Computed positions for compartment nodes.
    """
    compartment_labels = {n.label for n in graph.compartment_nodes()}

    if not compartment_labels:
        return {}

    # Detect disconnected components
    components = _find_connected_components(graph)

    if len(components) <= 1:
        # Single component: use the original single-component layout
        return _layout_single_component(
            graph,
            compartment_labels,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            horizontal_layout_preference=horizontal_layout_preference,
        )

    # Multiple components: lay each out independently, then place side-by-side
    # Order components by horizontal_layout_preference if provided
    if horizontal_layout_preference:
        pref_list = [s.strip() for s in horizontal_layout_preference.split("|")]
        pref_index = {name: i for i, name in enumerate(pref_list)}

        def _component_sort_key(comp: set[str]) -> tuple[int, str]:
            """Sort components by earliest preferred label, then alphabetically.

            Parameters
            ----------
            comp : set[str]
                A set of node labels forming a connected component.

            Returns
            -------
            tuple[int, str]
                Sort key tuple (preference index, minimum label).
            """
            best = len(pref_list)
            for label in comp:
                if label in pref_index:
                    best = min(best, pref_index[label])
            return (best, min(comp))

        components = sorted(components, key=_component_sort_key)

    all_positions: NodePositions = {}
    current_x_offset = 0.0

    for component_labels in components:
        comp_positions = _layout_single_component(
            graph,
            component_labels,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            horizontal_layout_preference=horizontal_layout_preference,
        )

        if not comp_positions:
            continue

        # Find x range of this component's positions
        x_values = [pos[0] for pos in comp_positions.values()]
        x_min = min(x_values)
        x_max = max(x_values)

        # Shift component so its leftmost node starts at current_x_offset
        shift = current_x_offset - x_min
        for label, (x, y) in comp_positions.items():
            all_positions[label] = (x + shift, y)

        # Advance offset past this component plus the gap
        current_x_offset += (x_max - x_min) + component_gap

    return all_positions


def _layout_single_component(
    graph: KineticGraph,
    compartment_labels: set[str],
    *,
    horizontal_spacing: float,
    vertical_spacing: float,
    horizontal_layout_preference: str | None,
) -> NodePositions:
    """Layered layout for a single connected component.

    Parameters
    ----------
    graph : KineticGraph
        The full graph (edges/adjacency used for ordering).
    compartment_labels : set[str]
        Labels of compartment nodes in this component.
    horizontal_spacing : float
        Horizontal distance between adjacent nodes.
    vertical_spacing : float
        Vertical distance between layers.
    horizontal_layout_preference : str | None
        Preferred left-to-right node ordering.

    Returns
    -------
    NodePositions
        Computed positions for compartment nodes in this component.
    """
    if not compartment_labels:
        return {}

    # Handle cyclic graphs by breaking back edges
    is_dag = graph.is_dag()
    back_edges: set[tuple[str, str]] = set()

    if not is_dag:
        back_edges = _find_back_edges(graph, compartment_labels)

    # Step 1: Layer assignment via longest-path
    layers = _assign_layers(graph, compartment_labels, back_edges)

    # Step 2: Group nodes by layer
    layer_groups: dict[int, list[str]] = {}
    for label, layer in layers.items():
        layer_groups.setdefault(layer, []).append(label)

    # Step 3: Within-layer ordering
    preference_order: list[str] | None = None
    if horizontal_layout_preference:
        preference_order = [s.strip() for s in horizontal_layout_preference.split("|")]

    for layer_idx in sorted(layer_groups):
        nodes_in_layer = layer_groups[layer_idx]
        layer_groups[layer_idx] = _order_within_layer(
            nodes_in_layer, graph, layers, back_edges, preference_order
        )

    # Step 4: Coordinate assignment (center each layer)
    max_layer_size = max(len(nodes) for nodes in layer_groups.values())
    positions: NodePositions = {}

    for layer_idx in sorted(layer_groups):
        nodes_in_layer = layer_groups[layer_idx]
        layer_width = (len(nodes_in_layer) - 1) * horizontal_spacing
        max_width = (max_layer_size - 1) * horizontal_spacing
        offset_x = (max_width - layer_width) / 2.0

        for i, label in enumerate(nodes_in_layer):
            x = offset_x + i * horizontal_spacing
            y = -layer_idx * vertical_spacing
            positions[label] = (x, y)

    return positions


def _assign_layers(
    graph: KineticGraph,
    compartment_labels: set[str],
    back_edges: set[tuple[str, str]],
) -> dict[str, int]:
    """Assign each compartment node to a layer using longest-path method.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    compartment_labels : set[str]
        Labels of compartment (non-GS) nodes.
    back_edges : set[tuple[str, str]]
        Edges to ignore (treat as reversed) for layer assignment.

    Returns
    -------
    dict[str, int]
        Mapping from node label to layer index (0 = top).
    """
    in_degree = _build_in_degrees(graph, compartment_labels, back_edges)
    layers, queue = _init_layer_sources(in_degree, compartment_labels)
    _propagate_layers(graph, queue, layers, in_degree, compartment_labels, back_edges)

    # Ensure all compartment nodes have a layer assignment
    for label in compartment_labels:
        layers.setdefault(label, 0)

    return layers


def _build_in_degrees(
    graph: KineticGraph,
    compartment_labels: set[str],
    back_edges: set[tuple[str, str]],
) -> dict[str, int]:
    """Build in-degree map for compartment nodes, ignoring back edges.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    compartment_labels : set[str]
        Labels of compartment nodes.
    back_edges : set[tuple[str, str]]
        Edges to ignore.

    Returns
    -------
    dict[str, int]
        In-degree for each compartment node.
    """
    in_degree: dict[str, int] = dict.fromkeys(compartment_labels, 0)
    seen_edges: set[tuple[str, str]] = set()
    for edge in graph.edges:
        edge_pair = (edge.source, edge.target)
        if edge_pair in seen_edges:
            continue
        seen_edges.add(edge_pair)
        if (
            edge.source in compartment_labels
            and edge.target in compartment_labels
            and edge_pair not in back_edges
        ):
            in_degree[edge.target] += 1
    return in_degree


def _init_layer_sources(
    in_degree: dict[str, int],
    compartment_labels: set[str],
) -> tuple[dict[str, int], deque[str]]:
    """Initialize layers and queue from source nodes (in-degree 0).

    Parameters
    ----------
    in_degree : dict[str, int]
        In-degree map.
    compartment_labels : set[str]
        All compartment labels.

    Returns
    -------
    tuple[dict[str, int], deque[str]]
        Initial layers dict and BFS queue.
    """
    queue = deque(sorted(label for label, deg in in_degree.items() if deg == 0))

    if not queue:
        queue.append(sorted(compartment_labels)[0])

    layers: dict[str, int] = dict.fromkeys(queue, 0)
    return layers, queue


def _propagate_layers(
    graph: KineticGraph,
    queue: deque[str],
    layers: dict[str, int],
    in_degree: dict[str, int],
    compartment_labels: set[str],
    back_edges: set[tuple[str, str]],
) -> None:
    """BFS to propagate layer assignments.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    queue : deque[str]
        BFS queue.
    layers : dict[str, int]
        Layer assignments to update.
    in_degree : dict[str, int]
        In-degree map.
    compartment_labels : set[str]
        Compartment node labels.
    back_edges : set[tuple[str, str]]
        Edges to skip.
    """
    processed: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    while queue:
        node = queue.popleft()
        if node in processed:
            continue
        processed.add(node)

        for neighbor in sorted(graph.successors(node)):
            edge_pair = (node, neighbor)
            if (
                neighbor not in compartment_labels
                or edge_pair in back_edges
                or edge_pair in seen_edges
            ):
                continue
            seen_edges.add(edge_pair)
            new_layer = layers[node] + 1
            if neighbor not in layers or new_layer > layers[neighbor]:
                layers[neighbor] = new_layer
            in_degree[neighbor] = max(0, in_degree[neighbor] - 1)
            if in_degree[neighbor] == 0:
                queue.append(neighbor)


def _order_within_layer(
    nodes_in_layer: list[str],
    graph: KineticGraph,
    layers: dict[str, int],
    back_edges: set[tuple[str, str]],
    preference_order: list[str] | None,
) -> list[str]:
    """Order nodes within a layer to minimize edge crossings.

    Parameters
    ----------
    nodes_in_layer : list[str]
        Node labels to order.
    graph : KineticGraph
        The graph.
    layers : dict[str, int]
        Layer assignments for all nodes.
    back_edges : set[tuple[str, str]]
        Edges treated as reversed.
    preference_order : list[str] | None
        User-specified ordering preference.

    Returns
    -------
    list[str]
        Ordered node labels.
    """
    if len(nodes_in_layer) <= 1:
        return nodes_in_layer

    if preference_order:
        # Nodes in the preference list get their specified order
        pref_index = {name: i for i, name in enumerate(preference_order)}

        def sort_key(label: str) -> tuple[int, int, str]:
            """Return sort key prioritizing preference order then alphabetical.

            Parameters
            ----------
            label : str
                Node label to compute sort key for.

            Returns
            -------
            tuple[int, int, str]
                Sort key tuple (priority group, preference index, label).
            """
            if label in pref_index:
                return (0, pref_index[label], label)
            return (1, 0, label)

        return sorted(nodes_in_layer, key=sort_key)

    # Default: barycenter heuristic (order by avg position of predecessors)
    compartment_labels = {n.label for n in graph.compartment_nodes()}
    barycenters: dict[str, float] = {}

    for label in nodes_in_layer:
        preds = [
            p
            for p in graph.predecessors(label)
            if p in compartment_labels
            and p in layers
            and layers[p] < layers.get(label, 0)
            and (p, label) not in back_edges
        ]
        if preds:
            barycenters[label] = sum(_node_sort_index(p) for p in preds) / len(preds)
        else:
            barycenters[label] = 0.0

    def barycenter_sort_key(label: str) -> tuple[float, str]:
        """Return sort key based on barycenter and label.

        Parameters
        ----------
        label : str
            Node label.

        Returns
        -------
        tuple[float, str]
            Tuple of barycenter value and label for sorting.
        """
        return barycenters[label], label

    return sorted(nodes_in_layer, key=barycenter_sort_key)


def _node_sort_index(label: str) -> float:
    """Get a numerical index for a node used in barycenter calculations.

    Parameters
    ----------
    label : str
        Node label.

    Returns
    -------
    float
        The sort index based on deterministic label hash.
    """
    digest = hashlib.md5(label.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") / 4294967296.0


def _find_back_edges(graph: KineticGraph, compartment_labels: set[str]) -> set[tuple[str, str]]:
    """Find back edges in the compartment subgraph using DFS.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    compartment_labels : set[str]
        Labels of compartment nodes.

    Returns
    -------
    set[tuple[str, str]]
        Set of (source, target) edge pairs identified as back edges.
    """
    white = set(compartment_labels)
    gray: set[str] = set()
    back_edges: set[tuple[str, str]] = set()

    def _dfs(node: str) -> None:
        """Traverse from ``node`` recording back edges that form cycles.

        Parameters
        ----------
        node : str
            The starting node for DFS traversal.
        """
        white.discard(node)
        gray.add(node)
        for neighbor in graph.successors(node):
            if neighbor not in compartment_labels:
                continue
            if neighbor in gray:
                back_edges.add((node, neighbor))
            elif neighbor in white:
                _dfs(neighbor)
        gray.discard(node)

    for node in sorted(compartment_labels):
        if node in white:
            _dfs(node)

    return back_edges


def _position_ground_state_nodes(
    graph: KineticGraph,
    positions: NodePositions,
    ground_state_offset: float,
) -> None:
    """Position ground state nodes directly below their parent compartment node.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    positions : NodePositions
        Positions dict to update in-place.
    ground_state_offset : float
        Vertical offset below parent.
    """
    for gs_node in graph.ground_state_nodes():
        parents = graph.predecessors(gs_node.label)
        if parents and parents[0] in positions:
            parent_x, parent_y = positions[parents[0]]
            positions[gs_node.label] = (parent_x, parent_y - ground_state_offset)
        else:
            # Fallback: place at origin
            fallback_position = (0.0, -ground_state_offset)
            _LOGGER.debug(
                "Ground-state fallback positioning used for node '%s': "
                "parents=%s, fallback_position=%s",
                gs_node.label,
                parents,
                fallback_position,
            )
            positions[gs_node.label] = fallback_position


def _avoid_ground_state_arrow_overlap(
    graph: KineticGraph,
    positions: NodePositions,
    horizontal_spacing: float,
) -> None:
    """Nudge compartment nodes that overlap a ground state decay arrow.

    When a compartment node ``A`` has a ground state decay AND a transfer
    edge to another compartment node ``B``, and ``B`` is positioned
    directly below ``A`` (same x column), the downward decay arrow from
    ``A`` visually overlaps the edge to ``B``.  This function detects
    that situation and shifts ``B`` horizontally so the paths separate.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    positions : NodePositions
        Positions dict to update in-place.
    horizontal_spacing : float
        Current horizontal spacing (used to size the nudge).
    """
    # Identify compartment nodes that have ground state decay edges
    nodes_with_gs_decay: set[str] = set()
    for node in graph.compartment_nodes():
        if graph.ground_state_edges_for_node(node.label):
            nodes_with_gs_decay.add(node.label)

    if not nodes_with_gs_decay:
        return

    nudge = horizontal_spacing * 0.4

    for parent_label in nodes_with_gs_decay:
        if parent_label not in positions:
            continue
        px, py = positions[parent_label]

        for successor_label in graph.successors(parent_label):
            successor_node = graph.nodes.get(successor_label)
            if successor_node is None or successor_node.is_ground_state:
                continue
            if successor_label not in positions:
                continue
            sx, sy = positions[successor_label]

            # Check if successor is approximately in the same x column
            # and below the parent
            if abs(sx - px) < _SAME_COLUMN_TOLERANCE and sy < py:
                positions[successor_label] = (sx + nudge, sy)


def _spring_layout(
    graph: KineticGraph,
    *,
    horizontal_spacing: float,
    vertical_spacing: float,
    iterations: int = 50,
    seed: int = 42,
) -> NodePositions:
    """Fruchterman-Reingold force-directed layout for compartment nodes.

    Parameters
    ----------
    graph : KineticGraph
        The graph.
    horizontal_spacing : float
        Used to scale the layout.
    vertical_spacing : float
        Used to scale the layout.
    iterations : int
        Number of iterations. Defaults to 50.
    seed : int
        Random seed for deterministic output. Defaults to 42.

    Returns
    -------
    NodePositions
        Computed positions for compartment nodes.
    """
    compartment_nodes = graph.compartment_nodes()
    n = len(compartment_nodes)

    if n == 0:
        return {}
    if n == 1:
        return {compartment_nodes[0].label: (0.0, 0.0)}

    labels = [node.label for node in compartment_nodes]
    label_to_idx = {label: i for i, label in enumerate(labels)}

    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (n, 2))

    # Optimal distance
    area = horizontal_spacing * vertical_spacing * n
    k = np.sqrt(area / n)
    temperature = max(horizontal_spacing, vertical_spacing)
    cooling = temperature / (iterations + 1)

    # Build edge list for compartment nodes only
    edge_pairs: list[tuple[int, int]] = []
    for edge in graph.edges:
        if edge.source in label_to_idx and edge.target in label_to_idx:
            edge_pairs.append((label_to_idx[edge.source], label_to_idx[edge.target]))

    for _ in range(iterations):
        displacement = np.zeros((n, 2))
        _apply_repulsive_forces(pos, displacement, n, k)
        _apply_attractive_forces(pos, displacement, edge_pairs, k)
        _apply_displacement(pos, displacement, n, temperature)
        temperature -= cooling

    # Scale to desired spacing
    if n > 1:
        pos_range = pos.max(axis=0) - pos.min(axis=0)
        scale_x = horizontal_spacing * (n - 1) / max(pos_range[0], 0.01)
        scale_y = vertical_spacing * (n - 1) / max(pos_range[1], 0.01)
        pos[:, 0] *= scale_x
        pos[:, 1] *= scale_y

        # Center at origin
        pos -= pos.mean(axis=0)

    return {labels[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def _apply_repulsive_forces(
    pos: np.ndarray,
    displacement: np.ndarray,
    n: int,
    k: float,
) -> None:
    """Apply repulsive forces between all node pairs.

    Parameters
    ----------
    pos : np.ndarray
        Current positions.
    displacement : np.ndarray
        Displacement array to update.
    n : int
        Number of nodes.
    k : float
        Optimal distance.
    """
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[i] - pos[j]
            dist = max(float(np.linalg.norm(delta)), 0.01)
            force = k * k / dist
            direction = delta / dist
            displacement[i] += direction * force
            displacement[j] -= direction * force


def _apply_attractive_forces(
    pos: np.ndarray,
    displacement: np.ndarray,
    edge_pairs: list[tuple[int, int]],
    k: float,
) -> None:
    """Apply attractive forces along edges.

    Parameters
    ----------
    pos : np.ndarray
        Current positions.
    displacement : np.ndarray
        Displacement array to update.
    edge_pairs : list[tuple[int, int]]
        Edge index pairs.
    k : float
        Optimal distance.
    """
    for i, j in edge_pairs:
        delta = pos[j] - pos[i]
        dist = max(float(np.linalg.norm(delta)), 0.01)
        force = dist * dist / k
        direction = delta / dist
        displacement[i] += direction * force
        displacement[j] -= direction * force


def _apply_displacement(
    pos: np.ndarray,
    displacement: np.ndarray,
    n: int,
    temperature: float,
) -> None:
    """Apply capped displacement to node positions.

    Parameters
    ----------
    pos : np.ndarray
        Current positions to update.
    displacement : np.ndarray
        Computed displacements.
    n : int
        Number of nodes.
    temperature : float
        Current temperature cap.
    """
    for i in range(n):
        disp_norm = max(float(np.linalg.norm(displacement[i])), 0.01)
        pos[i] += displacement[i] / disp_norm * min(disp_norm, temperature)


def _manual_layout(
    graph: KineticGraph,
    manual_positions: NodePositions | None,
) -> NodePositions:
    """Validate and return user-supplied positions.

    Parameters
    ----------
    graph : KineticGraph
        The graph (for validation).
    manual_positions : NodePositions | None
        User-supplied positions.

    Returns
    -------
    NodePositions
        The validated positions.

    Raises
    ------
    ValueError
        If positions are None or missing for compartment nodes.
    """
    if manual_positions is None:
        msg = "Manual layout requires 'manual_positions' to be provided."
        raise ValueError(msg)

    compartment_labels = {n.label for n in graph.compartment_nodes()}
    missing = compartment_labels - set(manual_positions)
    if missing:
        msg = f"Manual positions missing for nodes: {sorted(missing)}"
        raise ValueError(msg)

    return dict(manual_positions)
