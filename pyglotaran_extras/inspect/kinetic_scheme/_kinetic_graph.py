"""Layer 2: Lightweight directed graph datastructure for kinetic decay schemes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from pyglotaran_extras.inspect.kinetic_scheme._constants import GROUND_STATE_PREFIX
from pyglotaran_extras.inspect.kinetic_scheme._constants import PS_INVERSE_TO_NS_INVERSE

if TYPE_CHECKING:
    from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import Transition


@dataclass
class KineticNode:
    """A node in the kinetic scheme graph.

    Parameters
    ----------
    label : str
        Unique identifier for the node.
    display_label : str | None
        User-provided display name. Falls back to ``label`` if None.
    is_ground_state : bool
        Whether this node represents the ground state.
    megacomplex_labels : set[str]
        Which megacomplex(es) this node participates in.
    color : str | None
        Node fill color override. If None, uses the default from config.
    """

    label: str
    display_label: str | None = None
    is_ground_state: bool = False
    megacomplex_labels: set[str] = field(default_factory=set)
    color: str | None = None


@dataclass
class KineticEdge:
    """A directed edge representing a rate constant transition.

    Parameters
    ----------
    source : str
        Source node label.
    target : str
        Target node label.
    rate_constant_ps_inverse : float
        Raw rate constant in ps⁻¹. Never rounded.
    parameter_label : str
        Full parameter label from the pyglotaran model.
    """

    source: str
    target: str
    rate_constant_ps_inverse: float
    parameter_label: str

    def format_rate(
        self,
        unit: Literal["ps", "ns"] = "ns",
        decimal_places: int | None = None,
        show_label: bool = False,
        include_unit: bool = True,
    ) -> str:
        """Format the rate constant for display.

        Parameters
        ----------
        unit : Literal["ps", "ns"]
            Display unit. Defaults to ``"ns"``.
        decimal_places : int | None
            Number of decimal places. If None, uses smart rounding.
        show_label : bool
            Whether to prefix with the short parameter label (e.g., ``"k21 = "``).
        include_unit : bool
            Whether to append the unit suffix (e.g., ``"ns⁻¹"``). Defaults to True.

        Returns
        -------
        str
            Formatted rate constant string.
        """
        value = self.rate_constant_ps_inverse
        unit_suffix = "ps\u207b\u00b9"

        if unit == "ns":
            value *= PS_INVERSE_TO_NS_INVERSE
            unit_suffix = "ns\u207b\u00b9"

        if decimal_places is not None:
            formatted = f"{value:.{decimal_places}f}"
        elif abs(value) >= 1.0:
            formatted = f"{value:.0f}"
        else:
            formatted = f"{value:.2f}"

        result = f"{formatted} {unit_suffix}" if include_unit else formatted

        if show_label:
            short_label = self.parameter_label.split(".")[-1]
            result = f"{short_label} = {result}"

        return result


@dataclass
class KineticGraph:
    """Lightweight directed graph for kinetic decay schemes.

    Replaces ``networkx.DiGraph``. Adequate for the typical 3-20 node kinetic
    schemes encountered in pyglotaran models.

    Parameters
    ----------
    nodes : dict[str, KineticNode]
        Mapping from node label to KineticNode.
    edges : list[KineticEdge]
        All directed edges in the graph.
    """

    nodes: dict[str, KineticNode] = field(default_factory=dict)
    edges: list[KineticEdge] = field(default_factory=list)
    _adjacency: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _reverse_adjacency: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def add_node(self, node: KineticNode) -> None:
        """Add a node to the graph.

        Parameters
        ----------
        node : KineticNode
            The node to add. If a node with the same label already exists,
            its megacomplex labels are merged.
        """
        if node.label in self.nodes:
            self.nodes[node.label].megacomplex_labels.update(node.megacomplex_labels)
        else:
            self.nodes[node.label] = node
            self._adjacency.setdefault(node.label, [])
            self._reverse_adjacency.setdefault(node.label, [])

    def add_edge(self, edge: KineticEdge) -> None:
        """Add a directed edge to the graph.

        Parameters
        ----------
        edge : KineticEdge
            The directed edge to add. Source and target nodes must already exist.
        """
        self.edges.append(edge)
        self._adjacency[edge.source].append(edge.target)
        self._reverse_adjacency[edge.target].append(edge.source)

    def successors(self, label: str) -> list[str]:
        """Return labels of nodes reachable from the given node via outgoing edges.

        Parameters
        ----------
        label : str
            The node label.

        Returns
        -------
        list[str]
            Labels of successor nodes.
        """
        return list(self._adjacency.get(label, []))

    def predecessors(self, label: str) -> list[str]:
        """Return labels of nodes with edges pointing to the given node.

        Parameters
        ----------
        label : str
            The node label.

        Returns
        -------
        list[str]
            Labels of predecessor nodes.
        """
        return list(self._reverse_adjacency.get(label, []))

    def compartment_nodes(self) -> list[KineticNode]:
        """Return all non-ground-state nodes.

        Returns
        -------
        list[KineticNode]
            All nodes where ``is_ground_state`` is False.
        """
        return [n for n in self.nodes.values() if not n.is_ground_state]

    def ground_state_nodes(self) -> list[KineticNode]:
        """Return all ground state nodes.

        Returns
        -------
        list[KineticNode]
            All nodes where ``is_ground_state`` is True.
        """
        return [n for n in self.nodes.values() if n.is_ground_state]

    def edges_between(self, source: str, target: str) -> list[KineticEdge]:
        """Return all edges from source to target.

        Parameters
        ----------
        source : str
            Source node label.
        target : str
            Target node label.

        Returns
        -------
        list[KineticEdge]
            All edges between the specified nodes.
        """
        return [e for e in self.edges if e.source == source and e.target == target]

    def ground_state_edges_for_node(self, label: str) -> list[KineticEdge]:
        """Return all edges from the given node to ground state nodes.

        Parameters
        ----------
        label : str
            The compartment node label.

        Returns
        -------
        list[KineticEdge]
            Ground state decay edges for this node.
        """
        return [
            e
            for e in self.edges
            if e.source == label
            and e.target in self.nodes
            and self.nodes[e.target].is_ground_state
        ]

    def is_dag(self) -> bool:
        """Check if the graph (excluding ground state nodes) is a directed acyclic graph.

        Uses DFS-based cycle detection with WHITE/GRAY/BLACK coloring.

        Returns
        -------
        bool
            True if the compartment subgraph has no cycles.
        """
        compartment_labels = {n.label for n in self.compartment_nodes()}
        white = set(compartment_labels)
        gray: set[str] = set()

        def _has_cycle(node: str) -> bool:
            """Check if a cycle is reachable from ``node`` via DFS."""
            white.discard(node)
            gray.add(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in compartment_labels:
                    continue
                if neighbor in gray:
                    return True
                if neighbor in white and _has_cycle(neighbor):
                    return True
            gray.discard(node)
            return False

        return all(not (node in white and _has_cycle(node)) for node in sorted(compartment_labels))

    def topological_sort(self) -> list[str]:
        """Topological sort of compartment nodes using Kahn's algorithm.

        Returns
        -------
        list[str]
            Node labels in topological order.

        Raises
        ------
        ValueError
            If the compartment subgraph contains cycles.
        """
        compartment_labels = {n.label for n in self.compartment_nodes()}

        in_degree: dict[str, int] = dict.fromkeys(compartment_labels, 0)
        for edge in self.edges:
            if edge.source in compartment_labels and edge.target in compartment_labels:
                in_degree[edge.target] += 1

        queue = deque(sorted(label for label, deg in in_degree.items() if deg == 0))
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(self._adjacency.get(node, [])):
                if neighbor not in compartment_labels:
                    continue
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(compartment_labels):
            msg = "Compartment subgraph contains cycles; topological sort is not possible."
            raise ValueError(msg)

        return result

    @classmethod
    def from_transitions(
        cls,
        transitions: list[Transition],
        *,
        merge_ground_state_decays: bool = True,
    ) -> KineticGraph:
        """Construct a KineticGraph from a list of Transitions.

        Parameters
        ----------
        transitions : list[Transition]
            Transitions extracted from pyglotaran megacomplexes.
        merge_ground_state_decays : bool
            If True, ground state decays from the same compartment with the
            same rate constant are merged into a single edge. Defaults to True.

        Returns
        -------
        KineticGraph
            The constructed graph.
        """
        graph = cls()
        gs_dedup: dict[tuple[str, float], str] = {}
        gs_counter = 0

        for transition in transitions:
            # Ensure source compartment node exists
            graph.add_node(
                KineticNode(
                    label=transition.source,
                    megacomplex_labels={transition.megacomplex_label},
                )
            )

            if transition.is_ground_state_decay:
                dedup_key = (transition.source, transition.rate_constant)
                if merge_ground_state_decays and dedup_key in gs_dedup:
                    # Already have this ground state decay, skip
                    continue

                gs_counter += 1
                gs_label = f"{GROUND_STATE_PREFIX}{gs_counter}"
                gs_dedup[dedup_key] = gs_label

                graph.add_node(
                    KineticNode(
                        label=gs_label,
                        is_ground_state=True,
                        megacomplex_labels={transition.megacomplex_label},
                    )
                )
                graph.add_edge(
                    KineticEdge(
                        source=transition.source,
                        target=gs_label,
                        rate_constant_ps_inverse=transition.rate_constant,
                        parameter_label=transition.parameter_label,
                    )
                )
            else:
                # Ensure target compartment node exists
                graph.add_node(
                    KineticNode(
                        label=transition.target,
                        megacomplex_labels={transition.megacomplex_label},
                    )
                )
                graph.add_edge(
                    KineticEdge(
                        source=transition.source,
                        target=transition.target,
                        rate_constant_ps_inverse=transition.rate_constant,
                        parameter_label=transition.parameter_label,
                    )
                )

        return graph
