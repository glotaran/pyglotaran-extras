"""Tests for _kinetic_graph module."""

from __future__ import annotations

import pytest

from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import Transition
from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticEdge
from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticGraph


def _make_sequential_transitions() -> list[Transition]:
    """Create transitions for a simple sequential decay: A -> B -> C -> GS."""
    return [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("B", "C", 0.3, "rates.k_BC", False, "mc1"),
        Transition("C", "GS1", 0.1, "rates.k_C", True, "mc1"),
    ]


def _make_parallel_transitions() -> list[Transition]:
    """Create transitions for a parallel decay: A -> GS, B -> GS, C -> GS."""
    return [
        Transition("A", "GS1", 0.5, "rates.k_A", True, "mc1"),
        Transition("B", "GS2", 0.3, "rates.k_B", True, "mc1"),
        Transition("C", "GS3", 0.1, "rates.k_C", True, "mc1"),
    ]


def _make_cyclic_transitions() -> list[Transition]:
    """Create transitions with a cycle: A -> B -> C -> A."""
    return [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("B", "C", 0.3, "rates.k_BC", False, "mc1"),
        Transition("C", "A", 0.2, "rates.k_CA", False, "mc1"),
    ]


class TestKineticEdgeFormatRate:
    """Tests for KineticEdge.format_rate method."""

    def test_ns_unit_conversion(self) -> None:
        """Rate in ps⁻¹ should be converted to ns⁻¹."""
        edge = KineticEdge("A", "B", 0.5, "rates.k_AB")
        result = edge.format_rate(unit="ns")
        assert "500" in result
        assert "ns" in result

    def test_ps_unit_no_conversion(self) -> None:
        """Rate in ps⁻¹ should remain unchanged."""
        edge = KineticEdge("A", "B", 0.5, "rates.k_AB")
        result = edge.format_rate(unit="ps")
        assert "0.50" in result
        assert "ps" in result

    def test_decimal_places(self) -> None:
        """Explicit decimal places should be applied."""
        edge = KineticEdge("A", "B", 0.123456, "rates.k_AB")
        result = edge.format_rate(unit="ps", decimal_places=3)
        assert "0.123" in result

    def test_show_label(self) -> None:
        """Parameter short label should be shown when requested."""
        edge = KineticEdge("A", "B", 0.5, "rates.k_AB")
        result = edge.format_rate(unit="ns", show_label=True)
        assert "k_AB" in result
        assert "=" in result

    def test_smart_rounding_large(self) -> None:
        """Values >= 1 should be rounded to 0 decimal places."""
        edge = KineticEdge("A", "B", 0.005, "rates.k_AB")
        result = edge.format_rate(unit="ns")
        # 0.005 * 1000 = 5.0, should show as "5"
        assert "5" in result

    def test_smart_rounding_small(self) -> None:
        """Values < 1 should show 2 decimal places."""
        edge = KineticEdge("A", "B", 0.0005, "rates.k_AB")
        result = edge.format_rate(unit="ns")
        # 0.0005 * 1000 = 0.5, should show as "0.50"
        assert "0.50" in result

    def test_include_unit_false(self) -> None:
        """When include_unit is False, unit suffix should be omitted."""
        edge = KineticEdge("A", "B", 0.5, "rates.k_AB")
        result = edge.format_rate(unit="ns", include_unit=False)
        assert result == "500"
        assert "ns" not in result

    def test_include_unit_false_with_label(self) -> None:
        """include_unit=False should still work with show_label=True."""
        edge = KineticEdge("A", "B", 0.5, "rates.k_AB")
        result = edge.format_rate(unit="ns", include_unit=False, show_label=True)
        assert "k_AB" in result
        assert "=" in result
        assert "ns" not in result

    def test_include_unit_true_default(self) -> None:
        """Default include_unit=True should include the unit suffix."""
        edge = KineticEdge("A", "B", 0.5, "rates.k_AB")
        result = edge.format_rate(unit="ns")
        assert "ns" in result


class TestKineticGraphFromTransitions:
    """Tests for KineticGraph.from_transitions factory."""

    def test_sequential_creates_correct_nodes(self) -> None:
        """Sequential transitions create 3 compartment + 1 GS nodes."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        assert len(graph.compartment_nodes()) == 3
        assert len(graph.ground_state_nodes()) == 1

    def test_sequential_creates_correct_edges(self) -> None:
        """Sequential transitions create 3 edges (2 transfers + 1 GS decay)."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        assert len(graph.edges) == 3

    def test_parallel_creates_isolated_nodes(self) -> None:
        """Parallel decay creates 3 compartment + 3 GS nodes."""
        graph = KineticGraph.from_transitions(_make_parallel_transitions())
        assert len(graph.compartment_nodes()) == 3
        assert len(graph.ground_state_nodes()) == 3

    def test_node_megacomplex_labels(self) -> None:
        """Nodes should carry their megacomplex label."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        for node in graph.compartment_nodes():
            assert "mc1" in node.megacomplex_labels

    def test_merge_ground_state_decays(self) -> None:
        """Duplicate GS decays from same compartment+rate should be merged."""
        transitions = [
            Transition("A", "GS1", 0.5, "rates.k_A", True, "mc1"),
            Transition("A", "GS2", 0.5, "rates.k_A", True, "mc2"),
        ]
        graph = KineticGraph.from_transitions(transitions, merge_ground_state_decays=True)
        assert len(graph.ground_state_nodes()) == 1

    def test_no_merge_ground_state_decays(self) -> None:
        """When merging is disabled, each GS transition creates a separate node."""
        transitions = [
            Transition("A", "GS1", 0.5, "rates.k_A", True, "mc1"),
            Transition("A", "GS2", 0.5, "rates.k_A", True, "mc2"),
        ]
        graph = KineticGraph.from_transitions(transitions, merge_ground_state_decays=False)
        assert len(graph.ground_state_nodes()) == 2


class TestKineticGraphAnalysis:
    """Tests for graph analysis methods."""

    def test_is_dag_sequential(self) -> None:
        """Sequential decay graph is a DAG."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        assert graph.is_dag() is True

    def test_is_dag_parallel(self) -> None:
        """Parallel decay graph is a DAG."""
        graph = KineticGraph.from_transitions(_make_parallel_transitions())
        assert graph.is_dag() is True

    def test_is_dag_cyclic(self) -> None:
        """Cyclic graph is not a DAG."""
        graph = KineticGraph.from_transitions(_make_cyclic_transitions())
        assert graph.is_dag() is False

    def test_topological_sort_sequential(self) -> None:
        """Topological sort of sequential graph returns correct order."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        order = graph.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_topological_sort_cyclic_raises(self) -> None:
        """Topological sort of cyclic graph raises ValueError."""
        graph = KineticGraph.from_transitions(_make_cyclic_transitions())
        with pytest.raises(ValueError, match="cycles"):
            graph.topological_sort()

    def test_successors(self) -> None:
        """Successors returns correct neighbors."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        succs = graph.successors("A")
        assert "B" in succs

    def test_predecessors(self) -> None:
        """Predecessors returns correct neighbors."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        preds = graph.predecessors("B")
        assert "A" in preds

    def test_edges_between(self) -> None:
        """edges_between returns edges for specific node pair."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        edges = graph.edges_between("A", "B")
        assert len(edges) == 1
        assert edges[0].rate_constant_ps_inverse == 0.5

    def test_edges_between_no_match(self) -> None:
        """edges_between returns empty list when no edges exist."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        edges = graph.edges_between("A", "C")
        assert len(edges) == 0

    def test_ground_state_edges_for_node(self) -> None:
        """ground_state_edges_for_node returns GS decay edges."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        gs_edges = graph.ground_state_edges_for_node("C")
        assert len(gs_edges) == 1
        assert gs_edges[0].rate_constant_ps_inverse == 0.1

    def test_ground_state_edges_for_non_decaying_node(self) -> None:
        """Nodes without GS decay should return empty list."""
        graph = KineticGraph.from_transitions(_make_sequential_transitions())
        gs_edges = graph.ground_state_edges_for_node("A")
        assert len(gs_edges) == 0


class TestKineticGraphNodeMerging:
    """Tests for node merging when same compartment appears in multiple transitions."""

    def test_shared_compartment_merged(self) -> None:
        """Same compartment from different megacomplexes should be merged."""
        transitions = [
            Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
            Transition("A", "C", 0.3, "rates.k_AC", False, "mc2"),
        ]
        graph = KineticGraph.from_transitions(transitions)
        assert len(graph.compartment_nodes()) == 3  # A, B, C
        node_a = graph.nodes["A"]
        assert node_a.megacomplex_labels == {"mc1", "mc2"}
