"""Tests for _layout module."""

from __future__ import annotations

import hashlib

import pytest

from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import Transition
from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticGraph
from pyglotaran_extras.inspect.kinetic_scheme._layout import LayoutAlgorithm
from pyglotaran_extras.inspect.kinetic_scheme._layout import _find_connected_components
from pyglotaran_extras.inspect.kinetic_scheme._layout import _node_sort_index
from pyglotaran_extras.inspect.kinetic_scheme._layout import compute_layout


def _make_sequential_graph() -> KineticGraph:
    """Create a sequential decay graph: A -> B -> C -> GS."""
    transitions = [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("B", "C", 0.3, "rates.k_BC", False, "mc1"),
        Transition("C", "GS1", 0.1, "rates.k_C", True, "mc1"),
    ]
    return KineticGraph.from_transitions(transitions)


def _make_branching_graph() -> KineticGraph:
    """Create a branching graph: A -> B, A -> C."""
    transitions = [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("A", "C", 0.3, "rates.k_AC", False, "mc1"),
        Transition("B", "GS1", 0.1, "rates.k_B", True, "mc1"),
        Transition("C", "GS2", 0.1, "rates.k_C", True, "mc1"),
    ]
    return KineticGraph.from_transitions(transitions)


def _make_parallel_graph() -> KineticGraph:
    """Create a parallel decay graph: A -> GS, B -> GS, C -> GS."""
    transitions = [
        Transition("A", "GS1", 0.5, "rates.k_A", True, "mc1"),
        Transition("B", "GS2", 0.3, "rates.k_B", True, "mc1"),
        Transition("C", "GS3", 0.1, "rates.k_C", True, "mc1"),
    ]
    return KineticGraph.from_transitions(transitions)


def _make_cyclic_graph() -> KineticGraph:
    """Create a cyclic graph: A -> B -> C -> A."""
    transitions = [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("B", "C", 0.3, "rates.k_BC", False, "mc1"),
        Transition("C", "A", 0.2, "rates.k_CA", False, "mc1"),
    ]
    return KineticGraph.from_transitions(transitions)


class TestHierarchicalLayout:
    """Tests for the hierarchical layout algorithm."""

    def test_sequential_top_to_bottom(self) -> None:
        """Sequential graph should lay out nodes top to bottom."""
        graph = _make_sequential_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        # A should be above B, B above C
        assert positions["A"][1] > positions["B"][1]
        assert positions["B"][1] > positions["C"][1]

    def test_branching_siblings_side_by_side(self) -> None:
        """Branching graph should place siblings at the same y level."""
        graph = _make_branching_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        # B and C are both successors of A, should be at the same y level
        assert abs(positions["B"][1] - positions["C"][1]) < 0.01

        # A should be above B and C
        assert positions["A"][1] > positions["B"][1]

    def test_all_compartment_nodes_have_positions(self) -> None:
        """Every compartment node should get a position."""
        graph = _make_sequential_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for node in graph.compartment_nodes():
            assert node.label in positions

    def test_ground_state_below_parent(self) -> None:
        """Ground state node should be positioned below its parent."""
        graph = _make_sequential_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for gs_node in graph.ground_state_nodes():
            parents = graph.predecessors(gs_node.label)
            assert len(parents) > 0
            parent_y = positions[parents[0]][1]
            gs_y = positions[gs_node.label][1]
            assert gs_y < parent_y

    def test_no_overlapping_positions(self) -> None:
        """No two compartment nodes should have the same position."""
        graph = _make_branching_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        compartment_positions = [positions[n.label] for n in graph.compartment_nodes()]
        for i, pos_i in enumerate(compartment_positions):
            for j, pos_j in enumerate(compartment_positions):
                if i != j:
                    assert pos_i != pos_j

    def test_deterministic_output(self) -> None:
        """Same input should always produce the same positions."""
        graph = _make_sequential_graph()
        pos1 = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)
        pos2 = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for label in pos1:
            assert pos1[label] == pos2[label]

    def test_node_sort_index_is_deterministic(self) -> None:
        """Node sort index should use deterministic hashing."""
        label = "species_2"
        digest = hashlib.md5(label.encode(), usedforsecurity=False).digest()
        expected = int.from_bytes(digest[:4], "big") / 4294967296.0
        actual = _node_sort_index(label)
        assert actual == expected
        assert 0.0 <= actual < 1.0

    def test_parallel_nodes_side_by_side(self) -> None:
        """Parallel decay nodes (all isolated) should be on the same row."""
        graph = _make_parallel_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        y_values = {positions[n.label][1] for n in graph.compartment_nodes()}
        # All compartment nodes should be at the same y level
        assert len(y_values) == 1


class TestHierarchicalLayoutPreference:
    """Tests for horizontal_layout_preference parameter."""

    def test_preference_orders_nodes(self) -> None:
        """Layout preference should control left-to-right ordering."""
        graph = _make_parallel_graph()
        positions = compute_layout(
            graph,
            LayoutAlgorithm.HIERARCHICAL,
            horizontal_layout_preference="C|B|A",
        )
        # C should be leftmost, then B, then A
        assert positions["C"][0] < positions["B"][0]
        assert positions["B"][0] < positions["A"][0]

    def test_partial_preference(self) -> None:
        """Nodes not in preference list should still get positions."""
        graph = _make_parallel_graph()
        positions = compute_layout(
            graph,
            LayoutAlgorithm.HIERARCHICAL,
            horizontal_layout_preference="B|A",
        )
        # B before A
        assert positions["B"][0] < positions["A"][0]
        # C still gets a position
        assert "C" in positions


class TestCyclicLayout:
    """Tests for layout of cyclic graphs."""

    def test_cyclic_graph_gets_positions(self) -> None:
        """Cyclic graph should still produce positions for all nodes."""
        graph = _make_cyclic_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for node in graph.compartment_nodes():
            assert node.label in positions

    def test_cyclic_deterministic(self) -> None:
        """Cyclic layout should be deterministic."""
        graph = _make_cyclic_graph()
        pos1 = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)
        pos2 = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for label in pos1:
            assert pos1[label] == pos2[label]


class TestSpringLayout:
    """Tests for the spring layout algorithm."""

    def test_all_nodes_positioned(self) -> None:
        """All compartment nodes should get positions."""
        graph = _make_sequential_graph()
        positions = compute_layout(graph, LayoutAlgorithm.SPRING)

        for node in graph.compartment_nodes():
            assert node.label in positions

    def test_deterministic_with_seed(self) -> None:
        """Spring layout with same seed should be deterministic."""
        graph = _make_sequential_graph()
        pos1 = compute_layout(graph, LayoutAlgorithm.SPRING)
        pos2 = compute_layout(graph, LayoutAlgorithm.SPRING)

        for label in pos1:
            assert abs(pos1[label][0] - pos2[label][0]) < 1e-10
            assert abs(pos1[label][1] - pos2[label][1]) < 1e-10

    def test_ground_state_below_parent(self) -> None:
        """Ground state nodes should be below their parent in spring layout."""
        graph = _make_sequential_graph()
        positions = compute_layout(graph, LayoutAlgorithm.SPRING)

        for gs_node in graph.ground_state_nodes():
            parents = graph.predecessors(gs_node.label)
            if parents and parents[0] in positions:
                parent_y = positions[parents[0]][1]
                gs_y = positions[gs_node.label][1]
                assert gs_y < parent_y


class TestManualLayout:
    """Tests for manual layout."""

    def test_passthrough(self) -> None:
        """Manual positions should be returned unchanged."""
        graph = _make_sequential_graph()
        manual = {"A": (0.0, 2.0), "B": (1.0, 1.0), "C": (2.0, 0.0)}
        positions = compute_layout(graph, LayoutAlgorithm.MANUAL, manual_positions=manual)
        assert positions["A"] == (0.0, 2.0)
        assert positions["B"] == (1.0, 1.0)
        assert positions["C"] == (2.0, 0.0)

    def test_ground_state_below_parent(self) -> None:
        """Manual layout should also position ground state nodes below parents."""
        graph = _make_sequential_graph()
        manual = {"A": (0.0, 2.0), "B": (1.0, 1.0), "C": (2.0, 0.0)}
        positions = compute_layout(graph, LayoutAlgorithm.MANUAL, manual_positions=manual)

        assert "GS1" in positions
        assert positions["GS1"][0] == positions["C"][0]
        assert positions["GS1"][1] < positions["C"][1]

    def test_avoids_ground_state_arrow_overlap(self) -> None:
        """Manual layout should nudge a node below a GS-decaying parent."""
        graph = KineticGraph.from_transitions(
            [
                Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
                Transition("A", "GS1", 0.1, "rates.k_A", True, "mc1"),
            ]
        )
        manual = {"A": (0.0, 1.0), "B": (0.0, 0.0)}
        positions = compute_layout(graph, LayoutAlgorithm.MANUAL, manual_positions=manual)

        assert positions["B"][0] > 0.0
        assert positions["B"][1] == 0.0

    def test_missing_positions_raises(self) -> None:
        """Missing node positions should raise ValueError."""
        graph = _make_sequential_graph()
        with pytest.raises(ValueError, match="missing"):
            compute_layout(graph, LayoutAlgorithm.MANUAL, manual_positions={"A": (0.0, 0.0)})

    def test_none_positions_raises(self) -> None:
        """None manual_positions should raise ValueError."""
        graph = _make_sequential_graph()
        with pytest.raises(ValueError, match="manual_positions"):
            compute_layout(graph, LayoutAlgorithm.MANUAL)


class TestEmptyGraph:
    """Tests for empty graph layout."""

    def test_empty_graph(self) -> None:
        """Empty graph should produce empty positions."""
        graph = KineticGraph.from_transitions([])
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)
        assert positions == {}


def _make_two_component_graph() -> KineticGraph:
    """Create two disconnected components: A -> B -> C, D -> E -> F."""
    transitions = [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("B", "C", 0.3, "rates.k_BC", False, "mc1"),
        Transition("C", "GS1", 0.1, "rates.k_C", True, "mc1"),
        Transition("D", "E", 0.4, "rates.k_DE", False, "mc2"),
        Transition("E", "F", 0.2, "rates.k_EF", False, "mc2"),
        Transition("F", "GS2", 0.1, "rates.k_F", True, "mc2"),
    ]
    return KineticGraph.from_transitions(transitions)


def _make_three_component_graph() -> KineticGraph:
    """Create three disconnected components: A->B, C->D, E->F."""
    transitions = [
        Transition("A", "B", 0.5, "rates.k_AB", False, "mc1"),
        Transition("B", "GS1", 0.1, "rates.k_B", True, "mc1"),
        Transition("C", "D", 0.3, "rates.k_CD", False, "mc2"),
        Transition("D", "GS2", 0.1, "rates.k_D", True, "mc2"),
        Transition("E", "F", 0.2, "rates.k_EF", False, "mc3"),
        Transition("F", "GS3", 0.1, "rates.k_F", True, "mc3"),
    ]
    return KineticGraph.from_transitions(transitions)


class TestFindConnectedComponents:
    """Tests for _find_connected_components."""

    def test_single_component(self) -> None:
        """Connected graph should return one component."""
        graph = _make_sequential_graph()
        components = _find_connected_components(graph)
        assert len(components) == 1
        assert components[0] == {"A", "B", "C"}

    def test_disconnected_components(self) -> None:
        """Disconnected graph should return correct number of components."""
        graph = _make_two_component_graph()
        components = _find_connected_components(graph)
        assert len(components) == 2
        assert {"A", "B", "C"} in components
        assert {"D", "E", "F"} in components

    def test_three_components(self) -> None:
        """Three disconnected components should all be detected."""
        graph = _make_three_component_graph()
        components = _find_connected_components(graph)
        assert len(components) == 3

    def test_empty_graph(self) -> None:
        """Empty graph should return empty list."""
        graph = KineticGraph.from_transitions([])
        components = _find_connected_components(graph)
        assert components == []

    def test_parallel_decay_components(self) -> None:
        """Parallel decay (no inter-compartment edges) gives separate components."""
        graph = _make_parallel_graph()
        components = _find_connected_components(graph)
        # A, B, C are each isolated (only connected to their own GS node)
        assert len(components) == 3


class TestConnectedComponentLayout:
    """Tests for side-by-side layout of disconnected components."""

    def test_two_components_separated(self) -> None:
        """Two components should have non-overlapping x ranges."""
        graph = _make_two_component_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        comp1_x = [positions[n][0] for n in ["A", "B", "C"]]
        comp2_x = [positions[n][0] for n in ["D", "E", "F"]]

        # Component 1's rightmost x should be less than component 2's leftmost x
        assert max(comp1_x) < min(comp2_x)

    def test_three_components_separated(self) -> None:
        """Three components should all have non-overlapping x ranges."""
        graph = _make_three_component_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        comp1_x = [positions[n][0] for n in ["A", "B"]]
        comp2_x = [positions[n][0] for n in ["C", "D"]]
        comp3_x = [positions[n][0] for n in ["E", "F"]]

        assert max(comp1_x) < min(comp2_x)
        assert max(comp2_x) < min(comp3_x)

    def test_single_component_unchanged(self) -> None:
        """Single-component graph should produce same layout as before."""
        graph = _make_sequential_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        # Verify A -> B -> C ordering is preserved
        assert positions["A"][1] > positions["B"][1]
        assert positions["B"][1] > positions["C"][1]

    def test_component_gap_respected(self) -> None:
        """Gap between components should be approximately component_gap."""
        graph = _make_two_component_graph()
        gap = 5.0
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL, component_gap=gap)

        comp1_x = [positions[n][0] for n in ["A", "B", "C"]]
        comp2_x = [positions[n][0] for n in ["D", "E", "F"]]

        actual_gap = min(comp2_x) - max(comp1_x)
        assert abs(actual_gap - gap) < 0.01

    def test_all_nodes_get_positions(self) -> None:
        """All compartment nodes across components should get positions."""
        graph = _make_two_component_graph()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for node in graph.compartment_nodes():
            assert node.label in positions

    def test_deterministic(self) -> None:
        """Multi-component layout should be deterministic."""
        graph = _make_two_component_graph()
        pos1 = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)
        pos2 = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        for label in pos1:
            assert pos1[label] == pos2[label]
