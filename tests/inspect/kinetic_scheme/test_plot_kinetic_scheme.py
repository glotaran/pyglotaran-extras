"""Tests for plot_kinetic_scheme module."""

from __future__ import annotations

import pytest
from glotaran.testing.simulated_data.parallel_spectral_decay import SCHEME as SCHEME_PAR
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME as SCHEME_SEQ
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from pydantic import ValidationError

from pyglotaran_extras.inspect.kinetic_scheme.plot_kinetic_scheme import KineticSchemeConfig
from pyglotaran_extras.inspect.kinetic_scheme.plot_kinetic_scheme import NodeStyleConfig
from pyglotaran_extras.inspect.kinetic_scheme.plot_kinetic_scheme import _compute_text_color
from pyglotaran_extras.inspect.kinetic_scheme.plot_kinetic_scheme import (
    show_dataset_kinetic_scheme,
)
from pyglotaran_extras.inspect.kinetic_scheme.plot_kinetic_scheme import show_kinetic_scheme


class TestKineticSchemeConfig:
    """Tests for config validation."""

    def test_defaults(self) -> None:
        """Config should be constructable with all defaults."""
        config = KineticSchemeConfig()
        assert config.rate_unit == "ns"
        assert config.show_ground_state is False
        assert config.layout_algorithm == "hierarchical"

    def test_extra_forbid(self) -> None:
        """Extra fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            KineticSchemeConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_node_style_config(self) -> None:
        """NodeStyleConfig should accept all valid fields."""
        style = NodeStyleConfig(
            display_label="Custom", width=2.0, height=1.0, facecolor="red", fontsize=12
        )
        assert style.display_label == "Custom"
        assert style.facecolor == "red"

    def test_color_mapping(self) -> None:
        """Color mapping should accept valid input."""
        config = KineticSchemeConfig(color_mapping={"red": ["A", "B"], "blue": ["C"]})
        assert config.color_mapping["red"] == ["A", "B"]

    def test_ground_state_modes(self) -> None:
        """All ground state modes should be accepted."""
        config_off = KineticSchemeConfig(show_ground_state=False)
        assert config_off.show_ground_state is False

        config_shared = KineticSchemeConfig(show_ground_state="shared")
        assert config_shared.show_ground_state == "shared"

        config_per_mc = KineticSchemeConfig(show_ground_state="per_megacomplex")
        assert config_per_mc.show_ground_state == "per_megacomplex"


class TestComputeTextColor:
    """Tests for luminance-based text color computation."""

    def test_dark_background_white_text(self) -> None:
        """Dark backgrounds should yield white text."""
        assert _compute_text_color("black") == "white"
        assert _compute_text_color("#000000") == "white"
        assert _compute_text_color("navy") == "white"

    def test_light_background_black_text(self) -> None:
        """Light backgrounds should yield black text."""
        assert _compute_text_color("white") == "black"
        assert _compute_text_color("#FFFFFF") == "black"
        assert _compute_text_color("yellow") == "black"

    def test_medium_colors(self) -> None:
        """Medium colors should produce readable text."""
        # Just verify it returns a valid color string
        result = _compute_text_color("#4A90D9")
        assert result in ("black", "white")


class TestShowKineticScheme:
    """Tests for the main show_kinetic_scheme function."""

    def test_returns_figure_and_axes(self) -> None:
        """Function should return a (Figure, Axes) tuple."""
        fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        assert isinstance(fig, Figure)
        assert ax is not None

    def test_with_existing_axes(self) -> None:
        """Should render onto a provided axes."""
        fig_input = Figure()
        ax_input = fig_input.add_subplot(111)
        fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            ax=ax_input,
        )
        assert fig is fig_input
        assert ax is ax_input

    def test_sequential_has_nodes_and_edges(self) -> None:
        """Sequential decay should produce node patches and arrow patches."""
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        patches = ax.patches
        node_patches = [p for p in patches if isinstance(p, FancyBboxPatch)]
        arrow_patches = [p for p in patches if isinstance(p, FancyArrowPatch)]

        # 3 compartment nodes
        assert len(node_patches) == 3
        # 2 transfer + 1 GS decay = 3 arrows
        assert len(arrow_patches) == 3

    def test_parallel_has_nodes_and_edges(self) -> None:
        """Parallel decay should produce node patches and arrow patches."""
        _fig, ax = show_kinetic_scheme(
            "megacomplex_parallel_decay", SCHEME_PAR.model, SCHEME_PAR.parameters
        )
        patches = ax.patches
        node_patches = [p for p in patches if isinstance(p, FancyBboxPatch)]
        arrow_patches = [p for p in patches if isinstance(p, FancyArrowPatch)]

        # 3 compartment nodes
        assert len(node_patches) == 3
        # 3 GS decay arrows
        assert len(arrow_patches) == 3

    def test_with_title(self) -> None:
        """Title should be set on axes."""
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            title="Test Scheme",
        )
        assert ax.get_title() == "Test Scheme"

    def test_with_config(self) -> None:
        """Custom config should be accepted."""
        config = KineticSchemeConfig(
            rate_unit="ps",
            node_facecolor="green",
        )
        fig, _ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        assert isinstance(fig, Figure)

    def test_ground_state_shared(self) -> None:
        """Shared ground state bar should add a line to the plot."""
        config = KineticSchemeConfig(show_ground_state="shared")
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        # Should have at least one Line2D object (the bar)
        assert len(ax.lines) >= 1

    def test_ground_state_per_megacomplex(self) -> None:
        """Per-megacomplex ground state bar should add lines."""
        config = KineticSchemeConfig(show_ground_state="per_megacomplex")
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        assert len(ax.lines) >= 1

    def test_accepts_list_megacomplexes(self) -> None:
        """Function should accept a list of megacomplex labels."""
        fig, _ax = show_kinetic_scheme(
            ["megacomplex_sequential_decay"],
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
        )
        assert isinstance(fig, Figure)

    def test_savefig_svg(self, tmp_path) -> None:
        """Figure should be saveable as SVG."""
        fig, _ax = show_kinetic_scheme(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        svg_path = tmp_path / "test.svg"
        fig.savefig(str(svg_path), bbox_inches="tight")
        assert svg_path.exists()
        assert svg_path.stat().st_size > 0

    def test_savefig_png(self, tmp_path) -> None:
        """Figure should be saveable as PNG."""
        fig, _ax = show_kinetic_scheme(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        png_path = tmp_path / "test.png"
        fig.savefig(str(png_path), dpi=100)
        assert png_path.exists()
        assert png_path.stat().st_size > 0


class TestShowDatasetKineticScheme:
    """Tests for the dataset-level entry point."""

    def test_returns_figure_and_axes(self) -> None:
        """Dataset entry point should return (Figure, Axes)."""
        fig, _ax = show_dataset_kinetic_scheme(
            "dataset_1", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        assert isinstance(fig, Figure)

    def test_exclude_megacomplexes(self) -> None:
        """Excluding all megacomplexes should produce empty figure."""
        _fig, ax = show_dataset_kinetic_scheme(
            "dataset_1",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            exclude_megacomplexes={"megacomplex_sequential_decay"},
        )
        # No node patches when all megacomplexes excluded
        node_patches = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
        assert len(node_patches) == 0

    def test_invalid_dataset_raises(self) -> None:
        """Non-existent dataset should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            show_dataset_kinetic_scheme("nonexistent", SCHEME_SEQ.model, SCHEME_SEQ.parameters)


class TestShowKineticSchemeWithNodeStyles:
    """Tests for node style customization."""

    def test_custom_display_label(self) -> None:
        """Custom display labels should be rendered."""
        config = KineticSchemeConfig(
            node_styles={
                "species_1": NodeStyleConfig(display_label="S1"),
                "species_2": NodeStyleConfig(display_label="S2"),
            }
        )
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        # Check that text objects contain the custom labels
        texts = [t.get_text() for t in ax.texts]
        assert "S1" in texts
        assert "S2" in texts

    def test_color_mapping(self) -> None:
        """Color mapping should be applied to nodes."""
        config = KineticSchemeConfig(color_mapping={"red": ["species_1"], "blue": ["species_2"]})
        fig, _ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        assert isinstance(fig, Figure)

    def test_layout_preference(self) -> None:
        """Layout preference should be accepted."""
        config = KineticSchemeConfig(horizontal_layout_preference="species_3|species_2|species_1")
        fig, _ax = show_kinetic_scheme(
            "megacomplex_parallel_decay",
            SCHEME_PAR.model,
            SCHEME_PAR.parameters,
            config=config,
        )
        assert isinstance(fig, Figure)

    def test_rate_labels_shown(self) -> None:
        """When show_rate_labels is True, parameter names should appear in text."""
        config = KineticSchemeConfig(show_rate_labels=True)
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        texts = [t.get_text() for t in ax.texts]
        # At least one text should contain "=" from "k_label = value"
        assert any("=" in t for t in texts)


class TestRateFontsizeConfig:
    """Tests for rate_fontsize configuration."""

    def test_rate_fontsize_default(self) -> None:
        """Default rate_fontsize should be 9."""
        config = KineticSchemeConfig()
        assert config.rate_fontsize == 9

    def test_rate_fontsize_custom(self) -> None:
        """Custom rate_fontsize should be accepted."""
        config = KineticSchemeConfig(rate_fontsize=12)
        assert config.rate_fontsize == 12


class TestUnitAnnotation:
    """Tests for rate unit legend annotation."""

    def test_unit_annotation_present_by_default(self) -> None:
        """When show_rate_unit_per_label is False (default), unit annotation appears."""
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
        )
        texts = [t.get_text() for t in ax.texts]
        assert any("rates in" in t for t in texts)

    def test_no_unit_annotation_when_per_label(self) -> None:
        """When show_rate_unit_per_label is True, no annotation should appear."""
        config = KineticSchemeConfig(show_rate_unit_per_label=True)
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        texts = [t.get_text() for t in ax.texts]
        assert not any("rates in" in t for t in texts)

    def test_labels_omit_unit_by_default(self) -> None:
        """Rate labels should not contain unit suffix by default."""
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
        )
        texts = [t.get_text() for t in ax.texts]
        # Filter to just texts containing digits (rate values)
        # The only text with "ns" should be the annotation itself
        ns_texts = [t for t in texts if "ns\u207b\u00b9" in t]
        assert len(ns_texts) <= 1  # At most the annotation

    def test_labels_include_unit_when_per_label(self) -> None:
        """Rate labels should contain unit when show_rate_unit_per_label is True."""
        config = KineticSchemeConfig(show_rate_unit_per_label=True)
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        texts = [t.get_text() for t in ax.texts]
        # Rate labels should contain "ns⁻¹"
        ns_texts = [t for t in texts if "ns\u207b\u00b9" in t]
        assert len(ns_texts) >= 1

    def test_ps_unit_annotation(self) -> None:
        """Unit annotation should reflect the rate_unit setting."""
        config = KineticSchemeConfig(rate_unit="ps")
        _fig, ax = show_kinetic_scheme(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            config=config,
        )
        texts = [t.get_text() for t in ax.texts]
        assert any("ps\u207b\u00b9" in t for t in texts)


class TestLabelAntiOverlap:
    """Tests for label spreading when edges converge on a target."""

    def test_convergent_labels_at_distinct_positions(self) -> None:
        """Labels on edges sharing a target should be at different positions."""
        from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import Transition
        from pyglotaran_extras.inspect.kinetic_scheme._kinetic_graph import KineticGraph
        from pyglotaran_extras.inspect.kinetic_scheme._layout import LayoutAlgorithm
        from pyglotaran_extras.inspect.kinetic_scheme._layout import compute_layout
        from pyglotaran_extras.inspect.kinetic_scheme.plot_kinetic_scheme import _draw_all_edges

        # 3 edges converging on D
        transitions = [
            Transition("A", "D", 0.5, "rates.k_AD", False, "mc1"),
            Transition("B", "D", 0.3, "rates.k_BD", False, "mc1"),
            Transition("C", "D", 0.2, "rates.k_CD", False, "mc1"),
        ]
        graph = KineticGraph.from_transitions(transitions)
        config = KineticSchemeConfig()
        positions = compute_layout(graph, LayoutAlgorithm.HIERARCHICAL)

        fig = Figure()
        ax = fig.add_subplot(111)
        _draw_all_edges(ax, graph, positions, config)

        # Collect rate label positions (texts with digits = rate values)
        rate_texts = [t for t in ax.texts if any(c.isdigit() for c in t.get_text())]
        text_positions = [
            (round(t.get_position()[0], 6), round(t.get_position()[1], 6)) for t in rate_texts
        ]

        # All positions should be distinct
        assert len(text_positions) == len(set(text_positions))
