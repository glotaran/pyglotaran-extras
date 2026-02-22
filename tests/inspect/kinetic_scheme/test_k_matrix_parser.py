"""Tests for _k_matrix_parser module."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from glotaran.testing.simulated_data.parallel_spectral_decay import SCHEME as SCHEME_PAR
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME as SCHEME_SEQ

from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import Transition
from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import extract_dataset_transitions
from pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser import extract_transitions


class TestExtractTransitionsSequential:
    """Tests for sequential decay model extraction."""

    def test_returns_transitions(self) -> None:
        """Extract transitions from sequential decay megacomplex."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        assert len(transitions) > 0
        assert all(isinstance(t, Transition) for t in transitions)

    def test_transition_count(self) -> None:
        """Sequential decay with 3 species produces 2 off-diagonal + 1 diagonal."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        # species_1 -> species_2, species_2 -> species_3, species_3 -> GS
        assert len(transitions) == 3

    def test_ground_state_decay_identified(self) -> None:
        """Diagonal entries are marked as ground state decays."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        gs_transitions = [t for t in transitions if t.is_ground_state_decay]
        assert len(gs_transitions) == 1
        assert gs_transitions[0].source == "species_3"
        assert gs_transitions[0].target.startswith("GS")

    def test_off_diagonal_transfers(self) -> None:
        """Off-diagonal entries are identified as transfers."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        transfers = [t for t in transitions if not t.is_ground_state_decay]
        assert len(transfers) == 2
        sources = {t.source for t in transfers}
        targets = {t.target for t in transfers}
        assert "species_1" in sources
        assert "species_2" in sources
        assert "species_2" in targets
        assert "species_3" in targets

    def test_rate_constants_are_raw(self) -> None:
        """Rate constants should be stored as raw ps⁻¹ values, not rounded."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        for t in transitions:
            assert isinstance(t.rate_constant, float)
            # Values should be the raw float, not integer-rounded
            assert t.rate_constant > 0

    def test_parameter_labels_present(self) -> None:
        """Each transition should have a parameter label."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        for t in transitions:
            assert t.parameter_label != ""
            assert "." in t.parameter_label  # e.g. "rates.species_1"

    def test_megacomplex_label_set(self) -> None:
        """Each transition should carry its megacomplex label."""
        transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        for t in transitions:
            assert t.megacomplex_label == "megacomplex_sequential_decay"

    def test_accepts_list_input(self) -> None:
        """Megacomplexes parameter accepts a list."""
        transitions = extract_transitions(
            ["megacomplex_sequential_decay"], SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        assert len(transitions) == 3


class TestExtractTransitionsParallel:
    """Tests for parallel decay model extraction."""

    def test_all_ground_state_decays(self) -> None:
        """Parallel decay has only diagonal entries (all ground state decays)."""
        transitions = extract_transitions(
            "megacomplex_parallel_decay", SCHEME_PAR.model, SCHEME_PAR.parameters
        )
        assert len(transitions) == 3
        assert all(t.is_ground_state_decay for t in transitions)

    def test_distinct_species(self) -> None:
        """Each species has its own ground state decay."""
        transitions = extract_transitions(
            "megacomplex_parallel_decay", SCHEME_PAR.model, SCHEME_PAR.parameters
        )
        sources = {t.source for t in transitions}
        assert sources == {"species_1", "species_2", "species_3"}


class TestExtractTransitionsErrors:
    """Tests for error handling in extract_transitions."""

    def test_invalid_megacomplex_raises_value_error(self) -> None:
        """Non-existent megacomplex label raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            extract_transitions("nonexistent_mc", SCHEME_SEQ.model, SCHEME_SEQ.parameters)

    def test_non_decay_megacomplex_raises_type_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-decay megacomplex without get_k_matrix raises TypeError."""
        model = deepcopy(SCHEME_SEQ.model)
        mc_label = "megacomplex_coherent_artifact"
        model.megacomplex[mc_label] = SimpleNamespace(type="coherent-artifact")

        # Simulate filling a non-decay megacomplex object that does not expose
        # get_k_matrix, which should trigger the TypeError path.
        monkeypatch.setattr(
            "pyglotaran_extras.inspect.kinetic_scheme._k_matrix_parser.fill_item",
            lambda *_args, **_kwargs: SimpleNamespace(),
        )

        with pytest.raises(TypeError, match="does not support k-matrix extraction"):
            extract_transitions(mc_label, model, SCHEME_SEQ.parameters)


class TestExtractTransitionsFiltering:
    """Tests for parameter filtering."""

    def test_omit_parameters(self) -> None:
        """Parameters in omit_parameters set should be excluded."""
        all_transitions = extract_transitions(
            "megacomplex_sequential_decay", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        # Omit one parameter
        first_label = all_transitions[0].parameter_label
        filtered = extract_transitions(
            "megacomplex_sequential_decay",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            omit_parameters={first_label},
        )
        assert len(filtered) == len(all_transitions) - 1
        assert first_label not in {t.parameter_label for t in filtered}


class TestExtractDatasetTransitions:
    """Tests for dataset-level extraction."""

    def test_extracts_from_dataset(self) -> None:
        """Extraction via dataset name should find the decay megacomplex."""
        transitions = extract_dataset_transitions(
            "dataset_1", SCHEME_SEQ.model, SCHEME_SEQ.parameters
        )
        assert len(transitions) == 3

    def test_invalid_dataset_raises(self) -> None:
        """Non-existent dataset name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            extract_dataset_transitions(
                "nonexistent_dataset", SCHEME_SEQ.model, SCHEME_SEQ.parameters
            )

    def test_exclude_megacomplexes(self) -> None:
        """Excluded megacomplexes should not contribute transitions."""
        transitions = extract_dataset_transitions(
            "dataset_1",
            SCHEME_SEQ.model,
            SCHEME_SEQ.parameters,
            exclude_megacomplexes={"megacomplex_sequential_decay"},
        )
        assert len(transitions) == 0

    def test_missing_dataset_megacomplex_raises(self) -> None:
        """Dataset referencing undefined megacomplex raises ValueError."""
        model = deepcopy(SCHEME_SEQ.model)
        model.dataset["dataset_1"].megacomplex = ["missing_mc"]

        with pytest.raises(
            ValueError,
            match="referenced by dataset 'dataset_1' not found in model",
        ):
            extract_dataset_transitions("dataset_1", model, SCHEME_SEQ.parameters)
