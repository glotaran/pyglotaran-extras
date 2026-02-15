"""Layer 1: Extract rate constant transitions from pyglotaran decay megacomplexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from glotaran.model.item import fill_item

from pyglotaran_extras.inspect.kinetic_scheme._constants import GROUND_STATE_PREFIX

if TYPE_CHECKING:
    from glotaran.model import Model
    from glotaran.parameter import Parameters


@dataclass(frozen=True)
class Transition:
    """A single rate constant transition extracted from a k-matrix.

    Parameters
    ----------
    source : str
        Source compartment name (from which population flows).
    target : str
        Target compartment name (to which population flows), or a generated
        ground state node label for diagonal entries.
    rate_constant : float
        Raw rate constant value in ps⁻¹. Never rounded during extraction.
    parameter_label : str
        Full parameter label from the pyglotaran model (e.g., ``"rates.k21"``).
    is_ground_state_decay : bool
        True when source == target in the k-matrix diagonal (total decay to ground state).
    megacomplex_label : str
        Label of the megacomplex this transition was extracted from.
    """

    source: str
    target: str
    rate_constant: float
    parameter_label: str
    is_ground_state_decay: bool
    megacomplex_label: str


def extract_transitions(
    megacomplexes: str | list[str],
    model: Model,
    parameters: Parameters,
    *,
    omit_parameters: set[str] | None = None,
) -> list[Transition]:
    """Extract rate constant transitions from one or more decay megacomplexes.

    Parameters
    ----------
    megacomplexes : str | list[str]
        One or more megacomplex labels to extract transitions from.
    model : Model
        A pyglotaran model containing the megacomplex definitions.
    parameters : Parameters
        Resolved parameters for the model.
    omit_parameters : set[str] | None
        Parameter labels to exclude from the result. Defaults to None.

    Returns
    -------
    list[Transition]
        All rate constant transitions found in the specified megacomplexes.

    Raises
    ------
    ValueError
        If a megacomplex label is not found in the model.
    TypeError
        If a megacomplex does not support k-matrix extraction (i.e., is not a
        decay-type megacomplex).
    """
    if isinstance(megacomplexes, str):
        megacomplexes = [megacomplexes]

    omit = omit_parameters or set()
    transitions: list[Transition] = []
    gs_counter = 0

    for mc_label in megacomplexes:
        if mc_label not in model.megacomplex:
            msg = f"Megacomplex '{mc_label}' not found in model."
            raise ValueError(msg)

        filled_mc = fill_item(model.megacomplex[mc_label], model, parameters)

        if not hasattr(filled_mc, "get_k_matrix"):
            msg = (
                f"Megacomplex '{mc_label}' (type={model.megacomplex[mc_label].type}) "
                f"does not support k-matrix extraction. "
                f"Only decay-type megacomplexes are supported."
            )
            raise TypeError(msg)

        k_matrix = filled_mc.get_k_matrix()

        for (to_comp, from_comp), param in k_matrix.matrix.items():
            if param.label in omit:
                continue

            if from_comp == to_comp:
                # Diagonal entry: ground state decay
                gs_counter += 1
                gs_label = f"{GROUND_STATE_PREFIX}{gs_counter}"
                transitions.append(
                    Transition(
                        source=from_comp,
                        target=gs_label,
                        rate_constant=param.value,
                        parameter_label=param.label,
                        is_ground_state_decay=True,
                        megacomplex_label=mc_label,
                    )
                )
            else:
                # Off-diagonal entry: transfer between compartments
                transitions.append(
                    Transition(
                        source=from_comp,
                        target=to_comp,
                        rate_constant=param.value,
                        parameter_label=param.label,
                        is_ground_state_decay=False,
                        megacomplex_label=mc_label,
                    )
                )

    return transitions


def extract_dataset_transitions(
    dataset_name: str,
    model: Model,
    parameters: Parameters,
    *,
    exclude_megacomplexes: set[str] | None = None,
    omit_parameters: set[str] | None = None,
) -> list[Transition]:
    """Extract transitions from all decay megacomplexes of a dataset.

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
    omit_parameters : set[str] | None
        Parameter labels to exclude from the result. Defaults to None.

    Returns
    -------
    list[Transition]
        All rate constant transitions found in the dataset's decay megacomplexes.

    Raises
    ------
    ValueError
        If the dataset name is not found in the model.
    """
    if dataset_name not in model.dataset:
        msg = f"Dataset '{dataset_name}' not found in model."
        raise ValueError(msg)

    associated_megacomplexes = model.dataset[dataset_name].megacomplex
    exclude = exclude_megacomplexes or set()
    megacomplexes = [mc for mc in associated_megacomplexes if mc not in exclude]

    # Filter to only decay-type megacomplexes (silently skip non-decay)
    decay_megacomplexes = [
        mc
        for mc in megacomplexes
        if hasattr(fill_item(model.megacomplex[mc], model, parameters), "get_k_matrix")
    ]

    return extract_transitions(
        decay_megacomplexes,
        model,
        parameters,
        omit_parameters=omit_parameters,
    )
