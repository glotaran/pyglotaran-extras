from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from glotaran.model.experiment_model import ExperimentModel

from glotaran.project.result import Result
from glotaran.utils.ipython import MarkdownStr
from tabulate import tabulate


class CompatResult(Result):
    """A v0.7 compatibility wrapper for a v0.8 Result.

    Inherits:
    optimization_results: dict[str, OptimizationResult]
    scheme: Scheme
    optimization_info: OptimizationInfo
    initial_parameters: Parameters
    optimized_parameters: Parameters
    """

    # Storage for converted flat datasets (v0.7 style)
    _compat_datasets: dict[str, xr.Dataset] | None = None

    @property
    def number_of_function_evaluations(self) -> int:
        return self.optimization_info.number_of_function_evaluations

    @property
    def success(self) -> bool:
        return self.optimization_info.success

    @property
    def termination_reason(self) -> str:
        return self.optimization_info.termination_reason

    @property
    def glotaran_version(self) -> str:
        return self.optimization_info.glotaran_version

    @property
    def number_of_residuals(self) -> int:
        return self.optimization_info.number_of_data_points

    @property
    def number_of_free_parameters(self) -> int:
        return self.optimization_info.number_of_parameters

    @property
    def number_of_clps(self) -> int:
        return self.optimization_info.number_of_clps

    @property
    def degrees_of_freedom(self) -> int:
        return self.optimization_info.degrees_of_freedom

    @property
    def chi_square(self) -> float:
        return self.optimization_info.chi_square

    @property
    def reduced_chi_square(self) -> float:
        return self.optimization_info.reduced_chi_square

    @property
    def reduced_chi_squared(self) -> float:
        return self.optimization_info.reduced_chi_square

    @property
    def root_mean_square_error(self) -> float:
        return self.optimization_info.root_mean_square_error

    @property
    def additional_penalty(self) -> float | None:
        return self.optimization_info.additional_penalty

    @property
    def data(self) -> dict[str, xr.Dataset]:
        """Return v0.7-style flat datasets."""
        if self._compat_datasets is not None:
            return self._compat_datasets
        # Fallback: return empty dict if not yet converted
        return {}

    @property
    def model(self) -> ExperimentModel:
        return self.experiments[next(iter(self.experiments.keys()))]

    @classmethod
    def from_result(cls, result: Result) -> CompatResult:
        """Create a CompatResult from a Result object."""
        return cls(
            saving_options=result.saving_options,
            optimization_results=result.optimization_results,
            scheme=result.scheme,
            optimization_info=result.optimization_info,
            initial_parameters=result.initial_parameters,
            optimized_parameters=result.optimized_parameters,
        )

    def markdown(
        self,
        with_model: bool = True,
        *,
        base_heading_level: int = 1,
        wrap_model_in_details: bool = False,
    ) -> MarkdownStr:
        """Format the model as a markdown text.

        Parameters
        ----------
        with_model : bool
            If `True`, the model will be printed with initial and optimized parameters filled in.
        base_heading_level : int
            The level of the base heading.
        wrap_model_in_details: bool
            Wraps model into details tag. Defaults to ``False``

        Returns
        -------
        MarkdownStr : str
            The scheme as markdown string.
        """
        general_table_rows: list[list[Any]] = [
            ["Number of residual evaluation", self.number_of_function_evaluations],
            ["Number of residuals", self.number_of_residuals],
            ["Number of free parameters", self.number_of_free_parameters],
            ["Number of conditionally linear parameters", self.number_of_clps],
            ["Degrees of freedom", self.degrees_of_freedom],
            ["Chi Square", f"{self.chi_square or np.nan:.2e}"],
            ["Reduced Chi Square", f"{self.reduced_chi_square or np.nan:.2e}"],
            ["Root Mean Square Error (RMSE)", f"{self.root_mean_square_error or np.nan:.2e}"],
        ]
        if self.additional_penalty is not None:
            general_table_rows.append(["RMSE additional penalty", self.additional_penalty])

        result_table = tabulate(
            general_table_rows,
            headers=["Optimization Result", ""],
            tablefmt="github",
            disable_numparse=True,
        )
        if len(self.data) > 1:
            RMSE_rows = [
                [
                    f"{index}.{label}:",
                    dataset.attrs["weighted_root_mean_square_error"],
                    dataset.attrs["root_mean_square_error"],
                ]
                for index, (label, dataset) in enumerate(self.data.items(), start=1)
            ]

            RMSE_table = tabulate(
                RMSE_rows,
                headers=["RMSE (per dataset)", "weighted", "unweighted"],
                floatfmt=".2e",
                tablefmt="github",
            )

            result_table = f"{result_table}\n\n{RMSE_table}"

        if with_model:
            result_table += (
                "\n\n> **Warning:** Printing model is not yet implemented for `CompatResult`."
            )
            result_table += f"\n\n> **unused:** {base_heading_level=}, {wrap_model_in_details=}."

            # model_md = self.model.markdown(
            #     parameters=self.optimized_parameters,
            #     initial_parameters=self.initial_parameters,
            #     base_heading_level=base_heading_level,
            # )
            # if wrap_model_in_details is False:
            #     result_table = f"{result_table}\n\n{model_md}"
            # else:
            #     # The section part is just a hack to generate properly rendering docs due to a bug
            #     # in sphinx which causes a wrong tag opening and closing order of html tags
            #     # Since model_md contains 2 heading levels we need to close 2 sections
            #     result_table = (
            #         f"{result_table}\n\n<br><details>\n\n{model_md}\n"
            #         f"{'</section>'*(2)}"
            #         "</details>"
            #         f"{'<section>'*(2)}"
            #     )

        return MarkdownStr(result_table)

    def _repr_markdown_(self) -> str:
        """Return a markdown representation str.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str
            The scheme as markdown string.
        """
        return str(self.markdown(base_heading_level=3, wrap_model_in_details=True))

    def __str__(self) -> str:
        """Overwrite of ``__str__``."""
        return str(self.markdown(with_model=False))
