"""Module containing functionality to initialize a case study."""
from __future__ import annotations

import inspect
from os import PathLike
from pathlib import Path


def setup_case_study(
    output_folder_name: str = "pyglotaran_results",
    results_folder_root: None | str | PathLike[str] = None,
) -> tuple[Path, Path]:
    """Quickly get folders for a case study.

    This is an execution environment independent (works in python script files
    and notebooks, independent of where the python runtime was called from)
    way to get the folder the analysis code resides in and also creates the
    ``results_folder`` in case it didn't exist before.

    Parameters
    ----------
    output_folder_name : str
        Name of the base folder for the results. Defaults to "pyglotaran_results".
    results_folder_root : None
        The folder where the results named ``output_folder_name`` should be saved to.
        Defaults to None, which results in the users Home folder being used.

    Returns
    -------
    tuple[Path, Path]
        results_folder, script_folder:

        results_folder:
            Folder to be used to save results in of the pattern
            (``results_folder_root`` / ``output_folder_name`` / ``analysis_folder.parent``).
        analysis_folder:
            Folder the script or Notebook resides in.
    """
    analysis_folder = get_script_dir(nesting=1)
    print(f"Setting up case study for folder: {analysis_folder}")
    if results_folder_root is None:
        results_folder_root = Path.home() / output_folder_name
    else:
        results_folder_root = Path(str(results_folder_root)) / output_folder_name
    script_folder_rel = analysis_folder.relative_to(analysis_folder.parent)
    results_folder = (results_folder_root / script_folder_rel).resolve()
    results_folder.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")
    return results_folder, analysis_folder.resolve()


def get_script_dir(*, nesting: int = 0) -> Path:
    """Get parent folder a script/Notebook is executed in.

    This is a helper function for cross compatibility with jupyter notebooks.
    In notebooks the global ``__file__`` variable isn't set, thus we need different
    means to get the folder a script is defined in, which doesn't change with the
    current working director the ``python interpreter`` was called from.

    Parameters
    ----------
    nesting : int
        Number to go up in the call stack to get to the initially calling function.
        This is only needed for library code and not for user code. Defaults to 0 (direct call).

    Returns
    -------
    Path
        Path to the folder the script is located in.

    See Also
    --------
    setup_case_study
    """
    calling_frame = inspect.stack()[nesting + 1].frame
    file_var = calling_frame.f_globals.get("__file__", ".")
    file_path = Path(file_var).resolve()
    return file_path if file_var == "." else file_path.parent
