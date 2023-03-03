"""Module containing deprecation functionality."""
from __future__ import annotations

from importlib.metadata import distribution
from warnings import warn

FIG_ONLY_WARNING = (
    "The ``figure_only`` argument was deprecated please remove it from the function call.\n"
    "This usage will be an error in version: 0.8.0."
)


class OverDueDeprecation(Exception):
    """Error thrown when a deprecation should have been removed.

    See Also
    --------
    deprecate
    warn_deprecated
    deprecate_module_attribute
    deprecate_submodule
    deprecate_dict_entry
    """


class PyglotaranExtrasApiDeprecationWarning(UserWarning):
    """Warning to give users about API changes.

    See Also
    --------
    warn_deprecated
    """


def pyglotaran_extras_version() -> str:
    """Version of the distribution.

    This is basically the same as ``pyglotaran_extras.__version__`` but independent
    from pyglotaran_extras.
    This way all of the deprecation functionality can be used even in
    ``pyglotaran_extras.__init__.py`` without moving the import below the definition of
    ``__version__`` or causing a circular import issue.

    Returns
    -------
    str
        The version string.
    """
    return distribution("pyglotaran-extras").version


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse version string to tuple of three ints for comparison.

    Parameters
    ----------
    version_str : str
        Fully qualified version string of the form 'major.minor.patch'.

    Returns
    -------
    tuple[int, int, int]
        Version as tuple.

    Raises
    ------
    ValueError
        If ``version_str`` has less that three elements separated by ``.``.
    ValueError
        If ``version_str`` 's first three elements can not be casted to int.
    """
    error_message = (
        "version_str needs to be a fully qualified version consisting of "
        f"int parts (e.g. '0.0.1'), got {version_str!r}"
    )
    split_version = version_str.partition("-")[0].split(".")
    if len(split_version) < 3:
        raise ValueError(error_message)
    try:
        return tuple(
            map(int, (*split_version[:2], split_version[2].partition("rc")[0]))
        )  # type:ignore[return-value]
    except ValueError as error:
        raise ValueError(error_message) from error


def check_overdue(deprecated_qual_name_usage: str, to_be_removed_in_version: str) -> None:
    """Check if a deprecation is overdue for removal.

    Parameters
    ----------
    deprecated_qual_name_usage : str
        Old usage with fully qualified name e.g.:
        ``'glotaran.read_model_from_yaml(model_yml_str)'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.
    """
    if (
        parse_version(pyglotaran_extras_version()) >= parse_version(to_be_removed_in_version)
        and "dev" not in pyglotaran_extras_version()
    ):
        raise OverDueDeprecation(
            f"Support for {deprecated_qual_name_usage.partition('(')[0]!r} was "
            f"supposed to be dropped in version: {to_be_removed_in_version!r}.\n"
            f"Current version is: {pyglotaran_extras_version()!r}"
        )


def warn_deprecated(
    *,
    deprecated_qual_name_usage: str,
    new_qual_name_usage: str,
    to_be_removed_in_version: str,
    stacklevel: int = 2,
) -> None:
    """Raise deprecation warning with change information.

    The change information are old / new usage information and end of support version.

    Parameters
    ----------
    deprecated_qual_name_usage : str
        Old usage with fully qualified name e.g.:
        ``'glotaran.read_model_from_yaml(model_yml_str)'``
    new_qual_name_usage : str
        New usage as fully qualified name e.g.:
        ``'glotaran.io.load_model(model_yml_str, format_name="yml_str")'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.

    stacklevel: int
        Stack at which the warning should be shown as raise. Default: 2

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.


    -- noqa: DAR402
    """
    check_overdue(deprecated_qual_name_usage, to_be_removed_in_version)
    warn(
        PyglotaranExtrasApiDeprecationWarning(
            f"Usage of {deprecated_qual_name_usage!r} was deprecated, "
            f"use {new_qual_name_usage!r} instead.\n"
            f"This usage will be an error in version: {to_be_removed_in_version!r}."
        ),
        stacklevel=stacklevel,
    )
