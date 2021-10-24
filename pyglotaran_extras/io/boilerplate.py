"""Deprecated module."""
from pyglotaran_extras.deprecation import warn_deprecated
from pyglotaran_extras.io.setup_case_study import setup_case_study

__all__ = ["setup_case_study"]

warn_deprecated(
    deprecated_qual_name_usage="pyglotaran_extras.io.boilerplate",
    new_qual_name_usage="pyglotaran_extras.io",
    to_be_removed_in_version="0.7.0",
    stacklevel=3,
)
