from pyglotaran_extras.io.setup_case_study import get_script_dir


def wrapped_get_script_dir():
    """Testfunction for calls to get_script_dir used inside of other functions."""
    return get_script_dir(nesting=1)
