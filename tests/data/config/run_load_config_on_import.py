import json
from pathlib import Path

# This is needed to simulate the case were something else is imported from ``pyglotaran_extras``
# Before the config is
from pyglotaran_extras import plot_overview  # noqa: F401

# Added before the import so imports don't get sorted
HERE = Path(__file__).parent

# We just import the config directly to read out the values
from pyglotaran_extras import CONFIG  # noqa: E402

(HERE / "source_files.json").write_text(
    json.dumps([source_file.as_posix() for source_file in CONFIG._source_files])
)
(HERE / "plotting.json").write_text(CONFIG.plotting.model_dump_json())
