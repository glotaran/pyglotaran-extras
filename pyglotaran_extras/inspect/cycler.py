"""Module containing functionality to inspect plot cyclers.

This was inspired by:
https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
from cycler import Cycler
from cycler import cycler as cycler_func
from glotaran.utils.ipython import MarkdownStr
from tabulate import tabulate


def create_preview_cycler(cycler: Cycler) -> Cycler:
    """Create a cycler with preview images of plotstyles, as HTML ``img`` tags.

    The actual image in the ``img`` tag is stored as base64 encoded string.

    Parameters
    ----------
    cycler : Cycler
        Cycler to generate the preview for.

    Returns
    -------
    Cycler
        Preview images as HTML ``img`` with image data stored as base64 string.

    See Also
    --------
    inspect_cycler
    """
    preview_images = []
    buffer = io.BytesIO()
    x, y = list(range(10)), [0] * 10
    fig, ax = plt.subplots(1, 1, figsize=(0.5, 0.15))
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)
    for values in cycler:
        ax.plot(x, y, **values)
        ax.get_yaxis().set_ticks([])
        buffer.seek(0)
        fig.savefig(buffer, format="jpg")
        buffer.seek(0)
        preview_images.append(
            f'<img src="data:image/jpg;base64, {base64.b64encode(buffer.read()).decode()}">'
        )
        ax.cla()
    plt.close()
    return cycler_func(preview=preview_images)


def inspect_cycler(cycler: Cycler) -> MarkdownStr:
    """Inspect a plotstyle ``cycler`` as a markdown table including a preview.

    Parameters
    ----------
    cycler : Cycler
        Plotstyle cycler to inspect.

    Returns
    -------
    MarkdownStr
    """
    return MarkdownStr(
        tabulate(
            list(cycler + create_preview_cycler(cycler)),
            tablefmt="unsafehtml",
            showindex="always",
            headers="keys",
        )
    )
