"""Module containing predefined plot styles.

For reference see: https://glotaran.github.io/legacy/plot_styles
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cycler import cycler

if TYPE_CHECKING:
    from cycler import Cycler


class ColorCode(str, Enum):
    """Color definitions from legacy glotaran.

    See: https://glotaran.github.io/legacy/plot_styles
    """

    # Name	#Hex
    black = "#000000"
    red = "#ff0000"
    blue = "#0000ff"
    green = "#00ff00"
    magenta = "#ff00ff"
    cyan = "#00ffff"
    yellow = "#ffff00"
    green4 = "#008b00"
    orange = "#ff8c00"
    brown = "#964b00"
    grey = "#808080"
    violet = "#9400d3"
    turquoise = "#40e0d0"
    maroon = "#800000"
    indigo = "#4b0082"

    @staticmethod
    def hex_to_rgb(hex_string: str) -> tuple[int, ...]:
        """Convert hex code to rgb or rgba tuple.

        Parameters
        ----------
        hex_string : str
            Hey code representation of a color.

        Returns
        -------
        tuple[int, ...]
            rgb or rgba tuple representing the same color as ``hex_string``..
        """
        rgb = colors.hex2color(hex_string)
        return tuple(int(255 * x) for x in rgb)

    @staticmethod
    def rgb_to_hex(rgb_tuple: tuple[float, ...]) -> str:
        """Convert rgb value tuple to hex code.

        Parameters
        ----------
        rgb_tuple : tuple[float, ...]
            Tuple rgb or rgba values

        Returns
        -------
        str
            Hex code representing the same color as ``rgb_tuple``.
        """
        return colors.rgb2hex([1.0 * x / 255 for x in rgb_tuple])


class LineStyle(str, Enum):
    """Subset of line styles supported by matplotlib."""

    solid = "-"
    dashed = "--"
    dotted = ".."
    dashdot = "-."


class DataColorCode(str, Enum):
    """Colors used to plot data and fits.

    Pairs of visually similar looking colors whereby the
    first (lighter) color is used to plot the data,
    and the second (darker) color is used to represent
    the fitted trace (that goes 'through' the data).
    """

    # Name	#Hex
    grey = "#808080"
    black = "#000000"
    orange = "#ff8c00"
    red = "#ff0000"
    cyan = "#00ffff"
    blue = "#0000ff"
    green = "#00ff00"
    green4 = "#008b00"
    magenta = "#ff00ff"
    indigo = "#4b0082"
    brown = "#964b00"
    maroon = "#800000"
    yellow = "#ffff00"
    # Needs to be added manually to the list in PlotStyle
    # since Enum doesn't allow duplicates
    # orange = "#ff8c00"


class DataLineStyle(str, Enum):
    """Data plots can alternate between solid lines for data and dashed lines for fits.

    This is mostly useful for data with very low noise (e.g. simulated data),
    since data and fir often overlap
    """

    solid = "-"
    dashed = "--"


class PlotStyle:
    """Wrapper class to hold predefined Plot styles."""

    def __init__(self) -> None:
        """Initialize Stiles from Enums."""
        self._line_style = list(LineStyle)
        self._color_codes = list(ColorCode)
        # Since Enum only supports unique values we need to manually add the last one
        self._data_color_codes = list(DataColorCode) + [DataColorCode.orange]
        # Extend linecycler to same size as colorcycler
        self._data_line_style = list(DataLineStyle) * (len(self._data_color_codes) // 2)
        self.SMALL_SIZE = 12
        self.MEDIUM_SIZE = 14
        self.BIGGER_SIZE = 18

    def set_default_fontsize(self) -> None:
        """Set global plot settings for matplotlib."""
        plt.rc("font", size=self.SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=self.SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=self.MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=self.SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=self.SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=self.SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=self.BIGGER_SIZE)  # fontsize of the figure title

    def set_default_colors(self) -> None:
        """Set default color cycler for matplotlib."""
        plt.rc("axes", prop_cycle=self.cycler)

    @property
    def cycler(self) -> Cycler:
        """Cycle for default legacy like plots using color defined in :class:`ColorCode`.

        Returns
        -------
        Cycler
            Plot color cycler for general plots.
        """
        return cycler("color", self._color_codes)

    @property
    def data_cycler_solid_dashed(self) -> Cycler:
        """Cycle that alternates between solid and dashes and uses :class:`DataColorCode`.

        This is useful when plotting simulation or very low noise data, where the
        fit practically falls on top and hides the data lines.

        Returns
        -------
        Cycler
            Plot color cycler for low noise data+fit plots.
        """
        return cycler(color=self._data_color_codes, linestyle=self._data_line_style)

    @property
    def data_cycler_solid(self) -> Cycler:
        """Color cycler using :class:`DataColorCode`.

        Returns
        -------
        Cycler
            Plot color cycler for data+fit plots.
        """
        return cycler(color=self._data_color_codes)

    @property
    def line_cycler(self) -> Cycler:
        """Cycle iterating over line styles defined in :class:`LineStyle`.

        Returns
        -------
        Cycler
            Plot color cycler for multi line style plots.
        """
        return cycler("linestyle", self._line_style)
