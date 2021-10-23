from __future__ import annotations

from enum import Enum

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cycler import cycler


class ColorCode(Enum):
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
        rgb = colors.hex2color(hex_string)
        return tuple(int(255 * x) for x in rgb)

    @staticmethod
    def rgb_to_hex(rgb_tuple: tuple[float, ...]) -> str:
        return colors.rgb2hex([1.0 * x / 255 for x in rgb_tuple])


class LineStyle(Enum):
    solid = "-"
    dashed = "--"
    dotted = ".."
    dashdot = "-."


class DataColorCode(Enum):
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
    # orange = "#ff8c00"


class DataLineStyle(Enum):
    solid = "-"
    dashed = "--"


class PlotStyle:
    def __init__(self):
        self._line_style = [e.value for e in LineStyle]
        self._color_codes = [e.value for e in ColorCode]
        # Since Enum only supports unique values we need to manually add the last one
        self._data_color_codes = [e.value for e in DataColorCode] + [DataColorCode.orange.value]
        # Extend linecycler to same size as colorcycler
        self._data_line_style = [e.value for e in DataLineStyle] * (
            len(self._data_color_codes) // 2
        )
        self.SMALL_SIZE = 12
        self.MEDIUM_SIZE = 14
        self.BIGGER_SIZE = 18

    def set_default_fontsize(self):
        plt.rc("font", size=self.SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=self.SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=self.MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=self.SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=self.SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=self.SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=self.BIGGER_SIZE)  # fontsize of the figure title

    def set_default_colors(self):
        plt.rc("axes", prop_cycle=self.cycler)

    @property
    def cycler(self):
        return cycler("color", self._color_codes)

    @property
    def data_cycler_solid_dashed(self):
        """A style that alternates between solid and dashes and uses :class:`DataColorCode`.

        This is useful when plotting simulation or very low noise data, where the
        fit practically falls on top and hides the data lines.
        """
        return cycler(color=self._data_color_codes, linestyle=self._data_line_style)

    @property
    def data_cycler_solid(self):
        """Color cycler using :class:`DataColorCode`."""
        return cycler(color=self._data_color_codes)

    @property
    def line_cycler(self):
        return cycler("linestyle", self._line_style)
