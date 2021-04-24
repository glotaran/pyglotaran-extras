import matplotlib.pyplot as plt
import numpy as np


def plot_residual(res, ax, linlog=False, linthresh=1, show_data=False):
    data = res.data if show_data else res.residual
    title = "dataset" if show_data else "residual"
    shape = np.array(data.shape)
    dims = data.coords.dims
    if min(shape) == 1:
        # res.data.plot.line(x=dims[shape.argmax()], ax=ax)
        # res.fitted_data.plot.line(x=dims[shape.argmax()], ax=ax)
        data.plot.line(x=dims[shape.argmax()], ax=ax)
    elif min(shape) < 5:
        data.plot(x="time", ax=ax)
    else:
        data.plot(x="time", ax=ax, add_colorbar=False)
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_title(title)
