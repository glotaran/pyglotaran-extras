import matplotlib.pyplot as plt
import xarray as xr

from pyglotaran_extras.plotting.plot_svd import plot_lsv_data
from pyglotaran_extras.plotting.plot_svd import plot_rsv_data
from pyglotaran_extras.plotting.plot_svd import plot_sv_data


def plot_data_overview(
    dataset: xr.Dataset, title="Data overview", linlog: bool = False, linthresh: float = 1
):
    fig = plt.figure()
    data_ax = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3, fig=fig)
    lsv_ax = plt.subplot2grid((4, 3), (3, 0), fig=fig)
    sv_ax = plt.subplot2grid((4, 3), (3, 1), fig=fig)
    rsv_ax = plt.subplot2grid((4, 3), (3, 2), fig=fig)

    if len(dataset.data.time) > 1:
        dataset.data.plot(x="time", ax=data_ax, center=False)
    else:
        dataset.data.plot(ax=data_ax)
    plot_lsv_data(dataset, lsv_ax)
    plot_sv_data(dataset, sv_ax)
    plot_rsv_data(dataset, rsv_ax)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    if linlog:
        data_ax.set_xscale("symlog", linthresh=linthresh)
    return fig, (data_ax, lsv_ax, sv_ax, rsv_ax)
