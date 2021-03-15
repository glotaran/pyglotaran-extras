import matplotlib.pyplot as plt


def plot_svd(res, axes, linlog=False, linthresh=1):
    plot_lsv_residual(res, axes[0, 0], linlog=linlog, linthresh=linthresh)
    plot_rsv_residual(res, axes[0, 1])
    plot_sv_residual(res, axes[0, 2])
    plot_lsv_data(res, axes[1, 0], linlog=linlog, linthresh=linthresh)
    plot_rsv_data(res, axes[1, 1])
    plot_sv_data(res, axes[1, 2])
    plt.draw()
    plt.pause(0.001)


def plot_lsv_data(res, ax, indices=range(4), linlog=False, linthresh=1):
    """ Plot left singular vectors (time) of the data matrix """
    dLSV = res.data_left_singular_vectors
    dLSV.isel(
        left_singular_value_index=indices[: len(dLSV.left_singular_value_index)]
    ).plot.line(x="time", ax=ax)
    ax.set_title("data. LSV")
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)


def plot_rsv_data(res, ax, indices=range(4)):
    """ Plot right singular vectors (spectra) of the data matrix """
    dRSV = res.data_right_singular_vectors
    dRSV.isel(
        right_singular_value_index=indices[: len(dRSV.right_singular_value_index)]
    ).plot.line(x="spectral", ax=ax)
    ax.set_title("data. RSV")


def plot_sv_data(res, ax, indices=range(10)):
    """ Plot singular values of the data matrix """
    dSV = res.data_singular_values
    dSV.sel(singular_value_index=indices[: len(dSV.singular_value_index)]).plot.line(
        "ro-", yscale="log", ax=ax
    )
    ax.set_title("data. log(SV)")


def plot_lsv_residual(
    res, ax, indices=range(2), label="residual", linlog=False, linthresh=1
):
    """ Plot left singular vectors (time) of the residual matrix """
    if "weighted_residual_left_singular_vectors" in res:
        rLSV = res.weighted_residual_left_singular_vectors
    else:
        rLSV = res.residual_left_singular_vectors
    rLSV.isel(
        left_singular_value_index=indices[: len(rLSV.left_singular_value_index)]
    ).plot.line(x="time", ax=ax)
    ax.set_title("res. LSV")
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)


def plot_rsv_residual(res, ax, indices=range(2)):
    """ Plot right singular vectors (spectra) of the residual matrix """
    if "weighted_residual_right_singular_vectors" in res:
        rRSV = res.weighted_residual_right_singular_vectors
    else:
        rRSV = res.residual_right_singular_vectors
    rRSV.isel(
        right_singular_value_index=indices[: len(rRSV.right_singular_value_index)]
    ).plot.line(x="spectral", ax=ax)
    ax.set_title("res. RSV")


def plot_sv_residual(res, ax, indices=range(10)):
    """ Plot singular values of the residual matrix """
    if "weighted_residual_singular_values" in res:
        rSV = res.weighted_residual_singular_values
    else:
        rSV = res.residual_singular_values
    rSV.sel(singular_value_index=indices[: len(rSV.singular_value_index)]).plot.line(
        "ro-", yscale="log", ax=ax
    )
    ax.set_title("res. log(SV)")
