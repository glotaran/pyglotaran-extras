import matplotlib.pyplot as plt


def plot_residual(res, ax, linlog=False, linthresh=1):
    if min(res.data.shape) < 5:
        res.data.plot(x="time", ax=ax)
    else:
        res.data.plot(x="time", ax=ax, add_colorbar=False)
    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh)
    plt.draw()
    plt.pause(0.001)
