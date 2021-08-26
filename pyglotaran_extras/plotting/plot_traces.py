import matplotlib.pyplot as plt

from pyglotaran_extras.plotting.style import PlotStyle


def get_shifted_traces(res, center_λ=None):
    if "species_concentration" in res:
        traces = res.species_concentration
    elif "species_associated_concentrations" in res:
        traces = res.species_associated_concentrations
    else:
        raise ValueError(f"No concentrations in result:\n{res}")
    times = traces.coords["time"]
    if center_λ is None:  # center wavelength (λ in nm)
        center_λ = min(res.dims["spectral"], round(res.dims["spectral"] / 2))
    # preparation for https://github.com/glotaran/pyglotaran/pull/786
    # might need to be adjusted when "center_dispersion" gets renamed
    if "center_dispersion" in res:
        center_dispersion = res.center_dispersion
        irf_loc = center_dispersion.sel(spectral=center_λ, method="nearest").item()
    elif "center_dispersion_1" in res:
        # legacy compatibility pyglotaran<0.5.0
        center_dispersion = res.center_dispersion_1
        irf_loc = center_dispersion.sel(spectral=center_λ, method="nearest").item()
    elif "irf_center" in res:
        irf_loc = res.irf_center
    else:
        irf_loc = min(times)

    if hasattr(irf_loc, "shape"):
        irf_loc = irf_loc.mean()

    times_shifted = times - irf_loc
    return traces.assign_coords(time=times_shifted)


def calculate_x_ranges(res, linrange):
    print(f"{res=}")
    print(f"{linrange=}")
    print("Not yet implemented")
    pass


def plot_traces(res, ax, center_λ, linlog=False, linthresh=1, linscale=1):
    traces = get_shifted_traces(res, center_λ)
    plot_style = PlotStyle()
    plt.rc("axes", prop_cycle=plot_style.cycler)

    if "spectral" in traces.coords:
        traces.sel(spectral=center_λ, method="nearest").plot.line(x="time", ax=ax)
    else:
        traces.plot.line(x="time", ax=ax)

    if linlog:
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
