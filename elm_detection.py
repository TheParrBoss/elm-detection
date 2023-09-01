import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import xarray as xr
import cdb_extras.xarray_support as cdbxr
from pyCDB import client
from scipy.io import loadmat
from scipy.signal import find_peaks

cdb = client.CDBClient()


def calculate_gradient(x1, y1, x2, y2):
    gradient = (y2 - y1) / (x2 - x1)
    return gradient


def load_cdb_energy_signal(shot_number, variant="HIRES_ELM"):
    """Loads the energy signal from the CDB."""
    shot_accessor = cdbxr.Shot(shot_number)
    signal_name = f"W/EFIT:{shot_number}:{variant}"
    w_signal = shot_accessor[signal_name]

    # start_slice = None
    # end_slice = None
    # start_found = False

    # for i in range(len(w_signal.time.data) - 20):
    #     time_difference_forward = (w_signal.time.data[i + 20] - w_signal.time.data[i]).astype('timedelta64[ms]')
        
    #     if time_difference_forward <= np.timedelta64(1, 'ms') and not start_found:
    #         start_slice = w_signal.time.data[i]
    #         start_found = True
    #         continue  # Skip to the next iteration

    #     if start_found:
    #         time_difference_forward = (w_signal.time.data[i + 20] - w_signal.time.data[i]).astype('timedelta64[ms]')
            
    #         # Check if the time difference goes back to regular 1 ms sampling
    #         if time_difference_forward > np.timedelta64(1, 'ms'):
    #             end_slice = w_signal.time.data[i]
    #             break

    # w_signal = w_signal.sel(time=slice(start_slice,end_slice))

    return w_signal

# def slice_list_by_values(input_list, start_value, end_value):
#     start_index = None
#     end_index = None

#     for i, value in enumerate(input_list):
#         if value >= start_value and start_index is None:
#             start_index = i
#         if value > end_value:
#             end_index = i
#             break

#     if start_index is None or end_index is None or start_index >= end_index:
#         return []

#     return input_list[start_index:end_index]

def load_cdb_telm_precalc(shot_number):
    """Loads preprocessed ELM times from the CDB."""
    shot_accessor = cdbxr.Shot(shot_number)
    signal_name = f"t_ELM_start/SYNTHETIC_DIAGNOSTICS:{shot_number}"
    telm_signal = shot_accessor[signal_name]

    # telm_signal = telm_signal.data.tolist()
    # telm_signal = slice_list_by_values(telm_signal, start_slice,end_slice)

    return telm_signal


def load_michals_matfile_as_da(path):
    mat = loadmat(path, squeeze_me=True)

    time = mat["o"]["Wmhd_t"].item() * 1e3  # as ms
    signal = mat["o"]["Wmhd"].item()

    signal_da = xr.DataArray(signal, dims=["time"], coords={"time": time}, name="W")

    if "t_elm_start" in mat["o"].dtype.names:
        t_elm_start = mat["o"]["t_elm_start"].item() * 1e3  # as ms
        telm_da = xr.DataArray(
            t_elm_start,
            dims=["elm_idx"],
            name="t_ELM_start",
        )
    else:
        telm_da = None
    return signal_da, telm_da


def find_peaks_in_signal(signal, width=15, distance=5, rel_height=1):
    """Finds peaks in a signal."""
    peaks_up, _ = find_peaks(
        signal, width=width, distance=distance, rel_height=rel_height
    )
    peaks_down, _ = find_peaks(
        -signal, width=width, distance=distance, rel_height=rel_height
    )

    # Select only peak that fallows each other and save it as list of tuples
    i, j = 0, 0
    peaks = []

    for i, peak_up in enumerate(peaks_up):
        for j, peak_down in enumerate(peaks_down):
            if peak_up < peak_down and (
                    i == len(peaks_up) - 1 or peaks_up[i + 1] > peak_down
            ):
                peaks.append((peak_up, peak_down))
                break

    return peaks, peaks_up, peaks_down


def peaks_as_xarray(peaks, signal):
    """Converts peaks to xarray."""
    t_start = np.asarray([peak[0] for peak in peaks])
    t_end = np.asarray([peak[1] for peak in peaks])

    times = signal.coords["time"].values
    values = signal.values

    ds = xr.Dataset()
    ds["t_start"] = xr.DataArray(times[t_start], dims=["peak_idx"])
    ds["t_end"] = xr.DataArray(times[t_end], dims=["peak_idx"])
    ds["W_start"] = xr.DataArray(values[t_start], dims=["peak_idx"])
    ds["W_end"] = xr.DataArray(values[t_end], dims=["peak_idx"])

    return ds


def filter_peaks(peaks_ds, gradient_threshold=-100, energy_drop_threshold=0.01):
    """Filters peaks based on gradient."""

    ds = peaks_ds.copy()

    gradient = calculate_gradient(
        ds["t_start"], ds["W_start"], ds["t_end"], ds["W_end"]
    )
    ds["gradient"] = gradient

    energy_drop = 2 * (ds["W_start"] - ds["W_end"]) / (ds["t_start"] + ds["t_end"])
    ds["relative_energy_drop"] = np.abs(energy_drop)

    # Filter dataset
    ds = ds.where(ds["relative_energy_drop"] > energy_drop_threshold, drop=True)
    ds = ds.where(ds["gradient"] < gradient_threshold, drop=True)

    return ds


def detect_events(
    test_series: xr.DataArray, control_series: xr.DataArray, delta_t: float = 0.5
) -> dict:
    """Detects events in test_series that are also present in control_series."""
    results = []

    test_ptr, control_ptr = 0, 0

    while test_ptr < len(test_series) and control_ptr < len(control_series):
        if abs(test_series[test_ptr] - control_series[control_ptr]) <= delta_t:
            results.append(
                {
                    "index": test_series[test_ptr].item(),
                    "event_time": test_series[test_ptr].values.item(),
                    "flag": "True Positive",
                }
            )
            test_ptr += 1
            control_ptr += 1
        elif test_series[test_ptr] < control_series[control_ptr] - delta_t:
            results.append(
                {
                    "index": test_series[test_ptr].item(),
                    "event_time": test_series[test_ptr].values.item(),
                    "flag": "False Positive",
                }
            )
            test_ptr += 1
        else:
            results.append(
                {
                    "index": control_series[control_ptr].item(),
                    "event_time": control_series[control_ptr].values.item(),
                    "flag": "False Negative",
                }
            )
            control_ptr += 1

    while test_ptr < len(test_series):
        results.append(
            {
                "index": test_series[test_ptr].item(),
                "event_time": test_series[test_ptr].values.item(),
                "flag": "False Positive",
            }
        )
        test_ptr += 1

    while control_ptr < len(control_series):
        results.append(
            {
                "index": control_series[control_ptr].item(),
                "event_time": control_series[control_ptr].values.item(),
                "flag": "False Negative",
            }
        )
        control_ptr += 1

    return results


def plot_peaks(peaks_ds, ax=None):
    """Plots peaks on top of the signal."""
    if ax is None:
        ax: plt.Axes
        _, ax = plt.subplots()

    for t_start, t_end in zip(peaks_ds["t_start"], peaks_ds["t_end"]):
        ax.axvspan(t_start, t_end, color="C2", alpha=0.5)


def plot_telms(t_elms, ax=None):
    """Plots ELM times on top of the signal."""
    if ax is None:
        ax: plt.Axes
        _, ax = plt.subplots()

    for t_elm in t_elms:
        ax.axvline(t_elm, color="C2", alpha=0.5)


def plot_all(signal, peaks_ds, filtered_peak_ds, telms, xlim=(None, None)):
    fig, axs = plt.subplots(2, 1, sharex=True)

    signal.plot(ax=axs[0], color="k")
    signal.plot(ax=axs[1], color="k")

    axs[0].plot(peaks_ds["t_start"], peaks_ds["W_start"], "x", color="C1")
    axs[0].plot(peaks_ds["t_end"], peaks_ds["W_end"], "x", color="C2")

    plot_peaks(filtered_peak_ds, axs[0])
    if telms is not None:
        plot_telms(telms, axs[1])

    plt.xlim(xlim)
    return fig, axs


# Altair:


def plot_peaks_vega(peaks_df):
    """Returns a chart with peaks layered on top of the signal."""
    peak_chart = (
        alt.Chart(peaks_df)
        .mark_rule(opacity=0.3, strokeWidth=2)
        .encode(
            x="t_start:Q",
            color=alt.value("green"),
        )
    )
    return peak_chart


def plot_telms_vega(t_elms_df):
    """Returns a chart with ELM times layered on top of the signal."""
    telms_chart = (
        alt.Chart(t_elms_df)
        .mark_rule(opacity=0.3, strokeWidth=2)
        .encode(x="t_ELM_start:Q", color=alt.value("red"))
    )
    return telms_chart


def plot_all_vega(signal_xr, peaks_xr, filtered_peak_xr, telms_xr):
    signal_df = signal_xr.to_dataframe().reset_index()
    filtered_peak_df = filtered_peak_xr.to_dataframe()
    telms_df = telms_xr.to_dataframe()

    # Base signal plot
    signal_chart = (
        alt.Chart(signal_df).mark_line(color="orange").encode(x="time:Q", y="W:Q")
    )

    # Peaks and ELMs
    # peaks_chart = plot_peaks_vega(peaks_df)
    filtered_peak_chart = plot_peaks_vega(filtered_peak_df)
    telms_chart = plot_telms_vega(telms_df)

    # Layer everything together
    final_chart = alt.layer(signal_chart, telms_chart, filtered_peak_chart)

    # return alt.layer(signal_chart)
    return final_chart
