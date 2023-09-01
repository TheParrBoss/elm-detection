import numpy as np
import pandas as pd
import streamlit as st
import elm_detection as elmd


st.title("Elm Detection App")

with st.sidebar:

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Choose source of data")
        cudb_source = st.toggle("CDB", True)

    with col2:
        st.subheader("Choose plotting backend")
        vega_backend = st.toggle("Vega", True)

    with st.form("record_form"):
        if cudb_source:
            record_number = st.number_input(
                "Record number", value=18234, min_value=1000, max_value=21812, step=1
            )
            variant = st.selectbox("Variant", ["HIRES_ELM", "v7_std_hires"])

            w_signal = elmd.load_cdb_energy_signal(record_number, variant=variant)
            telm_signal = elmd.load_cdb_telm_precalc(record_number)
            st.form_submit_button("Load data")
        else:
            mat_file = st.file_uploader("Upload mat file", type=["mat"])
            st.form_submit_button("Load data")
            if mat_file is None:
                st.stop()
            w_signal, telm_signal = elmd.load_michals_matfile_as_da(mat_file)

    t_lims = (w_signal["time"].min().item(), w_signal["time"].max().item())
    # t_lims = (w_signal["time"].item(), np.max(w_signal["time"]).item())

    st.header("Parameters of find_peaks function")
    width = st.number_input("Width", value=15, min_value=1, max_value=100, step=1)
    distance = st.number_input("Distance", value=5, min_value=1, max_value=100, step=1)
    rel_height = st.number_input(
        "Relative height", value=1.0, min_value=0.0, max_value=1.0, step=0.01
    )

    st.header("Filtering parameters")
    gradient_threshold = st.number_input(
        "Gradient threshold", value=-100, min_value=-1000, max_value=0, step=1
    )
    energy_drop_threshold = st.number_input(
        "Energy drop threshold", value=0.01, min_value=0.0, max_value=1.0, step=0.01
    )

    tolerance = st.number_input(
        "ELM time tolerance [ms]", value=0.1, min_value=0.0, max_value=10.0, step=0.05
    )

    # st.header("Plotting parameters")
    t_range = st.slider(
        "Time range",
        min_value=t_lims[0],
        max_value=t_lims[1],
        value=t_lims,
        step=1e-4,
    )

tab1, tab2 = st.tabs(["Main", "Debug"])
with tab1:
    # Process data
    peaks, _, _ = elmd.find_peaks_in_signal(
        w_signal, width=width, distance=distance, rel_height=rel_height
    )
    peaks_ds = elmd.peaks_as_xarray(peaks, w_signal)
    filtered_peaks_ds = elmd.filter_peaks(
        peaks_ds,
        gradient_threshold=gradient_threshold,
        energy_drop_threshold=energy_drop_threshold,
    )

    # Plot data
    st.header("Energy signal")

    if vega_backend:
        vega_chart = elmd.plot_all_vega(w_signal, peaks_ds, filtered_peaks_ds, telm_signal)
        st.altair_chart(
            vega_chart.interactive(), theme="streamlit", use_container_width=True
        )
    else:
        fig, axs = elmd.plot_all(w_signal, peaks_ds, filtered_peaks_ds, telm_signal, t_range)
        st.pyplot(fig)


    st.header("Filtered peaks")
    st.dataframe(filtered_peaks_ds.to_dataframe())

    # Plot these quantities only if telm_signal is available
    if telm_signal is not None:
        test_results = elmd.detect_events(
            filtered_peaks_ds["t_start"], telm_signal, delta_t=tolerance
        )
        test_results_df = pd.DataFrame(test_results)

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "True Positives",
            len(test_results_df[test_results_df["flag"] == "True Positive"]),
        )
        col2.metric(
            "False Positives",
            len(test_results_df[test_results_df["flag"] == "False Positive"]),
        )
        col3.metric(
            "False Negatives",
            len(test_results_df[test_results_df["flag"] == "False Negative"]),
        )

        st.header("Comparison with preprocessed data")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Some header")
            st.dataframe(test_results_df)

        with col2:
            st.subheader("True positive only")
            st.dataframe(test_results_df[test_results_df["flag"] == "True Positive"])

with tab2:
    st.header("Signal df")
    st.dataframe(w_signal.to_dataframe())
    st.header("Peaks df")
    st.dataframe(peaks_ds.to_dataframe())
    st.header("Filtered peaks df")
    st.dataframe(filtered_peaks_ds.to_dataframe())
    st.header("Telm df")
    st.dataframe(telm_signal.to_dataframe())
