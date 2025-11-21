# streamlit_app/pages/3_STL_and_Spectrogram.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
from pathlib import Path

st.set_page_config(page_title="STL & Spectrogram", layout="wide")

# ---------- Data load ----------
APP_ROOT = Path(__file__).resolve().parents[1]   # .../streamlit_app
ELHUB_CSV = APP_ROOT / "elhub_prod_snapshot.csv"

@st.cache_data(show_spinner=False)
def load_elhub(path: Path = ELHUB_CSV) -> pd.DataFrame:
    """Load Elhub production data from CSV with error handling."""
    try:
        df = pd.read_csv(path, parse_dates=["startTime"])
        df = df.sort_values("startTime")
        return df[["priceArea", "productionGroup", "startTime", "quantityKwh"]]
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

elhub = load_elhub()

if elhub.empty:
    st.error("❌ No data available. Please check the data file.")
    st.stop()

# Defaults from area selector on Page 2 (if present)
default_area = st.session_state.get("selected_area", "NO1")

tab_stl, tab_spec = st.tabs(["STL decomposition", "Spectrogram"])

# ---------- Tab 1: STL ----------
with tab_stl:
    st.subheader("Seasonal-Trend decomposition using LOESS (STL)")

    area = st.selectbox(
        "Price area",
        sorted(elhub["priceArea"].unique()),
        index=sorted(elhub["priceArea"].unique()).index(default_area),
    )
    groups = sorted(elhub.loc[elhub["priceArea"] == area, "productionGroup"].unique())
    group = st.selectbox("Production group", groups)

    col1, col2, col3, col4 = st.columns(4)
    period   = col1.number_input("Period (hours)", 24, 24*60, 24*7, step=24)
    seasonal = col2.slider("Seasonal smoother", 7, 61, 13, step=2)
    trend    = col3.slider("Trend smoother", 51, 601, 301, step=10)
    robust   = col4.checkbox("Robust", value=True)

    s = (
        elhub[(elhub.priceArea == area) & (elhub.productionGroup == group)]
        .set_index("startTime")["quantityKwh"]
        .asfreq("H")
        .interpolate()
    )

    if s.empty:
        st.info("No data for the selected combination.")
    else:
        res = STL(s, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust).fit()

        # Create Plotly subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
            vertical_spacing=0.08
        )

        # Add traces
        fig.add_trace(go.Scatter(x=s.index, y=res.observed, mode='lines', name='Observed', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=res.trend, mode='lines', name='Trend', line=dict(width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=res.seasonal, mode='lines', name='Seasonal', line=dict(width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=res.resid, mode='lines', name='Residual', line=dict(width=1)), row=4, col=1)

        # Update layout
        fig.update_layout(
            title_text=f"STL Decomposition — area={area}, group={group}, period={period}",
            height=700,
            showlegend=False,
            hovermode='x unified'
        )

        # Update all y-axes
        fig.update_yaxes(title_text="kWh", row=1, col=1)
        fig.update_yaxes(title_text="kWh", row=2, col=1)
        fig.update_yaxes(title_text="kWh", row=3, col=1)
        fig.update_yaxes(title_text="kWh", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Spectrogram ----------
with tab_spec:
    st.subheader("Spectrogram (hourly production)")

    area2 = st.selectbox(
        "Price area (spectrogram)",
        sorted(elhub["priceArea"].unique()),
        index=sorted(elhub["priceArea"].unique()).index(default_area),
        key="sg_area",
    )
    groups2 = sorted(elhub.loc[elhub["priceArea"] == area2, "productionGroup"].unique())
    group2 = st.selectbox("Production group (spectrogram)", groups2, key="sg_group")

    colA, colB = st.columns(2)
    window_len = colA.number_input("Window length (hours)", 24, 24*60, 24*14, step=24)
    overlap    = colB.slider("Window overlap", 0.0, 0.9, 0.5, 0.05)

    s2 = (
        elhub[(elhub.priceArea == area2) & (elhub.productionGroup == group2)]
        .set_index("startTime")["quantityKwh"]
        .asfreq("H")
        .interpolate()
    )

    if s2.empty:
        st.info("No data for the selected combination.")
    else:
        nperseg = int(window_len)
        noverlap = int(nperseg * float(overlap))
        f, tt, Sxx = spectrogram(s2.values, fs=1.0, nperseg=nperseg, noverlap=noverlap, scaling="density")

        # Create Plotly heatmap
        fig2 = go.Figure(data=go.Heatmap(
            z=10 * np.log10(Sxx + 1e-12),
            x=tt,
            y=f,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)")
        ))

        fig2.update_layout(
            title=f"Spectrogram — area={area2}, group={group2}",
            xaxis_title="Window index",
            yaxis_title="Frequency (cycles/hour)",
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)

st.expander("Data source").write(
    "Production: Elhub snapshot (hourly). Parameters exposed per assignment: "
    "price area, production group, period length, seasonal smoother, trend smoother, robust flag; "
    "and for spectrogram: window length, overlap."
)
