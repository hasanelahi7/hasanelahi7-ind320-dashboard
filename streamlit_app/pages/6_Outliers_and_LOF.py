import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fftpack import dct, idct
from sklearn.neighbors import LocalOutlierFactor
import requests

st.set_page_config(page_title="Outliers (SPC) & LOF Anomalies", layout="wide")

# Price areas → city & coords (must match page 2)
PRICE_AREAS = {"NO1":"Oslo","NO2":"Kristiansand","NO3":"Trondheim","NO4":"Tromsø","NO5":"Bergen"}
CITY_COORDS = {
    "Oslo": (59.9139, 10.7522),
    "Kristiansand": (58.1467, 7.9956),
    "Trondheim": (63.4305, 10.3951),
    "Tromsø": (69.6492, 18.9553),
    "Bergen": (60.3913, 5.3221),
}

def get_selection():
    area = st.session_state.get("selected_area", "NO1")
    city = PRICE_AREAS.get(area, "Oslo")
    lat, lon = CITY_COORDS[city]
    return area, city, lat, lon

@st.cache_data(show_spinner=False)
def fetch_era5_2021(lat: float, lon: float) -> pd.DataFrame:
    """Hourly Open-Meteo ERA5 for 2021 (temperature_2m, precipitation)."""
    try:
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": "2021-01-01", "end_date": "2021-12-31",
            "hourly": "temperature_2m,precipitation",
            "timezone": "UTC",
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        js = response.json()
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")
        return None

tab_spc, tab_lof = st.tabs(["SPC (Temperature)", "LOF (Precipitation)"])

# ---------- Tab 1: SPC (Temperature) ----------
with tab_spc:
    st.subheader("Temperature outliers via SATV (DCT high-pass) + SPC bands")

    area, city, lat, lon = get_selection()
    col1, col2 = st.columns(2)
    dct_keep_frac = col1.slider("DCT high-pass (drop low frequency fraction)", 0.01, 0.20, 0.05, 0.01)
    k_sigma       = col2.slider("SPC k·sigma (robust)", 2.0, 5.0, 3.5, 0.1)

    wx = st.session_state.get("weather_2021")
    if wx is None or (isinstance(wx, pd.DataFrame) and wx.empty):
        wx = fetch_era5_2021(lat, lon)

    if wx is None or wx.empty:
        st.error("❌ No weather data available. Please check the Production page first.")
        st.stop()

    # SATV pipeline
    s = pd.to_numeric(wx["temperature_2m"], errors="coerce").to_numpy()
    x = dct(s, norm="ortho")
    cutoff = int(len(x) * dct_keep_frac)
    x[:max(1, cutoff)] = 0.0
    satv = idct(x, norm="ortho")
    med = np.median(satv)
    mad = np.median(np.abs(satv - med)) or 1e-9
    sigma = 1.4826 * mad
    lo, hi = med - k_sigma * sigma, med + k_sigma * sigma
    out = (satv < lo) | (satv > hi)

    # Create Plotly figure
    fig = go.Figure()

    # Add temperature line
    fig.add_trace(go.Scatter(
        x=wx["time"],
        y=s,
        mode='lines',
        name='Temperature (°C)',
        line=dict(width=1)
    ))

    # Add outlier markers
    fig.add_trace(go.Scatter(
        x=wx["time"][out],
        y=s[out],
        mode='markers',
        name='Outliers',
        marker=dict(size=6, color='red')
    ))

    fig.update_layout(
        title=f"Temperature with SPC Outliers — {area}/{city} (2021)",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    pct_outliers = out.mean() * 100
    st.caption(f"Outliers: {int(out.sum())} / {len(s)}  ({pct_outliers:.2f}%).  "
               f"Bounds from SATV (MAD→σ): [{lo:.3f}, {hi:.3f}].")

# ---------- Tab 2: LOF (Precipitation) ----------
with tab_lof:
    st.subheader("Precipitation anomalies via Local Outlier Factor (LOF)")

    area2, city2, lat2, lon2 = get_selection()
    colA, colB = st.columns(2)
    contamination = colA.slider("Proportion of anomalies", 0.001, 0.05, 0.01, 0.001)
    n_neighbors   = colB.number_input("n_neighbors", 5, 100, 35, step=1)

    wx2 = st.session_state.get("weather_2021")
    if wx2 is None or (isinstance(wx2, pd.DataFrame) and wx2.empty):
        wx2 = fetch_era5_2021(lat2, lon2)

    if wx2 is None or wx2.empty:
        st.error("❌ No weather data available. Please check the Production page first.")
        st.stop()

    z = pd.to_numeric(wx2["precipitation"], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
    nn = min(max(5, int(n_neighbors)), max(5, len(z) - 1))
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=float(contamination))
    labels = lof.fit_predict(z)
    out2 = labels == -1

    # Create Plotly figure
    fig2 = go.Figure()

    # Add precipitation line
    fig2.add_trace(go.Scatter(
        x=wx2["time"],
        y=z.flatten(),
        mode='lines',
        name='Precipitation (mm)',
        line=dict(width=1)
    ))

    # Add anomaly markers
    fig2.add_trace(go.Scatter(
        x=wx2["time"][out2],
        y=z[out2].flatten(),
        mode='markers',
        name='Anomalies',
        marker=dict(size=6, color='red')
    ))

    fig2.update_layout(
        title=f"Precipitation with LOF Anomalies — {area2}/{city2} (2021)",
        xaxis_title="Time",
        yaxis_title="Precipitation (mm)",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Anomalies: {int(out2.sum())} / {len(z)}  ({out2.mean()*100:.2f}%).  "
               f"LOF params — contamination={float(contamination)}, n_neighbors={nn}.")

st.expander("Data source").write(
    "Weather: Open-Meteo ERA5 (2021), variables: temperature_2m and precipitation. "
    "Area selection read from Page 2; defaults to NO1/Oslo if none is set."
)
