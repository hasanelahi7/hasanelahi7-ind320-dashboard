# streamlit_app/pages/5_Correlation.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from pymongo import MongoClient

st.set_page_config(page_title="Correlation Analysis", layout="wide")

# ---------- Constants ----------

PRICE_AREAS = {
    "NO1": "Oslo",
    "NO2": "Kristiansand",
    "NO3": "Trondheim",
    "NO4": "Tromsø",
    "NO5": "Bergen",
}

CITY_COORDS = {
    "Oslo": (59.9139, 10.7522),
    "Kristiansand": (58.1467, 7.9956),
    "Trondheim": (63.4305, 10.3951),
    "Tromsø": (69.6492, 18.9553),
    "Bergen": (60.3913, 5.3221),
}

# ---------- Data Loading Functions ----------

@st.cache_data(show_spinner=False)
def fetch_weather_data(lat: float, lon: float, start_date: str, end_date: str):
    """Fetch ERA5 weather data from Open-Meteo with error handling."""
    try:
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
            "timezone": "UTC",
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        js = response.json()
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

@st.cache_resource
def get_mongo_client():
    """Get MongoDB client with error handling."""
    try:
        uri = st.secrets["mongo"]["uri"]
        return MongoClient(uri, serverSelectionTimeoutMS=5000)
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_energy_data(area: str, year: int, data_type: str = "production"):
    """Load energy production or consumption data from MongoDB."""
    try:
        client = get_mongo_client()
        if client is None:
            return None

        db_name = st.secrets["mongo"]["db"]

        # Determine collection and group column based on year and type
        if data_type == "production":
            group_col = "productionGroup"
            if year == 2021:
                col_name = "elhub_production_2021"
            else:
                col_name = "elhub_production_2022_2024"
        else:  # consumption
            group_col = "consumptionGroup"
            col_name = "elhub_consumption_2021_2024"

        col = client[db_name][col_name]

        # Query data - don't use projection, get all fields
        docs = col.find({"priceArea": area})

        df = pd.DataFrame(list(docs))
        if df.empty:
            return None

        # Filter by year
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
        df = df[(df["startTime"].dt.year == year)]

        if df.empty:
            return None

        df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce").fillna(0)

        # Keep only necessary columns
        keep_cols = ["priceArea", group_col, "startTime", "quantityKwh"]
        df = df[keep_cols]

        return df
    except Exception as e:
        st.error(f"Error loading energy data: {e}")
        return None

# ---------- Correlation Functions ----------

@st.cache_data(show_spinner=False)
def compute_sliding_correlation(weather_df, energy_df, weather_col, energy_group, energy_group_col, window, lag):
    """Compute sliding window correlation with lag."""
    try:
        # Merge on time
        weather_subset = weather_df[["time", weather_col]].copy()
        weather_subset = weather_subset.rename(columns={weather_col: "weather_value"})

        # Filter energy data by group and aggregate
        energy_subset = energy_df[energy_df[energy_group_col] == energy_group].copy()
        energy_subset = energy_subset.groupby("startTime")["quantityKwh"].sum().reset_index()
        energy_subset = energy_subset.rename(columns={"startTime": "time", "quantityKwh": "energy_value"})

        # Merge
        merged = pd.merge(weather_subset, energy_subset, on="time", how="inner")
        if merged.empty or len(merged) < window:
            return None

        merged = merged.sort_values("time")

        # Apply lag
        if lag > 0:
            merged["energy_value"] = merged["energy_value"].shift(lag)
        elif lag < 0:
            merged["weather_value"] = merged["weather_value"].shift(-lag)

        merged = merged.dropna()

        if len(merged) < window:
            return None

        # Rolling correlation
        correlations = merged["weather_value"].rolling(window=window).corr(merged["energy_value"])
        result_df = merged[["time"]].copy()
        result_df["correlation"] = correlations

        return result_df.dropna()

    except Exception as e:
        st.error(f"Error computing correlation: {e}")
        return None

# ---------- Main UI ----------

st.title("📈 Sliding Window Correlation Analysis")
st.markdown("""
Analyze the correlation between meteorological properties and energy production/consumption
over time using a sliding window approach with configurable lag.
""")

# Price area selection
area = st.selectbox(
    "Price Area",
    options=list(PRICE_AREAS.keys()),
    format_func=lambda x: f"{x} - {PRICE_AREAS[x]}",
    help="Select Norwegian price area for analysis"
)

city = PRICE_AREAS[area]
lat, lon = CITY_COORDS[city]

# Year selection
year = st.selectbox("Year", options=[2021, 2022, 2023, 2024, 2025], index=4)

# Property selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Meteorological Property**")
    weather_property = st.selectbox(
        "Weather Variable",
        options=["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"],
        format_func=lambda x: x.replace("_", " ").title(),
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Energy Property**")
    energy_type = st.radio("Type", options=["production", "consumption"], horizontal=True, label_visibility="collapsed")

# Window and lag controls
col1, col2 = st.columns(2)

with col1:
    window_length = st.slider(
        "Window Length (hours)",
        min_value=24,
        max_value=720,
        value=168,  # 1 week
        step=24,
        help="Size of sliding window for correlation calculation"
    )

with col2:
    lag = st.slider(
        "Lag (hours)",
        min_value=-168,
        max_value=168,
        value=0,
        step=6,
        help="Positive lag: energy lags weather. Negative lag: weather lags energy."
    )

# Energy group selection (will be populated after loading data)
if st.button("🔄 Calculate Correlations", type="primary"):
    with st.spinner(f"Loading data for {area} ({year})..."):
        # Load energy data first to determine actual date range
        energy_df = load_energy_data(area, year, energy_type)
        if energy_df is None:
            st.error(f"❌ No {energy_type} data available for {area} in {year}")
            st.stop()

        # Get date range from energy data
        min_date = energy_df["startTime"].min().strftime("%Y-%m-%d")
        max_date = energy_df["startTime"].max().strftime("%Y-%m-%d")

        # Load weather data for the actual date range
        weather_df = fetch_weather_data(lat, lon, min_date, max_date)
        if weather_df is None:
            st.error("❌ Failed to load weather data")
            st.stop()

        # Get available energy groups
        group_col = f"{energy_type}Group"
        available_groups = sorted(energy_df[group_col].unique())

        st.success(f"✅ Data loaded: {len(weather_df):,} weather records, {len(energy_df):,} energy records (from {min_date} to {max_date})")

        # Group selection
        selected_group = st.selectbox(
            f"Select {energy_type.title()} Group",
            options=available_groups
        )

        # Calculate correlation
        with st.spinner("Calculating sliding window correlation..."):
            corr_df = compute_sliding_correlation(
                weather_df, energy_df, weather_property, selected_group, group_col, window_length, lag
            )

            if corr_df is None or corr_df.empty:
                st.error("❌ Insufficient data to compute correlation")
                st.stop()

            # Display results
            st.markdown("### 📊 Correlation Over Time")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Correlation", f"{corr_df['correlation'].mean():.3f}")
            with col2:
                st.metric("Max Correlation", f"{corr_df['correlation'].max():.3f}")
            with col3:
                st.metric("Min Correlation", f"{corr_df['correlation'].min():.3f}")
            with col4:
                st.metric("Std Deviation", f"{corr_df['correlation'].std():.3f}")

            # Plotly line chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=corr_df["time"],
                y=corr_df["correlation"],
                mode='lines',
                name='Correlation',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{x}</b><br>Correlation: %{y:.3f}<extra></extra>'
            ))

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            # Highlight strong correlations
            strong_pos = corr_df[corr_df['correlation'] > 0.5]
            if not strong_pos.empty:
                fig.add_trace(go.Scatter(
                    x=strong_pos["time"],
                    y=strong_pos["correlation"],
                    mode='markers',
                    name='Strong Positive (>0.5)',
                    marker=dict(color='green', size=8),
                    hovertemplate='<b>%{x}</b><br>Correlation: %{y:.3f}<extra></extra>'
                ))

            strong_neg = corr_df[corr_df['correlation'] < -0.5]
            if not strong_neg.empty:
                fig.add_trace(go.Scatter(
                    x=strong_neg["time"],
                    y=strong_neg["correlation"],
                    mode='markers',
                    name='Strong Negative (<-0.5)',
                    marker=dict(color='red', size=8),
                    hovertemplate='<b>%{x}</b><br>Correlation: %{y:.3f}<extra></extra>'
                ))

            fig.update_layout(
                title=f"Sliding Window Correlation: {weather_property.replace('_', ' ').title()} vs {selected_group} {energy_type.title()}",
                xaxis_title="Time",
                yaxis_title="Correlation Coefficient",
                yaxis=dict(range=[-1, 1]),
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            with st.expander("📖 Interpretation Guide"):
                st.markdown("""
                **Correlation Coefficient Ranges:**
                - **> 0.7**: Strong positive correlation
                - **0.3 to 0.7**: Moderate positive correlation
                - **-0.3 to 0.3**: Weak or no correlation
                - **-0.7 to -0.3**: Moderate negative correlation
                - **< -0.7**: Strong negative correlation

                **Lag Interpretation:**
                - **Positive lag**: Energy values are shifted forward (energy responds to weather)
                - **Negative lag**: Weather values are shifted forward (weather responds to energy - rare)
                - **Zero lag**: Simultaneous relationship

                **Findings:**
                - Look for correlation changes during extreme weather events
                - Seasonal patterns may affect correlation strength
                - Consider physical relationships (e.g., cold weather → heating demand)
                """)

            # Download data
            csv = corr_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Correlation Data (CSV)",
                data=csv,
                file_name=f"correlation_{area}_{year}_{weather_property}_{selected_group}.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Click 'Calculate Correlations' to begin analysis")

# Data source info
with st.expander("📄 Data Sources"):
    st.markdown(f"""
    - **Weather Data**: Open-Meteo ERA5 reanalysis for {city} ({lat:.4f}, {lon:.4f})
    - **Energy Data**: Elhub {energy_type} data from MongoDB Atlas
    - **Window Method**: Sliding window with configurable lag
    - **Calculation**: Pearson correlation coefficient
    """)
