# streamlit_app/pages/4_Snow_Drift.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from pathlib import Path
import sys

st.set_page_config(page_title="Snow Drift Analysis", layout="wide")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------- Helper Functions from Snow_drift.py ----------

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """Compute potential wind-driven snow transport (Qupot) in kg/m."""
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total

def sector_index(direction):
    """Return sector index (0-15) for a wind direction in degrees."""
    return int(((direction + 11.25) % 360) // 22.5)

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """Compute cumulative transport for each of 16 wind sectors."""
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """Compute snow drifting transport according to Tabler (2003)."""
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe

    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"

    Qt = Qinf * (1 - 0.14 ** (F / T))

    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control
    }

# ---------- Data Fetching ----------

@st.cache_data(show_spinner=False)
def fetch_weather_data(lat: float, lon: float, year: int):
    """Fetch ERA5 weather data for a full year from Open-Meteo."""
    try:
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m",
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

# ---------- Main UI ----------

st.title("❄️ Snow Drift Analysis")
st.markdown("""
Calculate annual snow drift transport using the Tabler (2003) methodology.
Select coordinates from the Map page or use the defaults below.
""")

# Get coordinates from session state (set by Map page)
if "selected_coords" in st.session_state and st.session_state.selected_coords:
    default_lat, default_lon = st.session_state.selected_coords
    selected_area = st.session_state.get("selected_area", "Custom")
    st.success(f"✅ Using coordinates from Map: {selected_area} ({default_lat:.4f}, {default_lon:.4f})")
else:
    # Default to Bergen
    default_lat, default_lon = 60.3913, 5.3221
    st.info("ℹ️ No coordinates selected from Map page. Using Bergen as default.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    lat = st.number_input("Latitude", value=default_lat, min_value=-90.0, max_value=90.0, format="%.4f")
    lon = st.number_input("Longitude", value=default_lon, min_value=-180.0, max_value=180.0, format="%.4f")

with col2:
    year_range = st.slider(
        "Year Range",
        min_value=2010,
        max_value=2024,
        value=(2019, 2021),
        help="Select range of years to analyze (July to June season)"
    )

# Advanced parameters in expander
with st.expander("⚙️ Advanced Parameters"):
    col1, col2, col3 = st.columns(3)
    with col1:
        T = st.number_input("Max transport distance (m)", value=3000, min_value=100, max_value=10000)
    with col2:
        F = st.number_input("Fetch distance (m)", value=30000, min_value=100, max_value=100000)
    with col3:
        theta = st.number_input("Relocation coefficient", value=0.5, min_value=0.0, max_value=1.0, format="%.2f")

# Calculate button
if st.button("🔄 Calculate Snow Drift", type="primary"):
    with st.spinner("Fetching weather data and calculating snow drift..."):
        # Fetch data for all years in range
        all_results = []
        all_sectors = []

        for year in range(year_range[0], year_range[1] + 1):
            df = fetch_weather_data(lat, lon, year)
            if df is None:
                continue

            # Define season: July 1 to June 30 next year
            season_start = pd.Timestamp(year=year, month=7, day=1, tz='UTC')
            season_end = pd.Timestamp(year=year+1, month=6, day=30, hour=23, minute=59, second=59, tz='UTC')

            df_season = df[(df['time'] >= season_start) & (df['time'] <= season_end)].copy()

            if df_season.empty:
                continue

            # Calculate hourly Swe (snow water equivalent when temp < 1°C)
            df_season['Swe_hourly'] = df_season.apply(
                lambda row: row['precipitation'] if row['temperature_2m'] < 1 else 0, axis=1
            )
            total_Swe = df_season['Swe_hourly'].sum()

            # Get wind data
            wind_speeds = df_season["wind_speed_10m"].tolist()
            wind_dirs = df_season["wind_direction_10m"].tolist()

            # Calculate transport
            result = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
            result["season"] = f"{year}-{year+1}"
            all_results.append(result)

            # Calculate sector contributions
            sectors = compute_sector_transport(wind_speeds, wind_dirs)
            all_sectors.append(sectors)

        if not all_results:
            st.error("❌ No data available for selected years")
            st.stop()

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df["Qt (tonnes/m)"] = results_df["Qt (kg/m)"] / 1000

        # Calculate averages
        avg_sectors = np.mean(all_sectors, axis=0)
        overall_avg_qt = results_df["Qt (kg/m)"].mean()

        # Display results
        st.markdown("### 📊 Snow Drift Results")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Avg Qt", f"{overall_avg_qt/1000:.1f} tonnes/m")
        with col2:
            most_common_control = results_df["Control"].mode()[0] if not results_df.empty else "N/A"
            st.metric("Dominant Control", most_common_control)
        with col3:
            st.metric("Years Analyzed", len(results_df))

        # Results table
        st.markdown("#### Yearly Results")
        display_df = results_df[['season', 'Qt (tonnes/m)', 'Control']].copy()
        st.dataframe(display_df, use_container_width=True)

        # Wind Rose Plot (Plotly)
        st.markdown("### 🌹 Wind Rose - Directional Snow Transport")

        # Convert sectors to tonnes/m
        avg_sectors_tonnes = np.array(avg_sectors) / 1000.0

        # Create polar bar chart
        num_sectors = 16
        angles = np.arange(0, 360, 360/num_sectors)
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

        fig = go.Figure()

        fig.add_trace(go.Barpolar(
            r=avg_sectors_tonnes,
            theta=angles,
            width=360/num_sectors,
            marker=dict(
                color=avg_sectors_tonnes,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="tonnes/m")
            ),
            hovertemplate='<b>%{text}</b><br>Transport: %{r:.2f} tonnes/m<extra></extra>',
            text=directions
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(showticklabels=True, ticks=''),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=angles,
                    ticktext=directions,
                    direction='clockwise',
                    rotation=90
                )
            ),
            title=f"Average Directional Snow Transport<br>Overall: {overall_avg_qt/1000:.1f} tonnes/m",
            showlegend=False,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Fence height calculations
        with st.expander("🚧 Fence Height Recommendations"):
            st.markdown("Required effective fence height to store the drift:")

            fence_types = {
                "Wyoming": 8.5,
                "Slat-and-wire": 7.7,
                "Solid": 2.9
            }

            fence_results = []
            for season_data in all_results:
                Qt_tonnes = season_data["Qt (kg/m)"] / 1000
                row = {"Season": season_data["season"]}
                for fence_name, factor in fence_types.items():
                    H = (Qt_tonnes / factor) ** (1 / 2.2)
                    row[f"{fence_name} (m)"] = H
                fence_results.append(row)

            fence_df = pd.DataFrame(fence_results)
            st.dataframe(fence_df, use_container_width=True)

        # Data provenance
        with st.expander("📄 Data Source"):
            st.markdown(f"""
            - **Weather Data**: Open-Meteo ERA5 reanalysis
            - **Location**: Latitude {lat:.4f}, Longitude {lon:.4f}
            - **Variables**: temperature_2m, precipitation, wind_speed_10m, wind_direction_10m
            - **Season Definition**: July 1 to June 30
            - **Methodology**: Tabler (2003) snow drift transport model
            """)

else:
    st.info("👆 Click 'Calculate Snow Drift' to begin analysis")
