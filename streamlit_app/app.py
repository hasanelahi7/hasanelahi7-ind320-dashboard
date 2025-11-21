import streamlit as st
import pandas as pd
import os

# Configure page
st.set_page_config(
    page_title="IND320 Energy & Weather Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Load sample weather data with caching
@st.cache_data
def load_weather_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'open-meteo-subset-1.csv')
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading weather data: {e}")
        return None

# Home page
st.title("⚡ IND320 Energy & Weather Analytics Dashboard")
st.markdown("## Complete Project: Parts 1-4")

st.markdown("""
Welcome to the comprehensive Norwegian energy and weather analytics platform. This dashboard integrates
data from multiple sources to provide insights into energy production, consumption, weather patterns,
and predictive analytics.
""")

# Quick overview
st.markdown("### 📊 Project Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔋 Energy Data**
    - Production data (2021-2024) from Elhub API
    - Consumption data (2021-2024) from Elhub API
    - Analysis by price areas (NO1-NO5)
    - Multiple energy sources tracked
    """)

    st.markdown("""
    **🌦️ Weather Data**
    - ERA5 reanalysis from Open-Meteo
    - Hourly temperature, precipitation, wind
    - Coverage across Norway's price areas
    """)

with col2:
    st.markdown("""
    **📈 Analytics Features**
    - Time series decomposition (STL)
    - Anomaly detection (SPC, LOF)
    - Correlation analysis
    - SARIMAX forecasting
    - Snow drift calculations
    - Interactive geographic mapping
    """)

    st.markdown("""
    **💾 Data Infrastructure**
    - Apache Cassandra (local storage)
    - MongoDB Atlas (cloud database)
    - Apache Spark (distributed processing)
    """)

# Navigation guide
st.markdown("### 🧭 Navigation Guide")

st.markdown("""
Use the sidebar to navigate between different analysis pages:

**Data Overview:**
- **Production** - Energy production analysis with MongoDB integration

**Time Series Analysis:**
- **STL & Spectrogram** - Seasonal decomposition and frequency analysis
- **Correlation** - Sliding window correlation between weather and energy

**Anomaly Detection:**
- **Outliers & LOF** - Statistical process control and local outlier factor analysis

**Predictions & Models:**
- **Forecasting** - SARIMAX time series forecasting
- **Snow Drift** - Snow transport calculations and wind rose visualization

**Geographic Analysis:**
- **Map** - Interactive map with price areas and choropleth visualization
""")

# Quick stats
df = load_weather_data()
if df is not None:
    st.markdown("### 📉 Sample Weather Statistics (2020 Data)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data Points", f"{len(df):,}")
    with col2:
        if 'temperature_2m (°C)' in df.columns:
            st.metric("Avg Temperature", f"{df['temperature_2m (°C)'].mean():.1f}°C")
    with col3:
        if 'wind_speed_10m (m/s)' in df.columns:
            st.metric("Max Wind Speed", f"{df['wind_speed_10m (m/s)'].max():.1f} m/s")
    with col4:
        if 'precipitation (mm)' in df.columns:
            st.metric("Total Precipitation", f"{df['precipitation (mm)'].sum():.1f} mm")

# Project information
st.markdown("---")
st.markdown("### ℹ️ Project Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Course:** IND320 - Data to Decision
    **Student:** Hasan Elahi
    **Institution:** Norwegian University of Life Sciences (NMBU)
    """)

with col2:
    st.markdown("""
    **Links:**
    - [GitHub Repository](https://github.com/hasanelahi7/hasanelahi7-ind320-dashboard)
    - [Streamlit App](https://hasanelahi7-ind320-dashboard.streamlit.app/)
    """)

st.markdown("""
**Technology Stack:** Python, Streamlit, Plotly, Pandas, Apache Spark, Cassandra, MongoDB,
Statsmodels, Scikit-learn, Folium
""")
