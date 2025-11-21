# streamlit_app/pages/8_Map.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta

st.set_page_config(page_title="Interactive Map", layout="wide")

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

# Approximate boundaries for Norwegian price areas (simplified polygons)
PRICE_AREA_BOUNDARIES = {
    "NO1": [(59.0, 10.0), (59.0, 11.5), (60.5, 11.5), (60.5, 10.0), (59.0, 10.0)],
    "NO2": [(57.5, 6.5), (57.5, 8.5), (59.5, 8.5), (59.5, 6.5), (57.5, 6.5)],
    "NO3": [(62.0, 9.0), (62.0, 12.0), (64.5, 12.0), (64.5, 9.0), (62.0, 9.0)],
    "NO4": [(68.5, 16.0), (68.5, 20.0), (70.5, 20.0), (70.5, 16.0), (68.5, 16.0)],
    "NO5": [(59.5, 4.5), (59.5, 6.5), (61.5, 6.5), (61.5, 4.5), (59.5, 4.5)],
}

# ---------- Data Loading ----------

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
def load_energy_data(area: str, data_type: str, start_date: str, end_date: str):
    """Load energy data from MongoDB for specified time range."""
    try:
        client = get_mongo_client()
        if client is None:
            return None

        db_name = st.secrets["mongo"]["db"]
        dfs = []

        if data_type == "production":
            for col_name in ["elhub_production_2021", "elhub_production_2022_2024"]:
                try:
                    col = client[db_name][col_name]
                    docs = col.find(
                        {
                            "priceArea": area,
                            "startTime": {
                                "$gte": start_date,
                                "$lte": end_date
                            }
                        },
                        {"_id": 0, "priceArea": 1, "productionGroup": 1, "startTime": 1, "quantityKwh": 1}
                    )
                    df = pd.DataFrame(list(docs))
                    if not df.empty:
                        dfs.append(df)
                except:
                    continue
        else:  # consumption
            try:
                col = client[db_name]["elhub_consumption_2021_2024"]
                docs = col.find(
                    {
                        "priceArea": area,
                        "startTime": {
                            "$gte": start_date,
                            "$lte": end_date
                        }
                    },
                    {"_id": 0, "priceArea": 1, "consumptionGroup": 1, "startTime": 1, "quantityKwh": 1}
                )
                df = pd.DataFrame(list(docs))
                if not df.empty:
                    dfs.append(df)
            except:
                pass

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=True)
        combined["startTime"] = pd.to_datetime(combined["startTime"], utc=True)
        combined["quantityKwh"] = pd.to_numeric(combined["quantityKwh"], errors="coerce").fillna(0)

        return combined

    except Exception as e:
        st.error(f"Error loading energy data: {e}")
        return None

@st.cache_data(show_spinner=False)
def compute_area_statistics(data_type: str, start_date: str, end_date: str):
    """Compute mean energy values for all price areas in the time interval."""
    area_stats = {}

    for area in PRICE_AREAS.keys():
        df = load_energy_data(area, data_type, start_date, end_date)
        if df is not None and not df.empty:
            area_stats[area] = df["quantityKwh"].mean()
        else:
            area_stats[area] = 0

    return area_stats

# ---------- Map Creation ----------

def create_choropleth_map(area_stats, selected_area=None):
    """Create an interactive Plotly map with choropleth coloring."""

    # Prepare data for plotting
    lats = []
    lons = []
    texts = []
    colors = []

    max_value = max(area_stats.values()) if area_stats.values() else 1
    min_value = min(area_stats.values()) if area_stats.values() else 0

    for area, city in PRICE_AREAS.items():
        lat, lon = CITY_COORDS[city]
        lats.append(lat)
        lons.append(lon)
        value = area_stats.get(area, 0)
        texts.append(f"{area} - {city}<br>Mean: {value:,.0f} kWh")

        # Normalize color (0-1 scale)
        if max_value > min_value:
            norm_value = (value - min_value) / (max_value - min_value)
        else:
            norm_value = 0.5
        colors.append(norm_value)

    # Create figure
    fig = go.Figure()

    # Add price area boundaries as polygons
    for area, boundary in PRICE_AREA_BOUNDARIES.items():
        lats_poly = [coord[0] for coord in boundary]
        lons_poly = [coord[1] for coord in boundary]

        value = area_stats.get(area, 0)
        if max_value > min_value:
            norm_value = (value - min_value) / (max_value - min_value)
        else:
            norm_value = 0.5

        # Color scale: blue (low) to red (high)
        color_rgb = f"rgb({int(255*norm_value)}, {int(100*(1-norm_value))}, {int(255*(1-norm_value))})"

        # Highlight selected area
        line_width = 3 if area == selected_area else 1
        line_color = "black" if area == selected_area else "gray"

        fig.add_trace(go.Scattermapbox(
            lat=lats_poly,
            lon=lons_poly,
            mode='lines',
            fill='toself',
            fillcolor=color_rgb,
            line=dict(width=line_width, color=line_color),
            opacity=0.5,
            name=area,
            hoverinfo='text',
            text=f"{area} - {PRICE_AREAS[area]}<br>Mean: {value:,.0f} kWh",
            showlegend=True
        ))

    # Add city markers
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers+text',
        marker=dict(size=12, color='darkblue'),
        text=[area for area in PRICE_AREAS.keys()],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=texts,
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=65.0, lon=13.0),
            zoom=3.5
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Norwegian Price Areas - Energy Choropleth Map",
        hovermode='closest'
    )

    return fig

# ---------- Main UI ----------

st.title("🗺️ Interactive Energy Map")
st.markdown("""
Visualize energy production or consumption across Norwegian price areas (NO1-NO5).
The map uses choropleth coloring based on mean values over the selected time interval.
Click on a price area to select coordinates for the Snow Drift analysis.
""")

# Selection controls
col1, col2 = st.columns(2)

with col1:
    data_type = st.radio("Data Type", options=["production", "consumption"], horizontal=True)

with col2:
    # Time interval selection
    time_preset = st.selectbox(
        "Time Interval",
        options=["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "All Data (2021-2024)"]
    )

# Calculate date range
end_date = datetime.utcnow()
if time_preset == "Last 7 Days":
    start_date = end_date - timedelta(days=7)
elif time_preset == "Last 30 Days":
    start_date = end_date - timedelta(days=30)
elif time_preset == "Last 90 Days":
    start_date = end_date - timedelta(days=90)
elif time_preset == "Last Year":
    start_date = end_date - timedelta(days=365)
else:  # All Data
    start_date = datetime(2021, 1, 1)

start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

# Price area selection
selected_area = st.selectbox(
    "Select Price Area",
    options=list(PRICE_AREAS.keys()),
    format_func=lambda x: f"{x} - {PRICE_AREAS[x]}"
)

# Store coordinates in session state for Snow Drift page
if selected_area:
    city = PRICE_AREAS[selected_area]
    lat, lon = CITY_COORDS[city]
    st.session_state['map_selected_lat'] = lat
    st.session_state['map_selected_lon'] = lon
    st.session_state['map_selected_city'] = city
    st.info(f"📍 Selected: {city} ({lat:.4f}°N, {lon:.4f}°E) - Coordinates saved for Snow Drift analysis")

# Compute statistics
with st.spinner("Computing area statistics..."):
    area_stats = compute_area_statistics(data_type, start_str, end_str)

# Display statistics
st.markdown("### 📊 Area Statistics")
cols = st.columns(5)
for i, (area, value) in enumerate(area_stats.items()):
    with cols[i]:
        st.metric(f"{area}", f"{value:,.0f} kWh")

# Create and display map
st.markdown("### 🗺️ Choropleth Map")
fig = create_choropleth_map(area_stats, selected_area)
st.plotly_chart(fig, use_container_width=True)

# Additional info
with st.expander("ℹ️ About This Map"):
    st.markdown("""
    **Features:**
    - **Choropleth Coloring**: Areas are colored based on mean energy values (blue = low, red = high)
    - **Price Area Boundaries**: Simplified polygons representing NO1-NO5 regions
    - **Interactive**: Hover over areas to see detailed statistics
    - **Coordinate Selection**: Select an area to save coordinates for Snow Drift calculations

    **Price Areas:**
    - **NO1 (Oslo)**: Southeast Norway
    - **NO2 (Kristiansand)**: Southern Norway
    - **NO3 (Trondheim)**: Central Norway
    - **NO4 (Tromsø)**: Northern Norway
    - **NO5 (Bergen)**: Western Norway

    **Data Source:** Elhub API via MongoDB Atlas

    **Note:** Boundaries are simplified approximations for visualization purposes.
    """)

# Debug info
if st.checkbox("Show Debug Info"):
    st.write("**Session State:**")
    st.write(f"- Selected Area: {selected_area}")
    st.write(f"- Latitude: {st.session_state.get('map_selected_lat', 'Not set')}")
    st.write(f"- Longitude: {st.session_state.get('map_selected_lon', 'Not set')}")
    st.write(f"- City: {st.session_state.get('map_selected_city', 'Not set')}")
    st.write("**Area Statistics:**")
    st.write(area_stats)
