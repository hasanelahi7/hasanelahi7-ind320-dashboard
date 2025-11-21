# streamlit_app/pages/8_Map.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
from pathlib import Path

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

# Load official NVE GeoJSON data for price area boundaries
APP_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = APP_ROOT.parent / "data" / "elspot_price_areas.geojson"

@st.cache_data
def load_geojson_boundaries():
    """Load official NVE GeoJSON data for Elspot price areas."""
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Parse GeoJSON features into a dict keyed by area code
    boundaries = {}
    for feature in geojson["features"]:
        # ElSpotOmr is like "NO 1", "NO 2", etc - convert to "NO1", "NO2"
        area_name = feature["properties"]["ElSpotOmr"].replace(" ", "")
        geometry = feature["geometry"]
        boundaries[area_name] = geometry

    return boundaries

PRICE_AREA_BOUNDARIES = load_geojson_boundaries()

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
def load_energy_data(area: str, data_type: str, start_date, end_date):
    """Load energy data from MongoDB for specified time range.

    Args:
        area: Price area code (e.g., "NO1")
        data_type: "production" or "consumption"
        start_date: datetime object for range start
        end_date: datetime object for range end
    """
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
def compute_area_statistics(data_type: str, start_date, end_date):
    """Compute mean energy values for all price areas in the time interval.

    Args:
        data_type: "production" or "consumption"
        start_date: datetime object for range start
        end_date: datetime object for range end
    """
    area_stats = {}

    for area in PRICE_AREAS.keys():
        df = load_energy_data(area, data_type, start_date, end_date)
        if df is not None and not df.empty:
            area_stats[area] = df["quantityKwh"].mean()
        else:
            area_stats[area] = 0

    return area_stats

# ---------- Map Creation ----------

def extract_polygon_coords(geometry):
    """Extract lat/lon coordinates from GeoJSON Polygon geometry.

    Handles both Polygon (single ring) and MultiPolygon geometries.
    Returns list of (lat, lon) tuples for the exterior ring.
    """
    if geometry["type"] == "Polygon":
        # Polygon has array of rings, first is exterior
        coords = geometry["coordinates"][0]
        # GeoJSON is [lon, lat], we need [lat, lon] for Plotly
        return [(lat, lon) for lon, lat in coords]
    elif geometry["type"] == "MultiPolygon":
        # For MultiPolygon, use the largest polygon
        max_ring = max(geometry["coordinates"], key=lambda poly: len(poly[0]))
        coords = max_ring[0]
        return [(lat, lon) for lon, lat in coords]
    else:
        return []

def create_choropleth_map(area_stats, selected_area=None):
    """Create an interactive Plotly map with choropleth coloring using official NVE GeoJSON data."""

    # Prepare data for city markers
    lats = []
    lons = []
    texts = []

    max_value = max(area_stats.values()) if area_stats.values() else 1
    min_value = min(area_stats.values()) if area_stats.values() else 0

    for area, city in PRICE_AREAS.items():
        lat, lon = CITY_COORDS[city]
        lats.append(lat)
        lons.append(lon)
        value = area_stats.get(area, 0)
        texts.append(f"{area} - {city}<br>Mean: {value:,.0f} kWh")

    # Create figure
    fig = go.Figure()

    # Add price area boundaries from official NVE GeoJSON
    for area, geometry in PRICE_AREA_BOUNDARIES.items():
        if area not in PRICE_AREAS:
            continue  # Skip if not in our standard price areas

        # Extract coordinates from GeoJSON geometry
        boundary_coords = extract_polygon_coords(geometry)
        if not boundary_coords:
            continue

        lats_poly = [coord[0] for coord in boundary_coords]
        lons_poly = [coord[1] for coord in boundary_coords]

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
        title="Norwegian Price Areas - Energy Choropleth Map (NVE Official Boundaries)",
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

# Calculate date range (as datetime objects for MongoDB)
from datetime import timezone
end_date = datetime.now(timezone.utc)
if time_preset == "Last 7 Days":
    start_date = end_date - timedelta(days=7)
elif time_preset == "Last 30 Days":
    start_date = end_date - timedelta(days=30)
elif time_preset == "Last 90 Days":
    start_date = end_date - timedelta(days=90)
elif time_preset == "Last Year":
    start_date = end_date - timedelta(days=365)
else:  # All Data
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)  # Use earlier date to capture all available data

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
    area_stats = compute_area_statistics(data_type, start_date, end_date)

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
    - **Price Area Boundaries**: Official NVE GeoJSON data for NO1-NO5 Elspot regions
    - **Interactive**: Hover over areas to see detailed statistics
    - **Coordinate Selection**: Select an area to save coordinates for Snow Drift calculations

    **Price Areas:**
    - **NO1 (Oslo)**: Southeast Norway
    - **NO2 (Kristiansand)**: Southern Norway
    - **NO3 (Trondheim)**: Central Norway
    - **NO4 (Tromsø)**: Northern Norway
    - **NO5 (Bergen)**: Western Norway

    **Data Sources:**
    - Energy Data: Elhub API via MongoDB Atlas
    - Price Area Boundaries: NVE (Norwegian Water Resources and Energy Directorate) GeoJSON

    **Note:** Price area boundaries are official data from NVE's ElSpot_omraade layer retrieved via ArcGIS REST API.
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
