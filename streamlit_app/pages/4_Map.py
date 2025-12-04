# streamlit_app/pages/8_Map.py

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
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

def get_color_for_value(value, min_val, max_val):
    """Get color for choropleth based on value."""
    if max_val > min_val:
        norm = (value - min_val) / (max_val - min_val)
    else:
        norm = 0.5
    # Blue (low) to red (high)
    r = int(255 * norm)
    g = int(100 * (1 - norm))
    b = int(255 * (1 - norm))
    return f'#{r:02x}{g:02x}{b:02x}'

def create_folium_map(area_stats, selected_area=None, clicked_coords=None):
    """Create simple Folium map with clickable areas."""

    # Create base map
    m = folium.Map(location=[65.0, 13.0], zoom_start=4, tiles="OpenStreetMap")

    # Get value range for coloring
    max_value = max(area_stats.values()) if area_stats.values() else 1
    min_value = min(area_stats.values()) if area_stats.values() else 0

    # Add GeoJSON boundaries with choropleth coloring
    for area, geometry in PRICE_AREA_BOUNDARIES.items():
        if area not in PRICE_AREAS:
            continue

        value = area_stats.get(area, 0)
        color = get_color_for_value(value, min_value, max_value)

        # Create feature for GeoJSON
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "name": f"{area} - {PRICE_AREAS[area]}",
                "value": f"{value:,.0f} kWh"
            }
        }

        # Highlight selected area
        line_weight = 4 if area == selected_area else 2
        line_color = "black" if area == selected_area else "gray"

        # Add GeoJSON with hover highlighting and tooltip
        folium.GeoJson(
            feature,
            style_function=lambda x, c=color, lw=line_weight, lc=line_color: {
                'fillColor': c,
                'color': lc,
                'weight': lw,
                'fillOpacity': 0.5
            },
            highlight_function=lambda x: {'weight': 5, 'fillOpacity': 0.7},
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'value'],
                aliases=['Area:', 'Energy:'],
                localize=True
            )
        ).add_to(m)

    # Add city markers
    for area, city in PRICE_AREAS.items():
        lat, lon = CITY_COORDS[city]
        folium.Marker(
            location=[lat, lon],
            popup=f"{area} - {city}",
            tooltip=f"{area}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Add clicked coordinate marker if exists
    if clicked_coords:
        folium.Marker(
            location=[clicked_coords[0], clicked_coords[1]],
            popup=f"Selected: {clicked_coords[0]:.4f}°N, {clicked_coords[1]:.4f}°E",
            tooltip="Selected Location",
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)

    return m

# ---------- Main UI ----------

st.title("🗺️ Interactive Energy Map")
st.markdown("""
Visualize energy production or consumption across Norwegian price areas (NO1-NO5).

**Select a price area from the dropdown below** - the map will update instantly to show:
- Choropleth coloring based on mean energy values (blue = low, red = high)
- Selected area highlighted with black border and red star marker
- Coordinates automatically saved for Snow Drift analysis
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

# Price area selection (dropdown) - instant update
selected_area = st.selectbox(
    "📍 Select Price Area",
    options=list(PRICE_AREAS.keys()),
    format_func=lambda x: f"{x} - {PRICE_AREAS[x]}",
    help="Choose a price area to view statistics and save coordinates for Snow Drift analysis"
)

# Get coordinates for selected area
city = PRICE_AREAS[selected_area]
lat, lon = CITY_COORDS[city]

# Save to session state for Snow Drift page
st.session_state['map_clicked_lat'] = lat
st.session_state['map_clicked_lon'] = lon

# Compute statistics
with st.spinner("Computing area statistics..."):
    area_stats = compute_area_statistics(data_type, start_date, end_date)

# Get the value for selected area
selected_area_value = area_stats.get(selected_area, 0)

# Show current selection with value BEFORE map
st.info(f"📍 Selected: **{selected_area} - {city}** ({lat:.4f}°N, {lon:.4f}°E) | Energy: **{selected_area_value:,.0f} kWh** | Coordinates saved for Snow Drift")

# Display statistics for all areas
st.markdown("### 📊 Area Statistics")
cols = st.columns(5)
for i, (area, value) in enumerate(area_stats.items()):
    with cols[i]:
        st.metric(f"{area}", f"{value:,.0f} kWh")

# Create and display map
st.markdown("### 🗺️ Choropleth Map")

# Show legend before map
min_value = min(area_stats.values()) if area_stats.values() else 0
max_value = max(area_stats.values()) if area_stats.values() else 1

col_legend1, col_legend2 = st.columns([3, 1])
with col_legend1:
    st.caption("**Color Scale Legend:**")
with col_legend2:
    st.caption(f"Min: {min_value:,.0f} kWh → Max: {max_value:,.0f} kWh")

# Color gradient HTML
st.markdown(f"""
<div style="
    background: linear-gradient(to right,
        rgb(0, 100, 255),
        rgb(128, 50, 128),
        rgb(255, 0, 0)
    );
    height: 25px;
    border: 1px solid #ccc;
    border-radius: 3px;
    margin-bottom: 10px;
">
</div>
<p style="text-align: center; font-size: 12px; color: #666; margin-top: -5px;">
    Blue (Low) ← → Red (High)
</p>
""", unsafe_allow_html=True)

m = create_folium_map(area_stats, selected_area, (lat, lon))

# Add CSS to remove focus outline on map
st.markdown("""
<style>
iframe {
    outline: none !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# Display map (height reduced, no event tracking to prevent loading on click)
st_folium(m, width=800, height=450, returned_objects=[])

# Additional info
with st.expander("ℹ️ About This Map"):
    st.markdown("""
    **Features:**
    - **Dropdown Selection**: Choose a price area to view and save coordinates for Snow Drift
    - **Choropleth Coloring**: Areas are colored based on mean energy values (blue = low, red = high)
    - **Price Area Boundaries**: Official NVE GeoJSON data for NO1-NO5 Elspot regions
    - **Red Star Marker**: Shows your selected price area location

    **Price Areas:**
    - **NO1 (Oslo)**: Southeast Norway
    - **NO2 (Kristiansand)**: Southern Norway
    - **NO3 (Trondheim)**: Central Norway
    - **NO4 (Tromsø)**: Northern Norway
    - **NO5 (Bergen)**: Western Norway

    **Data Sources:**
    - Energy Data: Elhub API via MongoDB Atlas
    - Price Area Boundaries: NVE (Norwegian Water Resources and Energy Directorate) GeoJSON
    """)

# Debug info
if st.checkbox("Show Debug Info"):
    st.write("**Session State:**")
    st.write(f"- Clicked Latitude: {st.session_state.get('map_clicked_lat', 'Not set')}")
    st.write(f"- Clicked Longitude: {st.session_state.get('map_clicked_lon', 'Not set')}")
    st.write("**Area Statistics:**")
    st.write(area_stats)
