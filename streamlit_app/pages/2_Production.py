# streamlit_app/pages/2_Production.py

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
import requests

st.set_page_config(page_title="Production (MongoDB) + Area Selector", layout="wide")

# ---------- Constants: price areas → cities & coords ----------
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

# ---------- Mongo connection (from secrets) ----------
uri = st.secrets["mongo"]["uri"]
db_name = st.secrets["mongo"]["db"]
col_name = st.secrets["mongo"]["col"]

client = MongoClient(uri)
col = client[db_name][col_name]

# ---------- Helper: Open-Meteo ERA5 fetch (year fixed to 2021) ----------
@st.cache_data(show_spinner=False)
def fetch_era5_2021(lat: float, lon: float) -> pd.DataFrame:
    """Download ERA5 hourly data for 2021 (temperature_2m, precipitation)."""
    try:
        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2021-01-01",
            "end_date": "2021-12-31",
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
        st.error(f"Error fetching weather data: {e}")
        return None

# ---------- UI: Area selector (writes to session_state) ----------
st.title("Production Overview (MongoDB)")

# Compute available areas from DB (fallback to full list if DB empty)
docs_preview = col.find({}, {"_id": 0, "priceArea": 1}).limit(2000)
areas_in_db = sorted({d.get("priceArea") for d in docs_preview if d.get("priceArea")})
areas = areas_in_db if areas_in_db else list(PRICE_AREAS.keys())

area = st.radio("Choose electricity price area", areas, horizontal=True, key="area_select")
city = PRICE_AREAS.get(area, "Oslo")
lat, lon = CITY_COORDS[city]

# Persist selection for other pages
st.session_state["selected_area"] = area
st.session_state["selected_city"] = city
st.session_state["selected_coords"] = (lat, lon)

st.caption(f"Selected: **{area} — {city}**  (lat={lat}, lon={lon}). Weather year fixed to **2021**.")

# Pre-fetch and cache weather for other pages (and quick sanity)
with st.spinner("Fetching ERA5 (Open-Meteo) for 2021…"):
    wx2021 = fetch_era5_2021(lat, lon)

if wx2021 is None:
    st.error("❌ Failed to fetch weather data. Some features may not work.")
    wx2021 = pd.DataFrame()  # Empty fallback

st.session_state["weather_2021"] = wx2021

# ---------- Load production data from MongoDB ----------
docs = col.find(
    {"priceArea": area},  # filter early by area for speed
    {"_id": 0, "priceArea": 1, "productionGroup": 1, "startTime": 1, "quantityKwh": 1},
)
df = pd.DataFrame(list(docs))
if df.empty:
    st.warning(f"No production data found in MongoDB for area {area}.")
    st.stop()

# Types
df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce").fillna(0)

# ---------- Layout: two columns (pie left, line right) ----------
left, right = st.columns(2, gap="large")

with left:
    pie_df = (
        df.groupby("productionGroup", as_index=False)["quantityKwh"].sum().sort_values("quantityKwh", ascending=False)
    )
    fig_pie = px.pie(
        pie_df,
        names="productionGroup",
        values="quantityKwh",
        title=f"Total production — {area}",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    months = sorted(df["startTime"].dt.month.unique())
    month = st.selectbox("Select month", months)
    line_src = df[df["startTime"].dt.month == month]
    if line_src.empty:
        st.info("No data for this month.")
    else:
        line_df = (
            line_src.groupby(["startTime", "productionGroup"])["quantityKwh"]
            .sum()
            .unstack()
            .fillna(0)
        )
        fig_line = px.line(line_df, title=f"Hourly production — {area}, Month {month:02d}")
        st.plotly_chart(fig_line, use_container_width=True)

# ---------- Provenance ----------
with st.expander("Data sources"):
    st.markdown(
        "- **Production**: Elhub → stored in MongoDB Atlas (queried live).\n"
        "- **Weather (for other pages)**: Open-Meteo ERA5, variables: `temperature_2m`, `precipitation`, **year=2021**.\n"
        "- City mapping: Oslo (NO1), Kristiansand (NO2), Trondheim (NO3), Tromsø (NO4), Bergen (NO5)."
    )
