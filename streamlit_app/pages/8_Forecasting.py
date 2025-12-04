# streamlit_app/pages/7_Forecasting.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from pymongo import MongoClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="SARIMAX Forecasting", layout="wide")

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
def load_energy_data(area: str, data_type: str = "production"):
    """Load all available energy data from MongoDB."""
    try:
        client = get_mongo_client()
        if client is None:
            return None

        db_name = st.secrets["mongo"]["db"]

        # Load from both collections and combine
        dfs = []

        if data_type == "production":
            for col_name in ["elhub_production_2021", "elhub_production_2022_2024"]:
                try:
                    col = client[db_name][col_name]
                    docs = col.find(
                        {"priceArea": area},
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
                    {"priceArea": area},
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
        combined = combined.sort_values("startTime")

        return combined

    except Exception as e:
        st.error(f"Error loading energy data: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_weather_data(lat: float, lon: float, start_year: int, end_year: int):
    """Fetch weather data for multiple years."""
    try:
        all_dfs = []
        for year in range(start_year, end_year + 1):
            try:
                url = "https://archive-api.open-meteo.com/v1/era5"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "hourly": "temperature_2m,precipitation,wind_speed_10m",
                    "timezone": "UTC",
                }
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                js = response.json()
                df = pd.DataFrame(js["hourly"])
                df["time"] = pd.to_datetime(df["time"], utc=True)
                all_dfs.append(df)
            except:
                # Skip years that don't have data (e.g., future years)
                continue

        if not all_dfs:
            return None

        return pd.concat(all_dfs, ignore_index=True).sort_values("time")
    except Exception as e:
        return None

# ---------- Forecasting Function ----------

def fit_sarimax(train_data, order, seasonal_order, exog_train=None):
    """Fit SARIMAX model with error handling."""
    try:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)
        return fitted
    except Exception as e:
        st.error(f"Error fitting model: {e}")
        return None

# ---------- Main UI ----------

st.title("🔮 SARIMAX Forecasting")
st.markdown("""
Time series forecasting using Seasonal AutoRegressive Integrated Moving Average with eXogenous variables (SARIMAX).
Configure all parameters and select exogenous weather variables for enhanced predictions.
""")

# Price area selection
area = st.selectbox(
    "Price Area",
    options=list(PRICE_AREAS.keys()),
    format_func=lambda x: f"{x} - {PRICE_AREAS[x]}"
)

city = PRICE_AREAS[area]
lat, lon = CITY_COORDS[city]

# Data type selection
data_type = st.radio("Data Type", options=["production", "consumption"], horizontal=True)

# Load data to get groups
with st.spinner(f"Loading {data_type} data..."):
    energy_df = load_energy_data(area, data_type)

if energy_df is None:
    st.error(f"❌ No {data_type} data available for {area}")
    st.stop()

# Group selection
group_col = f"{data_type}Group"
available_groups = sorted(energy_df[group_col].unique())
selected_group = st.selectbox(f"{data_type.title()} Group", options=available_groups)

# Filter by group and aggregate
group_data = energy_df[energy_df[group_col] == selected_group].copy()
group_data = group_data.groupby("startTime")["quantityKwh"].sum().reset_index()
group_data = group_data.set_index("startTime").asfreq("H", fill_value=0)

st.info(f"📊 Data range: {group_data.index.min()} to {group_data.index.max()} ({len(group_data):,} hours)")

# SARIMAX Parameters
st.markdown("### ⚙️ SARIMAX Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Non-Seasonal**")
    p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1, help="AutoRegressive order")
    d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1, help="Degree of differencing")
    q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1, help="Moving Average order")

with col2:
    st.markdown("**Seasonal**")
    P = st.number_input("P (Seasonal AR)", min_value=0, max_value=3, value=1)
    D = st.number_input("D (Seasonal Diff)", min_value=0, max_value=2, value=1)
    Q = st.number_input("Q (Seasonal MA)", min_value=0, max_value=3, value=1)
    m = st.number_input("m (Season length)", min_value=2, max_value=168, value=24, help="Hours in a season (e.g., 24 for daily)")

with col3:
    st.markdown("**Training & Forecast**")
    # Calculate available days from data
    available_hours = len(group_data)
    available_days = available_hours // 24
    max_train_days = max(7, available_days - 1)  # Reserve at least 1 day for validation
    default_train_days = min(21, max_train_days)  # Default to 3 weeks or available data

    train_days = st.number_input(
        "Training Days",
        min_value=7,
        max_value=min(365, max_train_days),
        value=default_train_days,
        help=f"Days of historical data to train on (max {max_train_days} days available)"
    )
    forecast_hours = st.slider("Forecast Horizon (hours)", min_value=6, max_value=168, value=48, step=6)

# Exogenous variables
st.markdown("### 🌦️ Exogenous Variables (Weather)")
use_exog = st.checkbox("Include weather variables", value=False, help="Add weather data as external predictors")

exog_vars = []
if use_exog:
    exog_vars = st.multiselect(
        "Select weather variables",
        options=["temperature_2m", "precipitation", "wind_speed_10m"],
        default=["temperature_2m"]
    )

# Forecast button
if st.button("🚀 Run Forecast", type="primary"):
    with st.spinner("Preparing data and fitting SARIMAX model..."):
        # Prepare training data
        train_hours = train_days * 24
        if len(group_data) < train_hours:
            st.error(f"❌ Insufficient data. Need at least {train_hours} hours, have {len(group_data)}")
            st.stop()

        train_data = group_data.iloc[-train_hours:]
        y_train = train_data["quantityKwh"]

        # Prepare exogenous variables if needed
        exog_train = None
        exog_forecast = None

        if use_exog and exog_vars:
            with st.spinner("Fetching weather data..."):
                start_year = train_data.index.min().year
                end_year = train_data.index.max().year

                # Ensure we don't request future data from ERA5 archive
                import datetime
                current_year = datetime.datetime.now().year
                if end_year > current_year - 1:
                    end_year = current_year - 1  # ERA5 archive lags by about a year

                weather_df = fetch_weather_data(lat, lon, start_year, end_year)

                if weather_df is None:
                    st.warning(
                        "⚠️ **Weather data unavailable** — Historical weather data not found for the selected date range. "
                        "The forecast will proceed using SARIMAX without exogenous variables (weather data). "
                        "This is still a valid forecast, but won't account for weather patterns."
                    )
                    use_exog = False
                else:
                    weather_df = weather_df.set_index("time")

                    # Align with training data
                    weather_aligned = weather_df.loc[train_data.index, exog_vars].fillna(method='ffill').fillna(0)
                    exog_train = weather_aligned

                    # Prepare future weather (use last known values - simplification)
                    last_values = weather_aligned.iloc[-1]
                    exog_forecast = pd.DataFrame([last_values] * forecast_hours, columns=exog_vars)

        # Fit model
        with st.spinner("Fitting SARIMAX model (this may take a minute)..."):
            order = (p, d, q)
            seasonal_order = (P, D, Q, m)

            fitted_model = fit_sarimax(y_train, order, seasonal_order, exog_train)

            if fitted_model is None:
                st.error("❌ Model fitting failed. Try adjusting parameters.")
                st.stop()

        # Generate forecast
        with st.spinner("Generating forecast..."):
            try:
                if use_exog and exog_forecast is not None:
                    forecast_result = fitted_model.get_forecast(steps=forecast_hours, exog=exog_forecast)
                else:
                    forecast_result = fitted_model.get_forecast(steps=forecast_hours)

                forecast_mean = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int()

                # Create future index
                last_time = train_data.index[-1]
                future_index = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=forecast_hours, freq='H')

                forecast_df = pd.DataFrame({
                    'forecast': forecast_mean.values,
                    'lower': forecast_ci.iloc[:, 0].values,
                    'upper': forecast_ci.iloc[:, 1].values
                }, index=future_index)

                # Display results
                st.success("✅ Forecast completed successfully!")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{fitted_model.aic:.2f}", help="Akaike Information Criterion (lower is better)")
                with col2:
                    st.metric("BIC", f"{fitted_model.bic:.2f}", help="Bayesian Information Criterion (lower is better)")
                with col3:
                    st.metric("Forecast Mean", f"{forecast_mean.mean():.0f} kWh")

                # Plot
                st.markdown("### 📈 Forecast Visualization")

                # Show last 7 days of historical + forecast
                plot_history_hours = min(168, len(train_data))  # Last 7 days
                plot_history = train_data.iloc[-plot_history_hours:]

                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=plot_history.index,
                    y=plot_history["quantityKwh"],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))

                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))

                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['upper'],
                    mode='lines',
                    name='Upper 95% CI',
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['lower'],
                    mode='lines',
                    name='Lower 95% CI',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    fill='tonexty',
                    showlegend=True
                ))

                fig.update_layout(
                    title=f"SARIMAX Forecast: {area} - {selected_group} {data_type.title()}",
                    xaxis_title="Time",
                    yaxis_title="Quantity (kWh)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Model summary
                with st.expander("📊 Model Summary"):
                    st.text(fitted_model.summary())

                # Download forecast
                csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download Forecast (CSV)",
                    data=csv,
                    file_name=f"forecast_{area}_{selected_group}_{data_type}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.stop()

else:
    st.info("👆 Configure parameters and click 'Run Forecast' to begin")

# Info
with st.expander("ℹ️ About SARIMAX"):
    st.markdown("""
    **SARIMAX Model Components:**

    - **AR (p)**: AutoRegressive - uses past values
    - **I (d)**: Integrated - degree of differencing to make data stationary
    - **MA (q)**: Moving Average - uses past forecast errors
    - **Seasonal (P, D, Q, m)**: Same components for seasonal patterns
    - **X (exogenous)**: External variables (e.g., weather)

    **Parameter Tuning Tips:**
    - Start with (1,1,1) x (1,1,1,24) for hourly data with daily seasonality
    - Increase p/q if residuals show autocorrelation
    - Use d=1 or d=2 for non-stationary data
    - Set m=24 for daily patterns, m=168 for weekly

    **Model Selection:**
    - Lower AIC/BIC indicates better fit
    - Balance complexity vs. performance
    - Validate on held-out test data
    """)
