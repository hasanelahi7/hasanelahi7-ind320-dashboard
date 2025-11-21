# IND320 Dashboard - Energy & Weather Analytics

**Course:** IND320 - Data to Decision
**Student:** Hasan Elahi
**Institution:** NMBU

## Links

- **GitHub:** https://github.com/hasanelahi7/hasanelahi7-ind320-dashboard
- **Streamlit App:** https://hasanelahi7-ind320-dashboard.streamlit.app/

## Overview

Dashboard analyzing Norwegian energy production/consumption and weather data across price areas NO1-NO5. Includes time series analysis, forecasting, anomaly detection, and interactive maps.

**Data sources:**
- Elhub API (production/consumption 2021-2024)
- Open-Meteo ERA5 (weather data)
- NVE GeoJSON (price area boundaries)

**Tech stack:**
- Cassandra + Spark for data processing
- MongoDB Atlas for cloud storage
- Streamlit + Plotly for visualization

## Structure

```
notebooks/
├── IND320_Part1.ipynb    # Part 1: Dashboard basics
├── IND320_Part2.ipynb    # Part 2: Data sources
├── IND320_Part3.ipynb    # Part 3: Data quality
└── IND320_Part4.ipynb    # Part 4: Machine learning

streamlit_app/
├── app.py                # Homepage
└── pages/
    ├── 2_Production.py
    ├── 3_STL_and_Spectrogram.py
    ├── 4_Snow_Drift.py
    ├── 5_Correlation.py
    ├── 6_Outliers_and_LOF.py
    ├── 7_Forecasting.py
    └── 8_Map.py

data/
├── open-meteo-subset-1.csv
├── elhub_prod_snapshot.csv
└── Snow_drift.py
```

## Setup

**Requirements:** Python 3.8+, Cassandra, Java 11

```bash
git clone https://github.com/hasanelahi7/hasanelahi7-ind320-dashboard.git
cd hasanelahi7-ind320-dashboard
pip install -r requirements.txt
cassandra -f  # Start Cassandra
```

Configure MongoDB in `.streamlit/secrets.toml`, then:

```bash
cd streamlit_app
streamlit run app.py
```

## Submission

Each assignment submits a different notebook but uses the same repo and Streamlit app.

**Export notebook:**
```bash
jupyter nbconvert --to html notebooks/IND320_Part4.ipynb
```

**Submit to Canvas:**
- Notebook PDF/HTML (IND320_Part4.html)
- Repo link: https://github.com/hasanelahi7/hasanelahi7-ind320-dashboard
- App link: https://hasanelahi7-ind320-dashboard.streamlit.app/

**Note:** All notebooks (Part1-4) are in this single repository as per instructor feedback.
