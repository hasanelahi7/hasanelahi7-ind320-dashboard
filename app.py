import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="IND320 Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv('open-meteo-subset-1.csv')
    return df

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["Home", "Data Table", "Data Visualization", "About"])

# Load data
df = load_data()

if page == "Home":
    st.title("🌤️ Weather Data Dashboard")
    st.markdown("## Welcome to the IND320 Project Dashboard")
    st.write("This dashboard displays weather data analysis including temperature, precipitation, and wind measurements.")
    
    st.markdown("### Quick Stats:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", len(df))
    with col2:
        st.metric("Avg Temperature", f"{df['temperature_2m (°C)'].mean():.1f}°C")
    with col3:
        st.metric("Max Wind Speed", f"{df['wind_speed_10m (m/s)'].max():.1f} m/s")

elif page == "Data Table":
    st.title("📊 Data Table")
    st.write("Raw weather data from the CSV file:")
    st.dataframe(df)
    
    # Display first month using st.line_chart
    st.markdown("### First Month Data Visualization")
    first_month = df.head(24*30)  # Approximately first 30 days (720 hours)
    
    # Create line chart for first month
    chart_data = first_month[['temperature_2m (°C)', 'wind_speed_10m (m/s)']]
    st.line_chart(chart_data)

elif page == "Data Visualization":
    st.title("📈 Data Visualization")
    
    # Column selector
    columns_to_plot = ['temperature_2m (°C)', 'precipitation (mm)', 'wind_speed_10m (m/s)', 
                      'wind_gusts_10m (m/s)', 'wind_direction_10m (°)']
    
    selected_column = st.selectbox("Select column to plot:", 
                                  ["All columns"] + columns_to_plot)
    
    # Month range selector (using indices as proxy for months)
    max_months = 12
    selected_range = st.select_slider(
        "Select month range:",
        options=list(range(1, max_months + 1)),
        value=1,
        format_func=lambda x: f"Month {x}"
    )
    
    # Calculate data range based on selected months
    hours_per_month = len(df) // 12
    end_idx = selected_range * hours_per_month
    plot_data = df.head(end_idx).copy()
    
    # Convert time column to datetime
    plot_data['time'] = pd.to_datetime(df['time'].head(end_idx))
    
    # Create plot based on selection
    if selected_column == "All columns":
        fig = go.Figure()
        for col in columns_to_plot:
            fig.add_trace(go.Scatter(
                x=plot_data['time'],
                y=plot_data[col],
                name=col,
                mode='lines'
            ))
        fig.update_layout(
            title="All Weather Variables Over Time",
            xaxis_title="Date",
            yaxis_title="Values (Mixed Units)"
        )
    else:
        fig = px.line(plot_data, 
                     x='time',
                     y=selected_column, 
                     title=f"{selected_column} Over Time")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=selected_column
        )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "About":
    st.title("ℹ️ About")
    st.markdown("### Project Information")
    st.write("**Course:** IND320 - Data to Decision")
    st.write("**Project:** Part 1 - Dashboard Basics")
    st.write("**Student:** Hasan Elahi")
    
    st.markdown("### Links")
    st.write("**GitHub Repository:** https://github.com/hasanelahi7/hasanelahi7-ind320-dashboard")
    st.write("**Streamlit App:** https://hasanelahi7-ind320-dashboard.streamlit.app/")
