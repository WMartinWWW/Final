import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# Function to load and cache data
@st.cache
def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(str) + '-' + df_sales['MONTH'].astype(str))
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

# Function to plot time series data with dynamic ranges
def plot_dynamic_time_series(df_sales):
    start_date, end_date = st.sidebar.select_slider(
        "Select Date Range:",
        options=pd.date_range(start=df_sales['Date'].min(), end=df_sales['Date'].max(), freq='M'),
        value=(df_sales['Date'].min(), df_sales['Date'].max())
    )
    filtered_data = df_sales[(df_sales['Date'] >= start_date) & (df_sales['Date'] <= end_date)]
    fig = px.line(filtered_data, x='Date', y='RETAIL SALES', title='Dynamic Retail Sales Over Time')
    return fig

# Function to plot market share
def plot_market_share(df_sales):
    market_data = df_sales.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
    fig = px.pie(market_data, values='RETAIL SALES', names='ITEM TYPE', title='Market Share by Item Type')
    return fig

# Main app function
def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Dynamic Time Series', 'Market Share Analysis'])

    if analysis_choice == 'Dynamic Time Series':
        st.write('Dynamic analysis of retail sales over time.')
        fig = plot_dynamic_time_series(df_sales)
        st.plotly_chart(fig)

    elif analysis_choice == 'Market Share Analysis':
        st.write('Market share by item type in the craft beer industry.')
        fig = plot_market_share(df_sales)
        st.plotly_chart(fig)

# Run the app
if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
