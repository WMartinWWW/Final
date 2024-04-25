import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime

# Function to load and cache data
@st.cache
def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    # Creating a 'Date' column from 'YEAR' and 'MONTH'
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(str) + '-' + df_sales['MONTH'].astype(str))
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

# Function to plot the dynamic time series of sales
def plot_dynamic_time_series(df_sales):
    min_date, max_date = df_sales['Date'].min(), df_sales['Date'].max()
    start_date, end_date = st.sidebar.select_slider(
        "Select Date Range:",
        options=pd.date_range(start=min_date, end=max_date, freq='M'),
        value=(min_date, max_date)
    )
    filtered_data = df_sales[(df_sales['Date'] >= start_date) & (df_sales['Date'] <= end_date)]
    fig = px.line(filtered_data, x='Date', y='RETAIL SALES', title='Dynamic Retail Sales Over Time')
    return fig

# Function to plot brewery distribution
def plot_brewery_distribution(df_breweries):
    fig = px.scatter_geo(df_breweries, lat='latitude', lon='longitude', hover_name='name', title='Brewery Distribution in the US')
    return fig

# Function to plot supplier performance (assuming this is defined)
def plot_supplier_performance(df_sales):
    df_grouped = df_sales.groupby('SUPPLIER')['RETAIL SALES'].sum().reset_index()
    fig = px.bar(df_grouped, x='SUPPLIER', y='RETAIL SALES', title='Supplier Contribution to Retail Sales')
    return fig

# Main app function that includes all features
def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Growth Trends', 'Brewery Distribution', 'Supplier Performance'])

    if analysis_choice == 'Growth Trends':
        fig = plot_dynamic_time_series(df_sales)
        st.plotly_chart(fig)

    elif analysis_choice == 'Brewery Distribution':
        fig = plot_brewery_distribution(df_breweries)
        st.plotly_chart(fig)

    elif analysis_choice == 'Supplier Performance':
        fig = plot_supplier_performance(df_sales)
        st.plotly_chart(fig)

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
