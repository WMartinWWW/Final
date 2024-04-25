import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

def load_data():
    # Load CSV data
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    # API request to Open Brewery DB
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

def preprocess_data(df_sales, df_breweries):
    # Creating a new 'Date' column from 'YEAR' and 'MONTH'
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(str) + '-' + df_sales['MONTH'].astype(str))
    # Clean up the breweries data
    df_breweries.dropna(subset=['state'], inplace=True)
    return df_sales, df_breweries

def plot_growth_trends(df_sales):
    # Group by the new 'Date' column and sum the sales for each month
    df_grouped = df_sales.groupby('Date').agg({'RETAIL SALES':'sum'}).reset_index()
    fig = px.line(df_grouped, x='Date', y='RETAIL SALES', title='Growth Trends in Retail Sales')
    return fig

def plot_brewery_distribution(df_breweries):
    fig = px.scatter_geo(df_breweries, lat='latitude', lon='longitude', hover_name='name', title='Brewery Distribution in the US')
    return fig

def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', ['Growth Trends', 'Brewery Distribution'])

    if analysis_choice == 'Growth Trends':
        st.write('Analyzing growth trends of the craft beer industry.')
        fig = plot_growth_trends(df_sales)
        st.plotly_chart(fig)

    elif analysis_choice == 'Brewery Distribution':
        st.write('Distribution of breweries across the US.')
        fig = plot_brewery_distribution(df_breweries)
        st.plotly_chart(fig)

# Main function to run the app
if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    df_sales, df_breweries = preprocess_data(df_sales, df_breweries)
    create_app(df_sales, df_breweries)
