import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
import pickle

# Set page config
st.set_page_config(page_title='Craft Beer Industry Analysis', layout='wide')

# Global variables
DATA_URL = 'Warehouse_and_Retail_Sales.csv'
BREWERS_STATS_URL = "https://www.brewersassociation.org/statistics-and-data/national-beer-stats/"
BREWERY_API_URL = "https://api.openbrewerydb.org/breweries"


@st.cache
def load_sales_data():
    """Load and cache the sales data"""
    data = pd.read_csv(DATA_URL)
    return data


@st.cache
def fetch_brewers_stats(url):
    """Fetch and cache the brewers statistics data"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.content  # assuming this returns the raw HTML content
    else:
        return "Failed to fetch data"


@st.cache
def fetch_breweries_data(api_url):
    """Fetch and cache the breweries data"""
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()  # assuming the API returns a json response
    else:
        return "Failed to fetch data"


def display_sales_data(sales_data):
    """Display sales data related components"""
    st.header("Sales Data Overview")
    if not sales_data.empty:
        st.write(sales_data.head())

        st.subheader("Sales by Item Type")
        item_type_sales = sales_data.groupby('ITEM TYPE')['RETAIL SALES'].sum().sort_values()
        st.bar_chart(item_type_sales)
    else:
        st.error("No sales data to display.")


def display_brewers_stats():
    """Display brewers stats related components"""
    st.header("Brewers Statistics")
    brewers_stats_html = fetch_brewers_stats(BREWERS_STATS_URL)
    st.markdown(brewers_stats_html, unsafe_allow_html=True)  # Displays raw HTML


def display_breweries_data():
    """Display breweries data related components"""
    st.header("Breweries Data Overview")
    breweries_data = fetch_breweries_data(BREWERY_API_URL)
    if isinstance(breweries_data, list):  # check if the response is a valid list
        st.write("Total Breweries:", len(breweries_data))
        breweries_df = pd.DataFrame(breweries_data)
        st.map(breweries_df)  # Displaying the map assumes the data has 'latitude' and 'longitude' columns
    else:
        st.error("No breweries data to display.")


def main():
    st.title("Craft Beer Industry Analysis Dashboard")
    
    # Load data
    sales_data = load_sales_data()

    # Display sales data components
    display_sales_data(sales_data)

    # Display brewers statistics
    display_brewers_stats()

    # Display breweries data
    display_breweries_data()

    # Assumptions and Conclusions section
    st.header("Assumptions and Conclusions")
    st.write("""
        Here we could discuss the assumptions made during the analysis, such as assuming linear growth in craft beer sales, or considering only certain types of breweries for specific analyses.
        The conclusions can cover trends observed in the data, for instance, a growing market share for craft beers, or correlations found between the number of breweries in a region and local sales volumes.
    """)

    # Interactive widgets for exploration
    st.sidebar.header("Data Exploration Controls")
    # Implement widgets for filtering and interacting with the dataset
    # ...


if __name__ == "__main__":
    main()

