import streamlit as st
import pandas as pd
import plotly.express as px
import requests

def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(str) + '-' + df_sales['MONTH'].astype(str))
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

def plot_growth_trends(df_sales):
    df_grouped = df_sales.groupby('Date').agg({'RETAIL SALES':'sum'}).reset_index()
    fig = px.line(df_grouped, x='Date', y='RETAIL SALES', title='Growth Trends in Retail Sales')
    return fig

def plot_brewery_distribution(df_breweries):
    fig = px.scatter_geo(df_breweries, lat='latitude', lon='longitude', hover_name='name', title='Brewery Distribution in the US')
    return fig

def plot_supplier_performance(df_sales):
    df_grouped = df_sales.groupby('SUPPLIER')['RETAIL SALES'].sum().reset_index()
    fig = px.bar(df_grouped, x='SUPPLIER', y='RETAIL SALES', title='Supplier Contribution to Retail Sales')
    return fig

def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Growth Trends', 'Brewery Distribution', 'Supplier Performance', 'Item Type Analysis'])

    if analysis_choice == 'Growth Trends':
        st.write('Analyzing growth trends of the craft beer industry.')
        fig = plot_growth_trends(df_sales)
        st.plotly_chart(fig)

    elif analysis_choice == 'Brewery Distribution':
        st.write('Distribution of breweries across the US.')
        fig = plot_brewery_distribution(df_breweries)
        st.plotly_chart(fig)

    elif analysis_choice == 'Supplier Performance':
        st.write('Analyzing supplier contribution to retail sales.')
        fig = plot_supplier_performance(df_sales)
        st.plotly_chart(fig)

    elif analysis_choice == 'Item Type Analysis':
        st.write('Sales analysis by item type.')
        item_type = st.selectbox('Select Item Type', df_sales['ITEM TYPE'].unique())
        df_filtered = df_sales[df_sales['ITEM TYPE'] == item_type]
        fig = px.bar(df_filtered, x='ITEM DESCRIPTION', y='RETAIL SALES', title=f'Sales by Item Description for {item_type}')
        st.plotly_chart(fig)
        if st.checkbox('Show data table'):
            st.dataframe(df_filtered[['ITEM DESCRIPTION', 'RETAIL SALES']])

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
