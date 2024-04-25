import streamlit as st
import pandas as pd
import plotly.express as px
import requests

@st.cache
def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(str) + '-' + df_sales['MONTH'].astype(str))
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

@st.cache
def compute_item_type_data(df_sales):
    return {item_type: df_sales[df_sales['ITEM TYPE'] == item_type] for item_type in df_sales['ITEM TYPE'].unique()}

def plot_supplier_performance(df_sales):
    df_grouped = df_sales.groupby('SUPPLIER')['RETAIL SALES'].sum().reset_index()
    fig = px.bar(df_grouped, x='SUPPLIER', y='RETAIL SALES', title='Supplier Contribution to Retail Sales')
    return fig

def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Growth Trends', 'Brewery Distribution', 'Supplier Performance', 'Item Type Analysis'])

    item_data = compute_item_type_data(df_sales)

    if analysis_choice == 'Item Type Analysis':
        st.write('Sales analysis by item type.')
        item_type = st.selectbox('Select Item Type', df_sales['ITEM TYPE'].unique())
        df_filtered = item_data[item_type]
        fig = px.bar(df_filtered, x='ITEM DESCRIPTION', y='RETAIL SALES', title=f'Sales by Item Description for {item_type}')
        st.plotly_chart(fig)
        if st.checkbox('Show data table'):
            st.dataframe(df_filtered[['ITEM DESCRIPTION', 'RETAIL SALES']])
        if st.button('Download Data as CSV'):
            st.download_button('Download CSV', df_filtered.to_csv(index=False), file_name=f'{item_type}_sales_data.csv')

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
