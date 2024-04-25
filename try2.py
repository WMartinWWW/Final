import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

@st.cache(allow_output_mutation=True)
def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    df_sales.dropna(subset=['YEAR', 'MONTH'], inplace=True)
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(int).astype(str) + '-' + df_sales['MONTH'].astype(int).astype(str) + '-01', errors='coerce')
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

def plot_dynamic_time_series(df_sales):
    if df_sales['Date'].isnull().any():
        st.error("Invalid or missing date data detected.")
        return None
    min_date, max_date = df_sales['Date'].min(), df_sales['Date'].max()
    start_date, end_date = st.sidebar.select_slider(
        "Select Date Range:", options=pd.date_range(start=min_date, end=max_date, freq='M'),
        value=(min_date, max_date)
    )
    filtered_data = df_sales[(df_sales['Date'] >= start_date) & (df_sales['Date'] <= end_date)]
    fig = px.line(filtered_data, x='Date', y='RETAIL SALES', title='Dynamic Retail Sales Over Time')
    return fig

def plot_brewery_distribution(df_breweries):
    fig = px.scatter_geo(df_breweries, lat='latitude', lon='longitude', hover_name='name', title='Brewery Distribution in the US')
    return fig

def plot_market_share(df_sales):
    market_data = df_sales.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
    fig = px.pie(market_data, values='RETAIL SALES', names='ITEM TYPE', title='Market Share by Item Type')
    return fig

def plot_correlation_matrix(df_sales):
    correlation_matrix = df_sales.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

def perform_regression(df_sales):
    df_sales.dropna(subset=['Date', 'RETAIL SALES'], inplace=True)
    df_sales['Month_Num'] = df_sales['Date'].dt.month
    X = df_sales[['Month_Num']]
    y = df_sales['RETAIL SALES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.subheader("Regression Analysis")
    st.write("Model Coefficients:", model.coef_)
    st.write("Mean Squared Error:", mse)
    # Plotting regression results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, predictions, color='blue', linewidth=3)
    plt.xlabel("Month Number")
    plt.ylabel("Retail Sales")
    plt.title("Regression Analysis of Retail Sales Over Months")
    st.pyplot(plt)

def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Growth Trends', 'Brewery Distribution', 'Market Share Analysis', 'Correlation Matrix', 'Regression Analysis'])

    if analysis_choice == 'Growth Trends':
        fig = plot_dynamic_time_series(df_sales)
        if fig:
            st.plotly_chart(fig)

    elif analysis_choice == 'Brewery Distribution':
        fig = plot_brewery_distribution(df_breweries)
        st.plotly_chart(fig)

    elif analysis_choice == 'Market Share Analysis':
        fig = plot_market_share(df_sales)
        st.plotly_chart(fig)

    elif analysis_choice == 'Correlation Matrix':
        plot_correlation_matrix(df_sales)

    elif analysis_choice == 'Regression Analysis':
        perform_regression(df_sales)

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
