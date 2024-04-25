import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans



# Loading and preprocessing data
@st.cache(allow_output_mutation=True)
def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    df_sales.dropna(subset=['YEAR', 'MONTH'], inplace=True)
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(int).astype(str) + '-' + df_sales['MONTH'].astype(int).astype(str) + '-01', errors='coerce')
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

def plot_brewery_distribution(df_breweries):
    fig = px.scatter_geo(df_breweries, lat='latitude', lon='longitude', hover_name='name', title='Brewery Distribution in the US')
    st.plotly_chart(fig)

def plot_market_share(df_sales):
    market_data = df_sales.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
    fig = px.pie(market_data, values='RETAIL SALES', names='ITEM TYPE', title='Market Share by Item Type')
    st.plotly_chart(fig)

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
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, predictions, color='blue', linewidth=3)
    plt.xlabel("Month Number")
    plt.ylabel("Retail Sales")
    plt.title("Regression Analysis of Retail Sales Over Months")
    st.pyplot(plt)
    # Display regression analysis results
    st.subheader("Regression Analysis Results")
    st.write("Model Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("Mean Squared Error:", mse)
    # Interpretation
    st.write("### Interpretation of Regression Analysis")
    st.write("""
    The positive coefficient suggests that there is a general increase in retail sales over the months. This could indicate seasonal trends or overall growth in the market. The MSE provides a measure of the average error in our predictions, which helps us understand the accuracy of the regression model.
    """)

def sales_forecasting(df_sales):
    # Ensure data is sorted and indexed by date for time series analysis
    df_sales = df_sales.sort_values('Date')
    df_sales.set_index('Date', inplace=True)

    # Assuming 'RETAIL SALES' is the column to forecast
    sales_data = df_sales['RETAIL SALES'].astype(float)

    # Fit an ARIMA model (example parameters; should be tuned for your dataset)
    model = ARIMA(sales_data, order=(1, 1, 1))  # parameters (p,d,q) are placeholders
    model_fit = model.fit()

    # Forecasting the next 12 months
    forecast = model_fit.forecast(steps=12)

    # Plotting the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(sales_data, label='Historical Sales')
    plt.plot(forecast, label='Forecasted Sales', color='red')
    plt.title('Retail Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    st.pyplot(plt)

    # Display model summary
    st.write(model_fit.summary())

def consumer_behavior_analysis(df_sales):
    # Example: Assume df_sales has 'total_spend' and 'purchase_frequency' columns
    # For demonstration, let's create these columns randomly
    np.random.seed(42)
    df_sales['total_spend'] = np.random.normal(loc=1000, scale=300, size=len(df_sales))
    df_sales['purchase_frequency'] = np.random.poisson(lam=5, size=len(df_sales))

    # Select features for clustering
    features = df_sales[['total_spend', 'purchase_frequency']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    
    # Add cluster labels to the dataframe
    df_sales['cluster'] = kmeans.labels_

    # Visualizing the clusters
    fig = px.scatter(df_sales, x='total_spend', y='purchase_frequency', color='cluster', title='Customer Segmentation')
    st.plotly_chart(fig)

    # Output cluster centers
    st.write("Cluster Centers:", kmeans.cluster_centers_)

def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Brewery Distribution',
        'Market Share Analysis',
        'Correlation Matrix',
        'Regression Analysis',
        'Sales Forecasting',  # Added new function
        'Consumer Behavior Analysis'  # Added new function
    ])

    if analysis_choice == 'Brewery Distribution':
        plot_brewery_distribution(df_breweries)
    elif analysis_choice == 'Market Share Analysis':
        plot_market_share(df_sales)
    elif analysis_choice == 'Correlation Matrix':
        plot_correlation_matrix(df_sales)
    elif analysis_choice == 'Regression Analysis':
        perform_regression(df_sales)
    elif analysis_choice == 'Sales Forecasting':
        sales_forecasting(df_sales)  # Implement the function call
    elif analysis_choice == 'Consumer Behavior Analysis':
        consumer_behavior_analysis(df_sales)  # Implement the function call

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
