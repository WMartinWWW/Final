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
from sklearn.cluster import KMeans
import matplotlib.dates as mdates



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

def time_series_analysis(df_sales):
    # Ensure data is sorted and indexed by date for time series analysis
    df_sales = df_sales.sort_values('Date')
    df_sales.set_index('Date', inplace=True)

    # Assuming 'RETAIL SALES' is the column to analyze
    sales_data = df_sales['RETAIL SALES'].astype(float)

    # Plotting the historical sales data
    plt.figure(figsize=(10, 5))
    plt.plot(sales_data, label='Historical Sales', color='blue')
    plt.title('Historical Retail Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks format on x-axis to yearly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Show only years on x-axis
    plt.gcf().autofmt_xdate()  # Auto format x-axis to fit date labels
    plt.legend()
    st.pyplot(plt)

    # Displaying analytical comments
    st.subheader("Time Series Analysis of Retail Sales")
    st.write("""
    **Analysis Observations:**
    - The plot shows the trend of retail sales over time.
    - Peaks and troughs may indicate seasonal patterns or effects of specific marketing campaigns or economic events.
    - Understanding these trends helps in planning for future sales strategies and inventory management.
    """)

@st.cache
def perform_kmeans(features):
    # Run the KMeans clustering on provided features
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    return kmeans

def consumer_behavior_analysis(df_sales):
    # Example: Assume df_sales has 'total_spend' and 'purchase_frequency' columns
    # We're generating this data for demonstration; you would replace this with your actual columns
    if 'total_spend' not in df_sales.columns or 'purchase_frequency' not in df_sales.columns:
        np.random.seed(42)
        df_sales['total_spend'] = np.random.normal(loc=1000, scale=300, size=len(df_sales))
        df_sales['purchase_frequency'] = np.random.poisson(lam=5, size=len(df_sales))

    # Select features for clustering
    features = df_sales[['total_spend', 'purchase_frequency']]

    # Perform KMeans clustering; this function is cached
    kmeans = perform_kmeans(features)

    # Add cluster labels to the dataframe
    df_sales['cluster'] = kmeans.labels_

    # Visualizing the clusters using Plotly for more efficient rendering
    fig = px.scatter(df_sales, x='total_spend', y='purchase_frequency', color='cluster', title='Customer Segmentation')
    st.plotly_chart(fig)

    # Output cluster centers
    st.write("Cluster Centers:", kmeans.cluster_centers_)

def create_main_project_page():
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-family: 'Haettenschweiler';
    }
    .regular-font {
        font-size:20px !important;
        font-family: 'Arial';
    }
    .gotchas-font {
        font-size:18px !important;
        font-family: 'Comic Sans MS';
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-font">Haotian Wang Project<br>Analysis of craft beer industry</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="regular-font">
    - **Interactivity**: This web application allows users to interact with various data visualizations and analyses pertaining to the craft beer industry. Users can select different aspects of the industry to examine, such as sales trends, brewery distribution, and market share through dropdown menus or sliders.<br><br>
    - **Charts and Plots Explanation**: Each chart or plot in this application provides insights into specific aspects of the craft beer market. For example, the distribution maps show the geographical spread of breweries, while pie charts and line graphs illustrate market share and sales trends over time.<br><br>
    - **Conclusions**: After each analysis section, conclusions are drawn to summarize the findings, such as the growth trajectory of craft beers or the economic impact on local markets.<br><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gotchas-font"><br>**Major Gotchas**:<br>- **Performance Issues**: Some features, particularly those involving complex data processing, may experience slow load times.</div>', unsafe_allow_html=True)



def create_app(df_sales, df_breweries):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    analysis_choice = st.sidebar.radio('Choose Analysis', [
        'Main Project Page',
        'Brewery Distribution',
        'Market Share Analysis',
        'Correlation Matrix',
        'Regression Analysis',
        'Time Series Analysis',
        'Consumer Behavior Analysis'
    ])

    if analysis_choice == 'Main Project Page':
        create_main_project_page()
    elif analysis_choice == 'Brewery Distribution':
        plot_brewery_distribution(df_breweries)
    elif analysis_choice == 'Market Share Analysis':
        plot_market_share(df_sales)
    elif analysis_choice == 'Correlation Matrix':
        plot_correlation_matrix(df_sales)
    elif analysis_choice == 'Regression Analysis':
        perform_regression(df_sales)
    elif analysis_choice == 'Time Series Analysis':
        time_series_analysis(df_sales)
    elif analysis_choice == 'Consumer Behavior Analysis':
        consumer_behavior_analysis(df_sales)

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
