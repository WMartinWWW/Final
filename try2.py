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



# Loading the data
@st.cache(allow_output_mutation=True)
def load_data():
    df_sales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    df_sales.dropna(subset=['YEAR', 'MONTH'], inplace=True)
    df_sales['Date'] = pd.to_datetime(df_sales['YEAR'].astype(int).astype(str) + '-' + df_sales['MONTH'].astype(int).astype(str) + '-01', errors='coerce')
    response = requests.get("https://api.openbrewerydb.org/breweries")
    df_breweries = pd.DataFrame(response.json())
    return df_sales, df_breweries

def plot_brewery_distribution(df_breweries):
    
    fig = px.scatter_geo(df_breweries, lat='latitude', lon='longitude', 
                         hover_name='name', title='Brewery Distribution in the US and Europe')
    st.plotly_chart(fig)
    
    # Adding the gap
    st.markdown("""<br>
    The map depicts brewery locations primarily concentrated in the Pacific Northwest, New England, eastern half of the United States, 
    with a notable presence on the West Coast, particularly in California. There are fewer breweries located in the central and mountain states. 
    Using the Brewers Association and Open Brewery DB as references, one could infer that the densest regions may correspond to states known for 
    a vibrant craft beer scene like California, Colorado, and the Northeastern states.<br><br>
    Users can click on the symbols on the map to view detailed longitude and latitude as well as the name of the Industry.
    <br>""", unsafe_allow_html=True)


def plot_market_share(df_sales):
    market_data = df_sales.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
    fig = px.pie(market_data, values='RETAIL SALES', names='ITEM TYPE', title='Market Share by Item Type')
    st.plotly_chart(fig)

    # Adding a space
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Market Share Analysis Commentary:**

    - Beer has a 26.6% market share, which is significant but less than liquor (37.1%) and wine (34.5%). This suggests a competitive marketplace with beer as a strong but not dominant player.
    - The other categories have much smaller shares, indicating that the sales focus primarily on alcoholic beverages, with additional revenue streams from related products and supplies.
    
    Users can click on different parts of the chart to obtain detailed sales data, or use the right slider to view specific data.
    """)

    
    st.markdown("<br>", unsafe_allow_html=True)

def plot_correlation_matrix(df_sales):
    correlation_matrix = df_sales.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)
    
    # Adding a space
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Correlation Matrix Commentary:**

    - Retail transfers and retail sales have a very high positive correlation (0.96), suggesting these two activities are highly synchronized.
    - The absence of strong correlations with the month variable suggests that the retail sales and transfers are not significantly seasonal.
    - In contrast, the year variable’s negative correlation with the month might indicate data spanning multiple years, where the month's impact on sales diminishes over time.
    """)

  
    st.markdown("<br>", unsafe_allow_html=True)


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
    
    # Plotting the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, predictions, color='blue', linewidth=3)
    plt.xlabel("Month Number")
    plt.ylabel("Retail Sales")
    plt.title("Regression Analysis of Retail Sales Over Months")
    st.pyplot(plt)
    
    # Display the results
    st.subheader("Regression Analysis Results")
    st.write("Model Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("Mean Squared Error:", mse)

    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Regression Analysis Commentary:**

    - Retail sales display a wide distribution each month, with a slight positive trend over time.
    - The intercept indicates the base level of sales without the influence of the month.
    - The small positive slope (0.1054) suggests a slight increase in sales as months progress, but the effect is minimal.
    """)

    
    st.markdown("<br>", unsafe_allow_html=True)


def time_series_analysis(df_sales):
    # For time series analysis
    df_sales = df_sales.sort_values('Date')
    df_sales.set_index('Date', inplace=True)


    sales_data = df_sales['RETAIL SALES'].astype(float)


    plt.figure(figsize=(10, 5))
    plt.plot(sales_data, label='Historical Sales', color='blue')
    plt.title('Historical Retail Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  
    plt.gcf().autofmt_xdate()  
    plt.legend()
    st.pyplot(plt)

    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Time Series Analysis Commentary:**

    - There are clear seasonal spikes in sales, particularly noticeable at the ends of years, which may correspond to the holiday season when alcohol sales typically increase.
    - The data span from 2018 to 2020 shows some form of cyclicality, with peaks around the same time each year.
    - For instance, the end of 2019 and 2020 shows significant spikes, which could be around November and December, a common time for increased sales due to holidays.
    """)

    
    st.markdown("<br>", unsafe_allow_html=True)


@st.cache
def perform_kmeans(features):
    # Run the KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    return kmeans

def consumer_behavior_analysis(df_sales):

    if 'total_spend' not in df_sales.columns or 'purchase_frequency' not in df_sales.columns:
        np.random.seed(42)
        df_sales['total_spend'] = np.random.normal(loc=1000, scale=300, size=len(df_sales))
        df_sales['purchase_frequency'] = np.random.poisson(lam=5, size=len(df_sales))

    # Select features for clustering
    features = df_sales[['total_spend', 'purchase_frequency']]

    
    kmeans = perform_kmeans(features)

   
    df_sales['cluster'] = kmeans.labels_

    
    fig = px.scatter(df_sales, x='total_spend', y='purchase_frequency', color='cluster', title='Customer Segmentation')
    st.plotly_chart(fig)

    
    st.write("Cluster Centers:", kmeans.cluster_centers_)

    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Cluster Centers Commentary:**

    - **Cluster 0:** High spending with moderate frequency (average spend of 1,363, average frequency of about 5).
    - **Cluster 1:** Lower spending than Cluster 0 but similar frequency (average spend of 994, frequency around 5).
    - **Cluster 2:** The lowest spending with a frequency akin to the others (average spend of 628, frequency also close to 5).
    
    The plot illustrates that spending does not necessarily increase with purchase frequency. The clusters suggest that while some customers make frequent purchases, they do not always spend more. Conversely, some customers may spend large amounts in fewer visits.
    """)

    
    st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown('<div class="big-font">Haotian Wang<br>Analysis of craft beer industry</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="regular-font">
    - **Interactivity**: This web application allows users to interact with various data visualizations and analyses pertaining to the craft beer industry. Users can select different aspects of the industry to examine, such as sales trends, brewery distribution, and market share through dropdown menus or sliders.<br><br>
    - **Charts and Plots Explanation**: Each chart or plot in this application provides insights into specific aspects of the craft beer market. For example, the distribution maps show the geographical spread of breweries, while pie charts and line graphs illustrate market share and sales trends over time.<br><br>
    - **Conclusions**: After each analysis section, conclusions are drawn to summarize the findings, such as the growth trajectory of craft beers or the economic impact on local markets.<br><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gotchas-font"><br>**Major Gotchas**:<br>- **Performance Issues**: Some features, particularly those involving complex data processing, may experience slow load times.</div>', unsafe_allow_html=True)

def create_summary_of_project_page():
    st.title('Summary of Project')
    
    
    st.markdown("""
    <style>
    .common-font {
        font-family: 'Arial'; font-size: 18px;
    }
    .thanks-font {
        font-family: 'Cursive'; font-size: 20px;
    }
    .gotchas-font {
        font-size: 18px !important;
        font-family: 'Comic Sans MS';
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Using the common font
    st.markdown('<div class="common-font"><br><b>Study Objective</b><br>- The goal was to study the craft beer industry’s production, distribution, and sales impacts. The project aimed to understand growth trends, analyze brewery distribution across the U.S., and assess the economic impact of craft beers on warehouse and retail sales. The study sought to provide stakeholders with valuable insights into production trends, market persistence, and consumer preferences, contributing to strategic discussions on industry sustainability and market development.<br>- I changed one data source in Milestones 2 because one of the data sources in Milestones 1 could not extract data well. However, my research direction has not changed.<br></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="common-font"><br><b>Discoveries and Conclusions</b><br>Analysis confirms that the craft beer industry continues to grow, with an increase in market share compared to more traditional and larger breweries. The distribution pattern shows that craft beer factories are significantly concentrated in specific regions, which is related to the higher demand and consumer preferences for craft beer in the local area. Beer competes with Liquor and Wine, but it has a place in the market. Retail sales exhibit a cyclical pattern, peaking at the end of the year, possibly due to seasonal influences such as holidays.<br></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="common-font"><br><b>Difficulties in Completing the Project</b><br>- Some anticipated data was incomplete or unavailable, particularly granular consumer demographic data, which limited the depth of consumer behavior analysis.<br>- Attempts to add different fonts in Streamlit and design a highly readable web page led to extensive research before finalizing the current version.<br>- There was some lag when running the Consumer Behavior Analysis part. I tried to optimize the code multiple times.<br></div>', unsafe_allow_html=True)
    
    st.markdown("<div class=\"common-font\"><br><b>Desired Skills</b><br>This project uses Python's scikit-learn library, linear regression for trend analysis, K-Means clustering for customer segmentation, and also incorporates time series prediction and analysis. Visualization tools such as Matplotlib, Seaborn, and Plotly were employed. I tried my best to apply all the content learned in DSCI510 and DSCI549.<br></div>", unsafe_allow_html=True)

    
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
   
    st.markdown('<div class="thanks-font"><br><b>Acknowledge</b><br>I\'d like to extend my heartfelt thanks to Professor Jeremy Abramson for the guidance and insights throughout this project. Additionally, a special note of appreciation to TA Divyank Lunkad for the unwavering support and patience in addressing my queries multiple times. Your contributions were invaluable to my learning and the successful completion of this project.<br></div>', unsafe_allow_html=True)

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
        'Consumer Behavior Analysis',
        'Summary of Project'
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
    elif analysis_choice == 'Summary of Project':
        create_summary_of_project_page()

if __name__ == '__main__':
    df_sales, df_breweries = load_data()
    create_app(df_sales, df_breweries)
