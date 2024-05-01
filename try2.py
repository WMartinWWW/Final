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
def Loadata():
    DFsales = pd.read_csv('Warehouse_and_Retail_Sales.csv')
    DFsales.dropna(subset=['YEAR', 'MONTH'], inplace=True)
    DFsales['Date'] = pd.to_datetime(DFsales['YEAR'].astype(int).astype(str) + '-' + DFsales['MONTH'].astype(int).astype(str) + '-01', errors='coerce')
    response = requests.get("https://api.openbrewerydb.org/breweries")
    DFbrewer = pd.DataFrame(response.json())
    return DFsales, DFbrewer

def Plotbrewer(DFbrewer):
    fig = px.scatter_geo(DFbrewer, lat='latitude', lon='longitude', hover_name='name', title='Brewery Distribution in the US and Europe')
    st.plotly_chart(fig)
    st.markdown("""<br>The map depicts brewery locations primarily concentrated in the Pacific Northwest, New England, eastern half of the United States, with a notable presence on the West Coast, particularly in California. There are fewer breweries located in the central and mountain states. Using the Brewers Association and Open Brewery DB as references, one could infer that the densest regions may correspond to states known for a vibrant craft beer scene like California, Colorado, and the Northeastern states.<br><br>Users can click on the symbols on the map to view detailed longitude and latitude as well as the name of the Industry.<br>""", unsafe_allow_html=True)

def Plotmarket(DFsales):
    Market = DFsales.groupby('ITEM TYPE')['RETAIL SALES'].sum().reset_index()
    fig = px.pie(Market, values='RETAIL SALES', names='ITEM TYPE', title='Market Share by Item Type')
    st.plotly_chart(fig)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""**Market Share Analysis Commentary:** - Beer has a 26.6% market share, which is significant but less than liquor (37.1%) and wine (34.5%). This suggests a competitive marketplace with beer as a strong but not dominant player. - The other categories have much smaller shares, indicating that the sales focus primarily on alcoholic beverages, with additional revenue streams from related products and supplies. Users can click on different parts of the chart to obtain detailed sales data.""")
    st.markdown("<br>", unsafe_allow_html=True)

def PlotCorrMatrix(DFsales):
    CorrMatrix = DFsales.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(CorrMatrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""**Correlation Matrix Commentary:** - Retail transfers and retail sales have a very high positive correlation (0.96), suggesting these two activities are highly synchronized. - The absence of strong correlations with the month variable suggests that the retail sales and transfers are not significantly seasonal. - In contrast, the year variable’s negative correlation with the month might indicate data spanning multiple years, where the month's impact on sales diminishes over time.""")
    st.markdown("<br>", unsafe_allow_html=True)

def PerRegression(DFsales):
    DFsales.dropna(subset=['Date', 'RETAIL SALES'], inplace=True)
    DFsales['Month_Num'] = DFsales['Date'].dt.month
    X = DFsales[['Month_Num']]
    y = DFsales['RETAIL SALES']
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
    st.subheader("Regression Analysis Results")
    st.write("Model Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("Mean Squared Error:", mse)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""**Regression Analysis Commentary:** - Retail sales display a wide distribution each month, with a slight positive trend over time. - The intercept indicates the base level of sales without the influence of the month. - The small positive slope (0.1054) suggests a slight increase in sales as months progress, but the effect is minimal.""")
    st.markdown("<br>", unsafe_allow_html=True)

def Timeseries(DFsales):
    DFsales = DFsales.sort_values('Date')
    DFsales.set_index('Date', inplace=True)
    SaleData = DFsales['RETAIL SALES'].astype(float)
    plt.figure(figsize=(10, 5))
    plt.plot(SaleData, label='Historical Sales', color='blue')
    plt.title('Historical Retail Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.legend()
    st.pyplot(plt)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""**Time Series Analysis Commentary:** - There are clear seasonal spikes in sales, particularly noticeable at the ends of years, which may correspond to the holiday season when alcohol sales typically increase. - The data span from 2018 to 2020 shows some form of cyclicality, with peaks around the same time each year. - For instance, the end of 2019 and 2020 shows significant spikes, which could be around November and December, a common time for increased sales due to holidays.""")
    st.markdown("<br>", unsafe_allow_html=True)

def APPBuild(DFsales, DFbrewer):
    st.title('Craft Beer Industry Analysis')
    st.sidebar.title('Navigation')
    AnaData = st.sidebar.radio('Choose Analysis', ['Main Project Page', 'Brewery Distribution', 'Market Share Analysis', 'Correlation Matrix', 'Regression Analysis', 'Time Series Analysis', 'Consumer Behavior Analysis', 'Summary of Project'])
    if AnaData == 'Main Project Page':
        Mainpage()
    elif AnaData == 'Brewery Distribution':
        Plotbrewer(DFbrewer)
    elif AnaData == 'Market Share Analysis':
        Plotmarket(DFsales)
    elif AnaData == 'Correlation Matrix':
        PlotCorrMatrix(DFsales)
    elif AnaData == 'Regression Analysis':
        PerRegression(DFsales)
    elif AnaData == 'Time Series Analysis':
        Timeseries(DFsales)
    elif AnaData == 'Consumer Behavior Analysis':
        consumer_behavior_analysis(DFsales)  # Need to update this function name in your code as well.
    elif AnaData == 'Summary of Project':
        Summaryproj()

if __name__ == '__main__':
    DFsales, DFbrewer = Loadata()
    APPBuild(DFsales, DFbrewer)
