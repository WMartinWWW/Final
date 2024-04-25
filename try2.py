import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the 'DATE' column in your CSV is named 'DATE' and is in a format that Pandas can parse.
# If the column has a different name or format, you will need to adjust the code accordingly.

@st.cache
def load_data():
    data = pd.read_csv('Warehouse_and_Retail_Sales.csv', on_bad_lines='skip')
    # Attempt to parse the 'DATE' column and set it as the index for time series analysis
    if 'DATE' in data.columns:
        data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
        data.set_index('DATE', inplace=True)
    return data

# Main app interface
def main():
    st.title('Craft Beer Industry Analysis Dashboard')

    # Load data
    df = load_data()

    # Sidebar - Filter settings
    st.sidebar.header('Filter Data')
    
    # Filtering by supplier
    supplier_list = df['SUPPLIER'].dropna().unique()
    supplier = st.sidebar.multiselect('Supplier', supplier_list, default=supplier_list)
    
    # Filtering by item type
    item_type_list = df['ITEM TYPE'].dropna().unique()
    item_type = st.sidebar.multiselect('Item Type', item_type_list, default=item_type_list)
    
    # Filtering by date range
    if 'DATE' in df.columns:
        date_range = st.sidebar.date_input('Date range', [df.index.min(), df.index.max()])
        filtered_df = df.loc[date_range[0]:date_range[1]]
    else:
        filtered_df = df

    # Filtering data by selected supplier and item type
    if supplier and item_type:
        filtered_df = filtered_df[filtered_df['SUPPLIER'].isin(supplier) & filtered_df['ITEM TYPE'].isin(item_type)]
    
    # Display data table
    st.header('Sales Data Overview')
    st.write(filtered_df)

    # Show statistics of the data
    st.header('Data Statistics')
    st.write(filtered_df.describe())

    # Plotting sales over time if 'DATE' column is available
    if 'DATE' in df.columns:
        st.header('Sales Over Time')
        fig, ax = plt.subplots()
        filtered_df.groupby(filtered_df.index).sum()['RETAIL SALES'].plot(ax=ax)
        plt.ylabel('Total Sales')
        st.pyplot(fig)

    # Additional visualizations go here
    # ...

    # Download data feature
    st.header('Download Data')
    st.download_button(label='Download CSV', data=filtered_df.to_csv().encode('utf-8'), file_name='filtered_data.csv', mime='text/csv')

    # Instructions or documentation about the app
    st.header('About the App')
    st.info(
        """
        This dashboard is designed to analyze the craft beer industry's production, 
        distribution, and retail sales impacts. You can filter the data based on supplier, item type, 
        and date range. The visualizations and statistics are updated in real-time based on your selections.
        """
    )

if __name__ == "__main__":
    main()
