import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm

data = pd.read_pickle('sales_data_anomaly.pkl')
data['Time'] = data['Time'].dt.strftime('%Y-%m-%d')

def calculate_majority_voting(data, threshold):
    """
    Takes in a dataframe and a threshold value and creates new column for majority voting

    Parameters
    data : dataframe
    threshold : int as percentage

    returns: dataframe with new column for majority voting
    """
    anomaly_columns = ['Outlier_IF', 'Outlier_LOF', 'Outlier_IQR', 'Outlier_percentile']
    data['Sum_of_Anomalies'] = data[anomaly_columns].sum(axis=1)
    
    threshold_number = len(anomaly_columns) * (threshold / 100)
    
    data['Outlier_Majority'] = data['Sum_of_Anomalies'] >= threshold_number

    return data

def main():
    st.title('Anomaly Detection Dashboard')

    with st.sidebar:
        # Method Selection
        methods = {
            'Majority Voting': 'Outlier_Majority',
            'Isolation Forest': 'Outlier_IF',
            'Local Outlier Factor': 'Outlier_LOF',
            'Interquartile Range': 'Outlier_IQR',
            'Percentile Range': 'Outlier_percentile'
        }
        method = st.selectbox(
            'Select Anomaly Detection Method',
            list(methods.keys())
        )

        # Setting threshold for Majority Voting
        if method == 'Majority Voting':
            threshold = st.slider('Select Majority Threshold', 1, 100, 50)
            calculate_majority_voting(data, threshold)

    anomaly_column = methods[method]
    total_anomalies = data[anomaly_column].sum()

    # Sidebar for displaying total anomalies
    st.sidebar.title("Anomaly Summary")
    st.sidebar.markdown(f"**Method:** {method}")
    st.sidebar.markdown(f"**Total Anomalies Detected:** {total_anomalies}")

    anomaly_column = methods[method]
    data['Color'] = data[anomaly_column].apply(lambda x: 'Anomaly' if x else 'Normal')

    color_mapping = {'Normal': 'blue', 'Anomaly': 'red'}

    # Plotting
    fig = px.scatter(data, x='Time', y='Sales', color='Color', 
                    color_discrete_map=color_mapping,
                    width=1000, height=600)

    st.plotly_chart(fig)

    # Display dataframe of anomalies
    st.write('Data for Anomalies:')
    st.dataframe(data[data[anomaly_column] == True][['Series','Time', 'Sales', 'WK_Num', 'MMM', 'Calendar Year', 'Fiscal Year',
                                                 'Sales_Change_percent', 'Sales_Change', 'Outlier_IF', 'Outlier_LOF', 'Outlier_percentile', 'Outlier_IQR']])

if __name__ == "__main__":
    main()