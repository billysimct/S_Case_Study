import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm

# Load and preprocess data
data = pd.read_pickle('data/sales_data_anomaly.pkl')
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

# Initialize Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1('Anomaly Detection Dashboard'),
    
    dcc.Dropdown(
        id='method-dropdown',
        options=[
            {'label': 'Majority Voting', 'value': 'Outlier_Majority'},
            {'label': 'Isolation Forest', 'value': 'Outlier_IF'},
            {'label': 'Local Outlier Factor', 'value': 'Outlier_LOF'},
            {'label': 'Interquartile Range', 'value': 'Outlier_IQR'},
            {'label': 'Percentile Range', 'value': 'Outlier_percentile'}
        ],
        value='Outlier_Majority'  # default value
    ),
    
    dcc.Slider(
        id='threshold-slider',
        min=1,
        max=100,
        value=50,  # default value
        marks={i: '{}%'.format(i) for i in range(0, 101, 10)},
        step=1
    ),
    
    dcc.Graph(id='anomaly-graph'),
    
    html.Div(id='anomaly-data')
])

# Define callback to update graph
@app.callback(
    [Output('anomaly-graph', 'figure'),
     Output('anomaly-data', 'children')],
    [Input('method-dropdown', 'value'),
     Input('threshold-slider', 'value')]
)
def update_graph(selected_method, threshold):
    if selected_method == 'Outlier_Majority':
        calculate_majority_voting(data, threshold)

    filtered_data = data.copy()
    filtered_data['Color'] = filtered_data[selected_method].apply(lambda x: 'Anomaly' if x else 'Normal')
    fig = px.scatter(filtered_data, x='Time', y='Sales', color='Color', color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'})
    
    anomaly_table = filtered_data[filtered_data[selected_method] == True][['Series', 'Time', 'Sales', 'WK_Num', 'MMM', 'Calendar Year', 'Fiscal Year', 'Sales_Change_percent', 'Sales_Change', 'Outlier_IF', 'Outlier_LOF', 'Outlier_percentile', 'Outlier_IQR']]

    return fig, html.Div([
        html.H3('Data for Anomalies:'),
        dash_table.DataTable(
            data=anomaly_table.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in anomaly_table.columns]
        )
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
