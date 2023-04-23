import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objs as go
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Activation, Flatten
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)
import yfinance as yf

favicon = "Images/bitcoin.png"
def set_page_config():
    st.set_page_config(
        page_title="Future Bitcoin Price Prediction",
        page_icon=favicon
    )
set_page_config()

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer:after{
                content:'Copyright © Bitcoin Price Prediction';
                display:block;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Future Bitcoin Price Prediction")
st.write("Here you can get the next 30 days future bitcoin predicted price in USD")

def get_bitcoin_price():
    # Get Bitcoin price data from Yahoo Finance in USD with daily frequency
    bitcoin_data = yf.download(tickers='BTC-USD', period='max', interval='1d')
    # Reset index to ensure that the DataFrame has a 'Date' column
    bitcoin_data = bitcoin_data.reset_index()
    # Return the Bitcoin price data
    return bitcoin_data

# Retrieve the Bitcoin price data
df = get_bitcoin_price()

#display the data
st.subheader("Future Bitcoin Next 30 Days Predicted Close Price Data in USD")

# Preprocessing
raw_dataset=df
# converting date to datetime
raw_dataset['Date'] = pd.to_datetime(raw_dataset['Date'], dayfirst=True)
# deleting rows with close values less than 100
delete = raw_dataset[raw_dataset['Close'] < 100].index
raw_dataset = raw_dataset.drop(delete)
raw_dataset = raw_dataset.round(2)
raw_dataset = raw_dataset.set_index(['Date'])
df = raw_dataset

# create the output variable for build the model
y = df['Close'].fillna(method='ffill')
y = y.values.reshape(-1, 1)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# generate the input and output sequences
n_lookback = 60  # length of input sequences (lookback period)
n_forecast = 30  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

#fit the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(None, 1)))
model.add(LSTM(units=50))
model.add(Dense(n_forecast))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, Y, epochs=10, batch_size=32, verbose=0)

# generate the forecasts
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)
Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# organize the results in a data frame
df_past = df[['Close']].reset_index()
df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = df_past.append(df_future).set_index('Date')
split_results = results[-60:]

# plot the reults future bitcoin price prediction

# Split the data into actual and forecast displays
display1 = split_results[0:30]['Actual']
display2 = split_results[30:]['Forecast']
# display the future price
st.write(display2)

# Set the date ranges for each display
data_range = split_results.index[0:30]
data_range2 = split_results.index[30:]

# Create traces for each data
trace_future = go.Scatter(x=data_range2, y=display2, name='BTC Future Prices', line=dict(color='green'))

layout = go.Layout(title={
        'text': 'Future Bitcoin Next 30 Days Price Forecasting',
        'font': {'size': 25}
    },
    xaxis={
        'title': 'Date',
        'titlefont': {'size': 22},
        'rangeslider': {'visible': True},
        'tickfont': {'size': 18} 
    },
    yaxis={
        'title': 'Close Price (USD)',
        'titlefont': {'size': 22},
        'tickfont': {'size': 18}
    },
    plot_bgcolor='white', 
    width=800, 
    height=700
)

# Create figure and add traces to it
fig = go.Figure(data=[trace_future], layout=layout)

# Display plot in Streamlit
st.plotly_chart(fig)