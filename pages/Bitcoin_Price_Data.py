import streamlit as st
from plotly import graph_objs as go
import yfinance as yf
from PIL import Image
import requests

favicon = "Images/bitcoin.png"
def set_page_config():
    st.set_page_config(
        page_title="Bitcoin Price Data",
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

st.title("Bitcoin Price Data")

# Load Bitcoin image from Images folder
btc_img = Image.open("Images/bitcoin.png")

# Define function to get Bitcoin price from Binance API
def get_live_bitcoin_price():
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {
        "symbol": "BTCUSDT"
    }
    response = requests.get(url, params=params)
    return float(response.json()["price"])

# Display Bitcoin image and price side by side
col1, col2 = st.columns(2)
with col1:
    st.image(btc_img, width=150)
with col2:
    st.title(f"Bitcoin Current Price: ${get_live_bitcoin_price()}")

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
st.subheader("Bitcoin Price Historical Data in USD")
st.write("Here you can see the bitcoin price data so far")
st.write(df)

# Create traces for each data
trace_open = go.Scatter(x=df['Date'], y=df['Open'], name='BTC Open Price', line=dict(color='red'))
trace_high = go.Scatter(x=df['Date'], y=df['High'], name='BTC High Price', line=dict(color='green'))
trace_low = go.Scatter(x=df['Date'], y=df['Low'], name='BTC Low Price', line=dict(color='yellow'))
trace_close = go.Scatter(x=df['Date'], y=df['Close'], name='BTC Close Price', line=dict(color='darkblue'))

layout = go.Layout(title={
        'text': 'Date Vs Price Graph of Bitcoin Price Data',
        'font': {'size': 25}
    },
    xaxis={
        'title': 'Date',
        'titlefont': {'size': 22},
        'rangeslider': {'visible': True},
        'tickfont': {'size': 18} 
    },
    yaxis={
        'title': 'Price (USD)',
        'titlefont': {'size': 22},
        'tickfont': {'size': 18}
    },
    plot_bgcolor='white', 
    width=900, 
    height=700
)

# Create figure and add traces to it
fig = go.Figure(data=[trace_open, trace_high, trace_low, trace_close], layout=layout)

# Display plot in Streamlit
st.plotly_chart(fig)