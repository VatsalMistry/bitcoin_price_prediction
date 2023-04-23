import streamlit as st

favicon = "Images/bitcoin.png"
def set_page_config():
    st.set_page_config(
        page_title="Bitcoin Price Prediction",
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

st.title("Bitcoin Price Prediction")
st.write("Now you can see the latest price of bitcoin and future bitcoin predicted price on a single platform, bitcoin price prediction aims to forecast the future price movements of bitcoin based on various data analysis and machine learning techniques. This web application uses historical price data, market trends and other factors to develop predictive models that can estimate future prices within a certain degree of uncertainty. System typically uses neural network to analyze large volumes of data and identify patterns and trends that may indicate future price movements.")

st.subheader("Basic Information")
st.subheader("Bitcoin")
st.write("Bitcoin is a digital currency that operates independently of any central authority or bank. It was created in 2009 by an unknown person or group using the name Satoshi Nakamoto. Bitcoin transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain. Bitcoin can be used to purchase goods and services, and it can also be traded on various exchanges for other currencies or assets.")