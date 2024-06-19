import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title('Financial Analysis Web App')

# Download stock data
st.sidebar.header('Stock Data')
ticker = st.sidebar.text_input('Enter Stock Ticker', 'GOTO.JK')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-31'))

data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Display the raw data
st.subheader('Raw Data')
st.write(data.tail())

# Processing historical stock data
n = int(len(data) / 2)
histo = data.iloc[:n]
histo['return'] = [None] + [(i - j) / j for (i, j)
                            in zip(histo['Close'].iloc[1:], histo['Close'].iloc[0:-1])]
histo = histo.dropna()

# Display the processed data
st.subheader('Processed Historical Data')
st.write(histo)

# Plot the closing price and return
st.subheader('Closing Price and Returns')
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(data['Close'], label='Closing Price')
ax[0].set_ylabel('Close Price')
ax[0].legend()

ax[1].plot(histo['return'], label='Return', color='orange')
ax[1].set_ylabel('Return')
ax[1].legend()

st.pyplot(fig)

# Add the prediction part
if st.sidebar.button('Run Model'):
    st.subheader('Prediction')

    # Preprocessing for model input
    scaler = MinMaxScaler()
    histo['Scaled Close'] = scaler.fit_transform(histo[['Close']])
    X = histo['Scaled Close'].values.reshape(-1, 1)
    y = histo['return'].values

    # Train the models
    gbm_model = GradientBoostingRegressor()
    gbm_model.fit(X, y)
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    # Make predictions on the second half of the data
    future_data = data.iloc[n:]
    future_data['Scaled Close'] = scaler.transform(future_data[['Close']])
    future_X = future_data['Scaled Close'].values.reshape(-1, 1)
    future_data['GBM Predicted Return'] = gbm_model.predict(future_X)
    future_data['LR Predicted Return'] = lr_model.predict(future_X)

    # Calculate predicted closing prices
    future_data['GBM Predicted Close'] = future_data['Close'].shift(
        1) * (1 + future_data['GBM Predicted Return'])
    future_data['LR Predicted Close'] = future_data['Close'].shift(
        1) * (1 + future_data['LR Predicted Return'])
    future_data = future_data.dropna()

    # Calculate errors
    future_data['GBM Absolute Error'] = np.abs(
        future_data['Close'] - future_data['GBM Predicted Close'])
    future_data['LR Absolute Error'] = np.abs(
        future_data['Close'] - future_data['LR Predicted Close'])

    # Calculate drift
    future_data['Actual Drift'] = future_data['Close'] / \
        future_data['Close'].shift(1) - 1
    future_data['GBM Predicted Drift'] = future_data['GBM Predicted Return']
    future_data['LR Predicted Drift'] = future_data['LR Predicted Return']
    future_data = future_data.dropna()

    # Display predicted data and errors
    st.write(future_data[['Close', 'GBM Predicted Close', 'LR Predicted Close',
             'GBM Absolute Error', 'LR Absolute Error']].tail())

    # Plot actual vs predicted closing price for both models
    st.subheader('Actual vs Predicted Closing Price')
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))
    ax2[0].plot(future_data['Close'], label='Actual Close')
    ax2[0].plot(future_data['GBM Predicted Close'],
                label='GBM Predicted Close', linestyle='--')
    ax2[0].set_ylabel('Price')
    ax2[0].legend()

    ax2[1].plot(future_data['Close'], label='Actual Close')
    ax2[1].plot(future_data['LR Predicted Close'],
                label='LR Predicted Close', linestyle='--')
    ax2[1].set_ylabel('Price')
    ax2[1].legend()

    st.pyplot(fig2)

    # Plot absolute errors
    st.subheader('Absolute Errors')
    fig3, ax3 = plt.subplots(2, 1, figsize=(10, 12))
    ax3[0].plot(future_data['GBM Absolute Error'],
                label='GBM Absolute Error', color='red')
    ax3[0].set_ylabel('Error')
    ax3[0].legend()

    ax3[1].plot(future_data['LR Absolute Error'],
                label='LR Absolute Error', color='blue')
    ax3[1].set_ylabel('Error')
    ax3[1].legend()

    st.pyplot(fig3)

    # Display drift
    st.subheader('Drift')
    st.write(future_data[['Actual Drift',
             'GBM Predicted Drift', 'LR Predicted Drift']].tail())

    # Plot drift
    st.subheader('Drift Comparison')
    fig4, ax4 = plt.subplots(3, 1, figsize=(10, 18))
    ax4[0].plot(future_data['Actual Drift'], label='Actual Drift')
    ax4[0].set_ylabel('Drift')
    ax4[0].legend()

    ax4[1].plot(future_data['GBM Predicted Drift'],
                label='GBM Predicted Drift', color='red')
    ax4[1].set_ylabel('Drift')
    ax4[1].legend()

    ax4[2].plot(future_data['LR Predicted Drift'],
                label='LR Predicted Drift', color='blue')
    ax4[2].set_ylabel('Drift')
    ax4[2].legend()

    st.pyplot(fig4)

    # Simulate stock price paths
    st.subheader('Simulated Stock Price Paths')
    num_simulations = 10
    num_days = len(future_data)
    last_close = future_data['Close'].iloc[-1]

    simulations = np.zeros((num_days, num_simulations))
    for i in range(num_simulations):
        simulated_prices = [last_close]
        for d in range(1, num_days):
            drift = future_data['GBM Predicted Drift'].iloc[d]
            shock = norm.ppf(np.random.rand()) * \
                future_data['GBM Absolute Error'].std()
            simulated_price = simulated_prices[-1] * (1 + drift + shock)
            simulated_prices.append(simulated_price)
        simulations[:, i] = simulated_prices

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.plot(simulations)
    ax5.set_ylabel('Simulated Prices')
    st.pyplot(fig5)
