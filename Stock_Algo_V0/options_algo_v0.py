import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, SGDRegressor, Perceptron
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from math import sqrt
from sklearn.metrics import r2_score

"""

https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth

prophet is a facebook timeseries forecasting model, i would be interested in using it in the future
"""

"""

So my goal is to predict options, using the previous 3 weeks of data to predict the next week

"""

def train_option_prediction_model(stock_ticker: str, time_frame='60d'): # i want time_frame to be the period of data we look at, like '15d'
    # right now I have period set to 180d, interval='1h', but I will try to shorten it
    # and see how that performs
    df = yf.download(tickers=stock_ticker, period=time_frame, interval='5m') # interval='1h' is what I have now, I want as much data as possible
    df.reset_index(inplace=True)
    df = df.rename(columns = {'Datetime':'ds'})
    df.rename(columns={"Close": "y"}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1, inplace=True)
    df['ds'] = df['ds'].dt.tz_localize(None)
    # print(f'len of data: {df}')

    # 60 has been my default
    # n_lookback = 32 # Length of input sequences (lookback period)
    n_lookback = 14 # Length of input sequences (lookback period)

    model = Prophet()
    model.add_regressor('Volume')
    model.fit(df)
 
    # Y_pred = model.make_future_dataframe(periods=32)
    """

    When you add a regressor, you have to add it to the orignal dataframe, the model, and the future / new data frame, 
    remeber that. I think it helped the model a ton too!

    """
    Y_pred = model.make_future_dataframe(periods=5)
    Y_pred['Volume'] = df['Volume']
    Y_pred['Volume'] = Y_pred['Volume'].fillna(0)
    
    Y_pred.tail()
    
    forecast = model.predict(Y_pred)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # print(forecast)

    # model.plot(forecast)
    # plt.show()
    # Plot the forecast

    # Extract the last predicted value and the last real value
    last_predicted_value = forecast.iloc[-1]['yhat']
    last_real_value = df.iloc[-1]['y']

    # Calculate the difference
    difference = last_predicted_value - last_real_value

    # Calculate the percentage increase or decrease
    percentage_change = (difference / last_real_value) * 100

    print(f"Last Predicted Value: {last_predicted_value}")
    print(f"Last Real Value: {last_real_value}")
    print(f"Difference: {difference}")
    print(f"Percentage Change: {percentage_change:.2f}%")

    # Extract the relevant date range from the forecast dataframe
    date_range = pd.date_range(start=forecast['ds'].min(), end=forecast['ds'].max(), freq='M')
    date_range_str = [date.strftime('%Y-%m-%d') for date in date_range]

    # # Plot the forecast
    # fig = model.plot(forecast)

    # # Customize x-axis ticks
    # plt.xticks(date_range, date_range_str, rotation=45, ha='right')
    # plt.show()



    # Generate the forecasts
    # X_last = df.iloc[:n_lookback]
    # Y_pred_scaled = model.predict(X_last.reshape(n_lookback, -1))
    # Y_pred = Y_pred_scaled

    # plt.title(stock_ticker)
    # plt.plot(Y_pred, label="svm prediction")
    # plt.plot(Y, color="red", label="test")
    # plt.xlabel("Time (minutes)")
    # plt.ylabel("Stock Price")
    # plt.legend()
    # plt.show()

    # Create a dataframe with future dates for prediction
    future = model.make_future_dataframe(periods=5) # periods=32 is what I had before, I am going to go to 5 days
    future['Volume'] = df['Volume']
    future['Volume'] = future['Volume'].fillna(0)

    # Make predictions
    forecast = model.predict(future)

    # Extract the actual values for the last n observations
    n = n_lookback  # Change this to the desired number of observations
    actual_values = df.iloc[-n:]['y'].values

    # Extract the predicted values for the last n observations
    predicted_values = forecast.iloc[-n:]['yhat'].values
    
    r2 = r2_score(actual_values, predicted_values)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(actual_values, predicted_values))
    scaled_rmse = 1 - (rmse / (actual_values.max() - actual_values.min()))

    print(f"scaled_rmse: {scaled_rmse:.2f}, R^2: {r2:.2f}")
    # # Plot actual vs. predicted values
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['ds'], df['y'], label='Actual Values', color='blue')
    # plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Values', color='red')
    # plt.title('Actual vs. Predicted Values with RMSE')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()

    # # Annotate the RMSE on the plot
    # plt.annotate(f'RMSE: {rmse:.2f}', xy=(0.02, 0.85), xycoords='axes fraction', fontsize=12)

    # plt.show()
    
    return percentage_change
