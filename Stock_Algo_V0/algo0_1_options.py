import numpy as np
import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import matplotlib.pyplot as plt
import schedule
import time
from datetime import datetime

# my stuff
from sell_stock import sell_stock
from train_model import train_model
from svm_model import train_svm_model
from options_algo_v0 import train_option_prediction_model


def stock_algo_v0_5(stock_ticker):
  # these model predictions return the percent gain or loss
  model_prediction_1 = train_option_prediction_model(stock_ticker, '15d')
  model_prediction_2 = train_option_prediction_model(stock_ticker, '30d')
  model_prediction_3 = train_option_prediction_model(stock_ticker, '60d')
  # model_prediction_4 = train_option_prediction_model(stock_ticker, '120d')
  # model_prediction_5 = train_option_prediction_model(stock_ticker, '240d')

  # predictions is a list of all of the model predictions
  # predictions = [model_prediction_1, model_prediction_2, model_prediction_3, model_prediction_4, model_prediction_5]
  predictions = [model_prediction_1, model_prediction_2, model_prediction_3]
  total_weight = np.mean(predictions)
  print(f'{stock_ticker} total weighted percent return is: {total_weight:.2f}%')

  if total_weight >= 2:
    print(f'{stock_ticker} looks like a good buy this week')
  else:
    print(f'{stock_ticker} looks like a bad buy this week')

# https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf#:~:text=LSTMs%20are%20one%20of%20the,model%20is%20not%20always%20straightforward.

stock_algo_v0_5('TQQQ') # try a bunch of different stocks
stock_algo_v0_5('SQQQ') # try a bunch of different stocks

