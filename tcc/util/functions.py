# %%
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
def get_data_ticker(ticker, period='5y', rolling=1):
    data_orig = yf.download(ticker, period=period)
    data_hist = pd.DataFrame(data_orig['Close'])
    data_hist['prev-day-1'] = data_hist['Close'].rolling(rolling).mean().shift(1)
    data_hist['prev-day-2'] = data_hist['Close'].rolling(rolling).mean().shift(2)
    data_hist['prev-day-3'] = data_hist['Close'].rolling(rolling).mean().shift(3)
    data_hist['ma_5'] = data_hist['Close'].rolling(21).mean()
    data_hist['ma_21'] = data_hist['Close'].rolling(21).mean()
    
    weights = pd.Series(range(1, 22))
    data_hist['wma_21'] = data_hist['Close'].rolling(21).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


    delta = data_hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()  # Ganhos médios
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()  # Perdas médias
    rs = gain / loss  # Strength relativa
    data_hist['rsi'] = 100 - (100 / (1 + rs))
    data_hist = data_hist.dropna()

    data_hist['tomorrow'] = data_hist['Close'].rolling(rolling).mean().shift(-1)

    return data_hist

# %%
def dias_uteis_entre_datas(start_date,end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    date_list = date_range.strftime('%Y-%m-%d').tolist()
    return date_list

# %%
def decompor_sinal(y, period):
    return seasonal_decompose(y, period=period)