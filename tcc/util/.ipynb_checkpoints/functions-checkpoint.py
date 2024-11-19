# %%
import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# %%
def get_data_ticker(ticker, period='5y', rolling=1):
    data_orig = yf.download(ticker, period=period)
    data_hist = pd.DataFrame(data_orig['Close'])
    data_hist['prev-day-1'] = data_hist['Close'].rolling(rolling).mean().shift(1)
    data_hist['prev-day-2'] = data_hist['Close'].rolling(rolling).mean().shift(2)
    data_hist['prev-day-3'] = data_hist['Close'].rolling(rolling).mean().shift(3)
    data_hist['mm_5'] = data_hist['Close'].rolling(5).mean()
    data_hist['mm_21'] = data_hist['Close'].rolling(21).mean()
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
    return seasonal_decompose(y, period=period, extrapolate_trend='freq')


# %%
def split_df_X_y(df_model):
    dates = df_model.index
    df_as_np = df_model.to_numpy()
    X = df_as_np[:,:-1]

    X = X.reshape(len(dates), X.shape[1], 1)
    y = df_as_np[:,-1]

    return (dates, X.astype(np.float32), y.astype(np.float32))

# %%
def find_repetition(sequence):
    length = len(sequence)
    
    # Testar todos os tamanhos possíveis de repetição
    for size in range(1, length // 2 + 1):
        pattern = sequence[:size]
        
        # Cria uma sequência repetida do tamanho do array original
        repeated = np.tile(pattern, length // size)
        
        # Verifica se a repetição é igual à sequência original
        if np.array_equal(repeated, sequence[:len(repeated)]):
            return pattern
            
    return None


# %%
def obter_sazonalidade_periodo(qtdDias, sazonalidade):
    padraoSazonalidade = find_repetition(sazonalidade)
    
    tamanhoSazonalidade = len(padraoSazonalidade)
    
    qtdCiclosCompletos = qtdDias//tamanhoSazonalidade

    posicaoFinal = qtdDias%tamanhoSazonalidade
    
    cicloSazonal = []
    for ciclo in range(qtdCiclosCompletos):
        cicloSazonal.extend(padraoSazonalidade)

    cicloSazonal.extend(padraoSazonalidade[:posicaoFinal])

    return cicloSazonal

# %%
tickers_ibov = [
    
]