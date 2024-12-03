# %%
import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# %%
def get_data_ticker(ticker, period='5y', rolling=1, dropna=True):
    # Baixar dados de fechamento
    data_orig = yf.download(ticker, period=period)
    data_hist = pd.DataFrame(data_orig['Close'])
    data_hist['prev-day-1'] = data_hist['Close'].rolling(rolling).mean().shift(1)
    data_hist['prev-day-2'] = data_hist['Close'].rolling(rolling).mean().shift(2)
    data_hist['prev-day-3'] = data_hist['Close'].rolling(rolling).mean().shift(3)
    data_hist['mm_5'] = data_hist['Close'].rolling(5).mean()
    data_hist['mm_21'] = data_hist['Close'].rolling(21).mean()
    # IRF
    delta = data_hist['Close'].diff(1)  # Diferença entre os dias
    gain = delta.where(delta > 0, 0)  # Ganhos
    loss = -delta.where(delta < 0, 0)  # Perdas
    avg_gain = gain.rolling(window=14, min_periods=1).mean()  # Média de ganhos
    avg_loss = loss.rolling(window=14, min_periods=1).mean()  # Média de perdas
    rs = avg_gain / avg_loss  # Cálculo do RS (força relativa)
    rsi = 100 - (100 / (1 + rs))  # Cálculo do RSI
    data_hist['RSI_14'] = rsi  # Adicionar coluna de RSI

    if dropna:
        data_hist = data_hist.dropna()
    data_hist['tomorrow'] = data_hist['Close'].rolling(rolling).mean().shift(-1)

    return data_hist

def prepare_new_record(df, new_record, rolling=1):
    data_orig = pd.DataFrame({'Close': new_record})
    data_hist = pd.concat([df,data_orig])
    data_hist = pd.DataFrame(data_orig['Close'])
    data_hist['prev-day-1'] = data_hist['Close'].rolling(rolling).mean().shift(1)
    data_hist['prev-day-2'] = data_hist['Close'].rolling(rolling).mean().shift(2)
    data_hist['prev-day-3'] = data_hist['Close'].rolling(rolling).mean().shift(3)
    data_hist['mm_5'] = data_hist['Close'].rolling(5).mean()
    data_hist['mm_21'] = data_hist['Close'].rolling(21).mean()
    data_hist['tomorrow'] = data_hist['Close'].rolling(rolling).mean().shift(-1)
    #IRF
    delta = data_hist['Close'].diff(1)
    gain = delta.where(delta > 0, 0)  # Ganhos
    loss = -delta.where(delta < 0, 0)  # Perdas
    avg_gain = gain.rolling(window=14, min_periods=1).mean()  # Média de ganhos
    avg_loss = loss.rolling(window=14, min_periods=1).mean()  # Média de perdas
    rs = avg_gain / avg_loss  # Cálculo do RS (força relativa)
    rsi = 100 - (100 / (1 + rs))  # Cálculo do RSI
    data_hist['RSI_14'] = rsi  # Adicionar coluna de RSI
    return data_hist
# %%

# %%
def dias_uteis_entre_datas(start_date,end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    date_list = date_range.strftime('%Y-%m-%d').tolist()
    return date_list

# %%
def gerar_dias_uteis(qtd_dias_uteis, offset=0):
    date_range = pd.bdate_range(start=pd.Timestamp.today(), periods=qtd_dias_uteis+offset)
    date_list = date_range.strftime('%Y-%m-%d').tolist()
    return date_list[offset:]

# %%
def decompor_sinal(y, period, two_sided=False):
    return seasonal_decompose(y, period=period, two_sided=two_sided, extrapolate_trend='freq')


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
'ALOS3',
'ALPA4',
'ABEV3',
'ASAI3',
'AURE3',
'AZUL4',
'AZZA3',
'B3SA3',
'BBSE3',
'BBDC3',
'BBDC4',
'BRAP4',
'BBAS3',
'BRKM5',
'BRAV3',
'BRFS3',
'BPAC11',
'CXSE3',
'CRFB3',
'CCRO3',
'CMIG4',
'COGN3',
'CPLE6',
'CSAN3',
'CPFE3',
'CMIN3',
'CVCB3',
'CYRE3',
'ELET3',
'ELET6',
'EMBR3',
'ENGI11',
'ENEV3',
'EGIE3',
'EQTL3',
'EZTC3',
'FLRY3',
'GGBR4',
'GOAU4',
'NTCO3',
'HAPV3',
'HYPE3',
'IGTI11',
'IRBR3',
'ITSA4',
'ITUB4',
'JBSS3',
'KLBN11',
'RENT3',
'LREN3',
'LWSA3',
'MGLU3',
'MRFG3',
'BEEF3',
'MRVE3',
'MULT3',
'PCAR3',
'PETR3',
'PETR4',
'RECV3',
'PRIO3',
'PETZ3',
'RADL3',
'RAIZ4',
'RDOR3',
'RAIL3',
'SBSP3',
'SANB11',
'STBP3',
'SMTO3',
'CSNA3',
'SLCE3',
'SUZB3',
'TAEE11',
'VIVT3',
'TIMS3',
'TOTS3',
'TRPL4',
'UGPA3',
'USIM5',
'VALE3',
'VAMO3',
'VBBR3',
'VIVA3',
'WEGE3',
'YDUQ3'
]