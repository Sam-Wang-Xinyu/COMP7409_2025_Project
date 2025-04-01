import ta
import numpy as np
import pandas as pd
import akshare as ak


def create_time_series_data(X, time_steps):
    X_t = []
    for i in range(len(X) - time_steps):
        X_t.append(X[i:i + time_steps])  # 时间步数据
        
    return np.array(X_t)


def get_data(symbol, start_date, end_date, model_num="1111"):
    
    stock_us_hist_df = ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date)
    df = stock_us_hist_df.iloc[:,[0,2,3,4,5]]
    df.columns = ['date','close','high','low','volume']

    # Momentum indicators
    if model_num[0] == '1':
        df['roc'] = ta.momentum.roc(close=df.close) # Rate of Change (ROC)
        df['rsi'] = ta.momentum.rsi(close=df.close) # Relative Strength Index (RSI)
        df['tsi'] = ta.momentum.tsi(close=df.close) # True strength index (TSI)

    # Volatility indicators
    if model_num[1] == '1':
        bb_indicator = ta.volatility.BollingerBands(close=df.close)
        df['bb_bbhi'] = bb_indicator.bollinger_hband_indicator() # Bollinger Band high indicator
        df['bb_bbli'] = bb_indicator.bollinger_lband_indicator() # Bollinger Band low indicator

    # Trend indicators
    if model_num[2] == '1':

        #aroon_indicator = ta.trend.AroonIndicator(close=df.close)
        aroon_indicator = ta.trend.AroonIndicator(df['high'], df['low'])
        macd_indicator = ta.trend.MACD(close=df.close)
        kst_indicator = ta.trend.KSTIndicator(close=df.close)
        df['aroon_down'] = aroon_indicator.aroon_down() # Aroon Down Channel
        df['aroon'] = aroon_indicator.aroon_indicator() # Aroon Indicator
        df['aroon_up'] = aroon_indicator.aroon_up() # Aroon Up Channel
        df['macd_line'] = macd_indicator.macd() # MACD Line
        df['macd_hist'] = macd_indicator.macd_diff() # MACD Histogram
        df['macd_signal'] = macd_indicator.macd_signal() # MACD Signal Line
        df['kst'] = kst_indicator.kst() # Know Sure Thing (KST)
        df['kst_diff'] = kst_indicator.kst_diff() # Diff Know Sure Thing (KST)
        df['kst_signal'] = kst_indicator.kst_sig() # Signal Line Know Sure Thing (KST)
        df['dpo'] = ta.trend.dpo(close=df.close) # Detrended Price Oscillator (DPO)
        df['trix'] = ta.trend.trix(close=df.close) # Trix (TRIX)
        df['sma_10'] = ta.trend.sma_indicator(close=df.close, window=10) # SMA n=10
        df['sma_20'] = ta.trend.sma_indicator(close=df.close, window=20) # SMA n=20
        df['sma_30'] = ta.trend.sma_indicator(close=df.close, window=30) # SMA n=30
        df['sma_60'] = ta.trend.sma_indicator(close=df.close, window=60) # SMA n=60
        df['ema_10'] = ta.trend.ema_indicator(close=df.close, window=10) # EMA n=10
        df['ema_20'] = ta.trend.ema_indicator(close=df.close, window=20) # EMA n=20
        df['ema_30'] = ta.trend.ema_indicator(close=df.close, window=30) # EMA n=30
        df['ema_60'] = ta.trend.ema_indicator(close=df.close, window=60) # EMA n=60

    # Volume indicators
    if model_num[3] == '1':
        df['obv'] = ta.volume.on_balance_volume(close=df.close, volume=df.volume) # On Balance Volume (OBV)
        df['vpt'] = ta.volume.volume_price_trend(close=df.close, volume=df.volume) # Volume-price trend (VPT)
        df['fi'] = ta.volume.force_index(close=df.close, volume=df.volume) # Force Index (FI)
        df['nvi'] = ta.volume.negative_volume_index(close=df.close, volume=df.volume) # Negative Volume Index (NVI)

    df = df.set_index('date')
    df['datetime'] = pd.to_datetime(df.index)
    df['min_sin'] = np.sin(2 * np.pi * df.datetime.dt.minute / 60)
    df['min_cos'] = np.cos(2 * np.pi * df.datetime.dt.minute / 60)
    df['hour_sin'] = np.sin(2 * np.pi * df.datetime.dt.hour / 60)
    df['hour_cos'] = np.cos(2 * np.pi * df.datetime.dt.hour / 60)
    df['day_sin'] = np.sin(2 * np.pi * df.datetime.dt.day / 30)
    df['day_cos'] = np.cos(2 * np.pi * df.datetime.dt.day / 30)
    df['month_sin'] = np.sin(2 * np.pi * df.datetime.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.datetime.dt.month / 12)

    df = df.drop(['datetime'], axis=1)
    df = df.drop(['high'], axis=1)
    df = df.drop(['low'], axis=1)
    
    df_na = df.dropna(axis=0)
    
    return df_na

