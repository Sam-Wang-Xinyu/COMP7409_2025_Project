import ta
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import akshare as ak
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


def create_time_series_data(X, time_steps):
    X_t = []
    for i in range(len(X) - time_steps):
        X_t.append(X[i:i + time_steps])  # 时间步数据
        
    return np.array(X_t)




def get_data(symbol, start_date, end_date):
    
    stock_us_hist_df = ak.stock_us_hist(symbol=symbol, start_date=start_date, end_date=end_date)
    df = stock_us_hist_df.iloc[:,[0,2,3,4,5]]
    df.columns = ['date','close','high','low','volume']

    # Momentum indicators
    df['roc'] = ta.momentum.roc(close=df.close) # Rate of Change (ROC)
    df['rsi'] = ta.momentum.rsi(close=df.close) # Relative Strength Index (RSI)
    df['tsi'] = ta.momentum.tsi(close=df.close) # True strength index (TSI)

    # Volatility indicators
    bb_indicator = ta.volatility.BollingerBands(close=df.close)
    df['bb_bbhi'] = bb_indicator.bollinger_hband_indicator() # Bollinger Band high indicator
    df['bb_bbli'] = bb_indicator.bollinger_lband_indicator() # Bollinger Band low indicator

    # Trend indicators
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



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis,:],
                           d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# 定义自注意力层
class SelfAttentionLayer(layers.Layer):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()

    def build(self, input_shape):
        self.features = input_shape[-1]
        self.Wq = self.add_weight(shape=(self.features, self.features), initializer='random_normal', trainable=True)
        self.Wk = self.add_weight(shape=(self.features, self.features), initializer='random_normal', trainable=True)
        self.Wv = self.add_weight(shape=(self.features, self.features), initializer='random_normal', trainable=True)

    def call(self, inputs):
        # Compute queries, keys, and values
        Q = tf.matmul(inputs, self.Wq)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)

        # Compute the attention scores
        score = tf.matmul(Q, K, transpose_b=True)

        # Scale the scores
        scale = tf.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        scaled_score = score / scale

        # Compute attention weights
        weights = tf.nn.softmax(scaled_score, axis=-1)

        # Compute the context vector
        context = tf.matmul(weights, V)
        return context

# 定义 LSTM 模型
def create_model(input_shape):
    
    # input embedding
    
    inputs = layers.Input(shape=input_shape) 
    
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    
    position = positional_encoding(input_shape[0],256)
    x = x+position
    # self attention layer
    attention_out = SelfAttentionLayer()(x)
    # add and norm
    x1 = x + attention_out
    x1 = layers.LayerNormalization()(x)
    
    # # forward
    x2 = layers.Dense(256, activation='relu')(x1)
    x2 = layers.Dense(256, activation='relu')(x2)
    # add and norm
    x = x2 + x1
    x = layers.LayerNormalization()(x)
    
    # self attention layer2
    x = x+position
    attention_out = SelfAttentionLayer()(x)
    # add and norm
    x1 = x + attention_out
    x1 = layers.LayerNormalization()(x)
    
    # # forward
    x2 = layers.Dense(256, activation='relu')(x1)
    x2 = layers.Dense(256, activation='relu')(x2)
    # add and norm
    x = x2 + x1
    x = layers.LayerNormalization()(x)
    
    x = layers.Flatten()(x)
    # 添加全连接层
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1)(x)  # 最后的输出层

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
