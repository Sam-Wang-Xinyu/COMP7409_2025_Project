import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
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
