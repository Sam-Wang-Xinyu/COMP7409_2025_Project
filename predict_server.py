import ta
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from transformer import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from flask import Flask, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_temorrow_stock_price():
    
    TIMESTEP = 32
    current_date = datetime.now()

    # 计算100天前的日期
    days_300_ago = current_date - timedelta(days=300)
    current_date = current_date.strftime("%Y%m%d")
    days_300_ago = days_300_ago.strftime("%Y%m%d")

    df_na = get_data(symbol='106.TTE', start_date=days_300_ago, end_date=current_date)
    scaler = MinMaxScaler()
    
    x_values = scaler.fit_transform(df_na.values)
    x_values = x_values[:,1:]


    def create_time_series_data(X, time_steps):
        X_t = []
        for i in range(len(X) - time_steps):
            X_t.append(X[i:i + time_steps])  # 时间步数据
            
        return np.array(X_t)

    x_values = create_time_series_data(x_values , TIMESTEP)

    x_values = np.array([x_values[-1]])

    model1 = tf.keras.models.load_model('my_model.keras')

    predictions = model1.predict(x_values)
    dummy = np.zeros((len(predictions), df_na.shape[-1]))
    dummy[:, 0] = predictions[:, 0]
    predictions = scaler.inverse_transform(dummy)[:, 0]
        
    # 返回结果
    return jsonify({
        "tomorrow_stiock_price": predictions.tolist(),
    })






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567)
    