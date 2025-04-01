import ta
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from transformer import *
from utils import * 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from flask import Flask, jsonify, request
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_temorrow_stock_price():
    
    options = request.args.getlist('options')  # 获取传入的参数
    model_num = ""
    if "Momentum Feature" in options:
        model_num += '1'
    else:
        model_num += '0'
    if "Volatility Feature" in options:
        model_num += '1'
    else:
        model_num += '0'
    if "Trend Feature" in options:
        model_num += '1'
    else:
        model_num += '0'
    if "Volume Feature" in options:
        model_num += '1'
    else:
        model_num += '0'
    
    
    TIMESTEP = 32
    current_date = datetime.now()

    # 计算100天前的日期
    days_300_ago = current_date - timedelta(days=300)
    current_date = current_date.strftime("%Y%m%d")
    days_300_ago = days_300_ago.strftime("%Y%m%d")

    df_na = get_data(symbol='106.TTE', start_date=days_300_ago, end_date=current_date, model_num=model_num)
    scaler = MinMaxScaler()
    
    y = df_na.values[-10:,0].tolist()
    print(y)
    x_values = scaler.fit_transform(df_na.values)
    x_values = x_values[:,1:]

    x_values = create_time_series_data(x_values , TIMESTEP)

    x_values = np.array([x_values[-1]])
    
    
    GRU_model_path = "gru_model/gru_"+model_num+".keras"
    Attention_model_path = "attention_model/attention_"+model_num+".keras"
    
    attention_model = tf.keras.models.load_model(Attention_model_path)
    gru_model = tf.keras.models.load_model(GRU_model_path)
    
    attention_prediction = attention_model.predict(x_values)
    gru_prediction = gru_model.predict(x_values)
    
    dummy = np.zeros((len(gru_prediction), df_na.shape[-1]))
    dummy[:, 0] = attention_prediction[:, 0]
    attention_prediction = scaler.inverse_transform(dummy)[:, 0]
    dummy[:, 0] = gru_prediction[:, 0]
    gru_prediction = scaler.inverse_transform(dummy)[:, 0]
        
    y = y+[(gru_prediction[0]+attention_prediction[0])/2]
    # 返回结果
    return jsonify({
        "Attention Prediction": attention_prediction[0],
        "GRU Prediction": gru_prediction[0],
        "Used Features": options,
        "price_data": y
    })




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567)
    