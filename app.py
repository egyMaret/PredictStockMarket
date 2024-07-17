import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# Muat kembali model LSTM
model = load_model('model/model_lstm.h5')

# Muat kembali scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Muat kembali DataFrame
df = pd.read_pickle('model/dataframe.pkl')

@app.route("/")
def stock():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    date_predict_str = request.form.get("date_predict")  # Ubah dari "date-predict" menjadi "date_predict"
    predicted_price, predicted_date = predict_future_close(date_predict_str, model, scaler, df)
    return render_template("index.html", result_predict=predicted_price, date_predict=predicted_date)

def predict_future_close(date_to_predict, model, scaler, df, look_back=60):
    date_to_predict = datetime.strptime(date_to_predict, '%Y-%m-%d')
    last_date = df.index[-1]
    date_range = pd.date_range(start=last_date, end=date_to_predict, freq='B')
    last_close = df['Close'].values[-look_back:]
    last_close_scaled = scaler.transform(last_close.reshape(-1, 1))
    predictions = []

    for _ in date_range[1:]:
        input_data = np.array(last_close_scaled[-look_back:]).reshape(1, look_back, 1)
        predicted_close_scaled = model.predict(input_data)
        predicted_close = scaler.inverse_transform(predicted_close_scaled)
        predictions.append(predicted_close[0][0])
        last_close_scaled = np.append(last_close_scaled, predicted_close_scaled, axis=0)

    return predictions[-1], date_range[-1]

if __name__ == "__main__":
    app.run(debug=True)
