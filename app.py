from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

matplotlib.use('Agg')

app = Flask(__name__)

model = load_model('model.keras')

def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format = "png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock = stock, no_of_days = no_of_days))
    return render_template("index.html")


@app.route("/predict")
def predict():
    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    # Fetch stock data
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)
    if stock_data.empty:
        return render_template("result.html", error = "Invalid stock ticker od no data available")

    # Data Preparation
    splitting_len = int(len(stock_data) * 0.9)
    x_test = stock_data[['Close']][splitting_len:]
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100: i])
        y_data.append(scaled_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Prediction
    predictions = model.predict(x_data)
    inversed_prediction = scaler.inverse_transform(predictions)
    inversed_y_test = scaler.inverse_transform(y_data)

    # Prepare data for plotting
    plotting_data = pd.DataFrame(
    {
        'Original Test Data': inversed_y_test.flatten(),
        'Predictive Test Data': inversed_prediction.flatten(),
    }, index = x_test.index[100:]
    )

    # Generate Plots
    # Plot-1: Original Closing Prices
    fig1 = plt.figure(figsize = (15, 6), facecolor = ('lightblue'))
    plt.plot(stock_data['Close'], label = 'Close Price', color = 'cyan', linewidth = 2)
    plt.title("Close price of bitcoin overtime", fontsize = 16)
    plt.xlabel("Years", fontsize = 14)
    plt.ylabel("Close Price", fontsize = 14)
    plt.grid(alpha = 0.3, linewidth = 0.2)
    plt.legend(fontsize = 12)
    original_plot = plot_to_html(fig1)

    # Plot-2: Original vs predicted test data
    fig2 =plt.figure(figsize = (15, 6), facecolor = "cyan")
    plt.plot(plotting_data['Original Test Data'], label = "Actual Values", color = "violet", linewidth = 2)
    plt.plot(plotting_data['Predictive Test Data'], label = "Predictive Values", color = "yellow", linewidth = 2)
    plt.title("Actual Values V/S Predictive Values", fontsize = 15)
    plt.xlabel("Years", fontsize = 15)
    plt.ylabel("Close Price", fontsize = 15)
    plt.grid(alpha = 0.3)
    plt.legend(fontsize = 12)
    predicted_plot = plot_to_html(fig2)

    # Plot-3: Future Predictions
    last_100_days = stock_data[['Close']].tail(100)
    last_100_days_scaled = scaler.transform(last_100_days)
    future_predictions = []
    last_100_days_scaled = last_100_days_scaled.reshape(1, -1, 1)
    for _ in range(no_of_days):
        next_days = model.predict(last_100_days_scaled)
        future_predictions.append(scaler.inverse_transform(next_days))
        last_100_days_scaled = np.append(last_100_days_scaled[:, 1:, :], next_days.reshape(1, 1, -1), axis = 1)

    future_predictions = np.array(future_predictions).flatten()

    fig3 = plt.figure(figsize = (15, 6), facecolor = "lightgreen")
    plt.plot(range(1, no_of_days + 1), future_predictions, marker = "o", label = "Prediction for Future Price", color = "cyan", linewidth = 2)
    plt.title("Future Close Price For 10 Days", fontsize = 15)
    plt.xlabel("Ten Days Ahead", fontsize = 15)
    plt.ylabel("Close Price", fontsize = 15)
    plt.grid(alpha = 0.3)
    plt.legend(fontsize = 12)
    future_plot = plot_to_html(fig3)


    return render_template(
        "result.html",
        stock = stock,
        original_plot = original_plot,
        predicted_plot = predicted_plot,
        future_plot = future_plot,
        enumerate = enumerate,
        future_predictions = future_predictions
    )

if __name__ == "__main__":
    app.run(debug = True)