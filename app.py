from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from flask_cors import CORS
import json
import traceback
import plotly
import requests
from tensorflow.keras.models import load_model  # Updated to tf.keras
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://stockanalyzer-qtk1.onrender.com"]}})

# Load the model with error handling using tf.keras
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'tf.keras')

try:
    model = load_model(model_path)
    print("Model loaded successfully with tf.keras.")
except Exception as e:
    print(f"Error loading model: {e}")

cleaned_data_cache = {}

def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = pd.DataFrame(data)
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(3)
        return df
    except Exception as e:
        return str(e)

def clean_stock_data(data):
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def fetch_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        info = stock.info
        return {
            "name": info.get("longName", "N/A"),
            "open": round(data['Open'].iloc[0], 2) if not data.empty else "N/A",
            "close": round(data['Close'].iloc[0], 2) if not data.empty else "N/A",
            "low": round(data['Low'].iloc[0], 2) if not data.empty else "N/A",
            "high": round(data['High'].iloc[0], 2) if not data.empty else "N/A",
            "exchange": info.get("exchange", "N/A")
        }
    except Exception as e:
        return str(e)

def fetch_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get("longName", ticker)
        search_query = company_name.replace("&", "and")
        NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        url = f'https://newsapi.org/v2/everything?q={search_query}&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        news_data = response.json()
        
        if news_data.get("status") == "ok":
            return news_data.get("articles", [])[:3]
        else:
            return []
    except Exception as e:
        return str(e)

@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    stock = request.args.get('ticker')
    chart_period = request.args.get('chart_period', '1mo')
    table_period = request.args.get('table_period', '1mo')

    if not stock:
        return jsonify({"error": "Ticker parameter is required"}), 400

    try:
        # Fetch stock data for chart and table
        chart_data = fetch_stock_data(stock, chart_period)
        table_data = fetch_stock_data(stock, table_period)

        if isinstance(chart_data, str) or isinstance(table_data, str):
            return jsonify({"error": "Failed to fetch stock data"}), 500

        chart_data = clean_stock_data(chart_data)
        table_data = clean_stock_data(table_data)

        chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.strftime('%d-%m-%Y')
        table_data['Date'] = pd.to_datetime(table_data['Date']).dt.strftime('%d-%m-%Y')

        fig1 = px.line(chart_data, x='Date', y='Close', title=f'{stock} Stock Price Over Time')
        fig1.update_layout(width=1200, height=600)
        fig1.update_xaxes(autorange="reversed")
        graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

        fig2 = px.area(chart_data, x='Date', y='Close', title=f'{stock} Stock Price Bar Chart')
        fig2.update_layout(width=1200, height=600)
        fig2.update_xaxes(autorange="reversed")
        graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        stock_info = fetch_stock_info(stock)
        stock_news = fetch_stock_news(stock)

        return jsonify({
            "stock_data": table_data.to_dict(orient='records'),
            "graph_data1": graphJSON1,
            "graph_data2": graphJSON2,
            "stock_info": stock_info,
            "stock_news": stock_news,
            "chart_period": chart_period,
            "table_period": table_period
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from datetime import datetime, timedelta

@app.route('/api/stock/predict', methods=['GET', 'OPTIONS'])
def predict_stock():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        # Fetch historical stock data
        data = yf.download(ticker, period='5y')
        if data.empty:
            return jsonify({'error': 'No data found for the given ticker'}), 404

        # Preprocess data
        data['Date'] = data.index
        data['Date'] = data['Date'].map(datetime.toordinal)
        X = data['Date'].values.reshape(-1, 1)
        y = data['Close'].values
        actual_dates = data.index.strftime('%Y-%m-%d').tolist()  # Actual dates

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future prices
        future_dates = [datetime.now() + timedelta(days=i*365) for i in range(1, 11)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]
        predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))
        predicted_dates = [date.strftime('%Y-%m-%d') for date in future_dates]  # Predicted dates

        # Calculate returns
        current_price = y[-1]
        stocks = [10, 20, 50, 100]
        returns = []
        for stock in stocks:
            returns.append({
                'stocks_bought': stock,
                'current_price': round(current_price * stock, 2),
                'after_1_year': round(stock * predictions[0], 0),
                'after_5_years': round(stock * predictions[4], 0),
                'after_10_years': round(stock * predictions[9], 0)
            })

        return jsonify({
            'predictions': predictions.tolist(),
            'predicted_dates': predicted_dates,
            'actual': y.tolist(),
            'actual_dates': actual_dates,
            'returns': returns
        })
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Error during prediction: {error_message}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


@app.route('/api/stock/info', methods=['GET', 'OPTIONS'])
def get_stock_info():
    stock_ticker = request.args.get('ticker')
    chart_period = request.args.get('chart_period', '1mo')
    table_period = request.args.get('table_period', '1mo')

    if not stock_ticker:
        return jsonify({"error": "Ticker parameter is required"}), 400

    chart_cache_key = f"{stock_ticker}_{chart_period}_chart"
    if chart_cache_key in cleaned_data_cache:
        chart_data = cleaned_data_cache[chart_cache_key]
    else:
        chart_data = fetch_stock_data(stock_ticker, chart_period)
        if isinstance(chart_data, str):
            return jsonify({"error": chart_data}), 500
        chart_data = clean_stock_data(chart_data)
        chart_data = chart_data.sort_values(by='Date', ascending=False)
        cleaned_data_cache[chart_cache_key] = chart_data

    table_cache_key = f"{stock_ticker}_{table_period}_table"
    if table_cache_key in cleaned_data_cache:
        table_data = cleaned_data_cache[table_cache_key]
    else:
        table_data = fetch_stock_data(stock_ticker, table_period)
        if isinstance(table_data, str):
            return jsonify({"error": table_data}), 500
        table_data = clean_stock_data(table_data)
        table_data = table_data.sort_values(by='Date', ascending=False)
        cleaned_data_cache[table_cache_key] = table_data

    chart_data['Date'] = pd.to_datetime(chart_data['Date'])
    table_data['Date'] = pd.to_datetime(table_data['Date'])

    chart_data['Date'] = chart_data['Date'].dt.strftime('%d-%m-%Y')
    table_data['Date'] = table_data['Date'].dt.strftime('%d-%m-%Y')

    fig1 = px.line(chart_data, x='Date', y='Close', title=f'{stock_ticker} Stock Price Over Time')
    fig1.update_layout(width=1200, height=600)
    fig1.update_xaxes(autorange="reversed")
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.area(chart_data, x='Date', y='Close', title=f'{stock_ticker} Stock Price Bar Chart')
    fig2.update_layout(width=1200, height=600)
    fig2.update_xaxes(autorange="reversed")
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    stock_info = fetch_stock_info(stock_ticker)
    stock_news = fetch_stock_news(stock_ticker)

    return jsonify({
        "stock_data": table_data.to_dict(orient='records'),
        "graph_data1": graphJSON1,
        "graph_data2": graphJSON2,
        "stock_info": stock_info,
        "stock_news": stock_news,
        "chart_period": chart_period,
        "table_period": table_period
    })

@app.route('/api/stocks', methods=['GET'])
def get_stocks_by_exchange():
    exchange = request.args.get('exchange')
    if not exchange:
        return jsonify({"error": "Exchange parameter is required"}), 400

    exchange_stocks = {
        "BSE": ["RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "ICICIBANK.BO", "MARUTI.BO", "HINDALCO.BO", "ITC.BO", 
        "SBIN.BO", "AXISBANK.BO", "KOTAKBANK.BO", "NTPC.BO", "BAJAJ-AUTO.BO", "HAL.BO", "BHEL.BO", "ADANIPORTS.BO", "M&M.BO",
        "TATAMOTORS.BO", "BHARTIARTL.BO", "ULTRACEMCO.BO", "GAIL.BO", "ASIANPAINT.BO", "ONGC.BO", "SUNPHARMA.BO", "TATASTEEL.BO", 
        "WIPRO.BO", "DIVISLAB.BO", "TECHM.BO", "NESTLEIND.BO", "GRASIM.BO", "LUPIN.BO"],
        "NSE": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "MARUTI.NS", "HINDALCO.NS", "ITC.NS", "SBIN.NS", 
        "AXISBANK.NS", "KOTAKBANK.NS", "NTPC.NS", "BAJAJ-AUTO.NS", "HAL.NS", "BHEL.NS", "ADANIPORTS.NS", "M&M.NS", "TATAMOTORS.NS", 
        "BHARTIARTL.NS", "ULTRACEMCO.NS", "GAIL.NS", "ASIANPAINT.NS", "ONGC.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "WIPRO.NS", 
        "DIVISLAB.NS", "TECHM.NS", "NESTLEIND.NS", "GRASIM.NS", "LUPIN.NS"],
        "US": ["GOOG", "AAPL", "MSFT", "AMZN", "TSLA", "FB", "NFLX", "NVDA", "AMD", "PYPL", "DIS", "BA", "V", "JNJ", "WMT", 
        "MA", "INTC", "CVX", "GM", "IBM", "META", "CSCO", "AMGN", "NVDA", "ZM", "AMD", "PEP", "ADBE", "WFC", "KO", "BA", 
        "SQ", "GS", "UBER", "ABT", "MMM", "PFE", "BMY", "PYPL", "TXN", "INTU", "SPGI", "ISRG", "BIIB", "EXPE", "PINS", 
        "ATVI", "LMT", "RTX", "COP", "JNJ", "VZ", "PG", "MO", "XOM", "COST", "SYF", "MS", "GS", "FISV"],
        "Cryptocurrency": ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "DOT-USD", "MATIC-USD", 
        "BNB-USD", "AVAX-USD", "SHIB-USD", "LINK-USD", "VET-USD", "XLM-USD", "TRX-USD", "BCH-USD", "EOS-USD", 
        "MKR-USD", "AAVE-USD", "CRV-USD", "SUSHI-USD", "FTT-USD", "YFI-USD", "ZRX-USD", "LEND-USD", "FLOKI-USD"],
        "UK": ["BP.L", "HSBC.L", "BARC.L", "VOD.L", "RDSB.L", "LLOY.L", "GSK.L", "TSCO.L", "RIO.L", "AAL.L", "SHEL.L", 
        "BATS.L", "ULVR.L", "IMB.L", "DGE.L", "LSEG.L", "BT.L", "PERF.L", "CRH.L", "STAN.L", "CINE.L", "ITV.L", "AHT.L", "TW.L", "SMDS.L"]
    }

    if exchange not in exchange_stocks:
        return jsonify({"error": "Invalid exchange"}), 400

    tickers = exchange_stocks[exchange]
    stocks_data = []

    try:
        stock_data = yf.download(tickers, period="1d", group_by='ticker')
        for ticker in tickers:
            try:
                stock_info = yf.Ticker(ticker).info
                stocks_data.append({
                    "symbol": ticker,
                    "name": stock_info.get("longName", ticker),
                    "open": round(stock_data[ticker]["Open"].iloc[0], 2) if not stock_data[ticker]["Open"].isna().all() else "N/A",
                    "high": round(stock_data[ticker]["High"].iloc[0], 2) if not stock_data[ticker]["High"].isna().all() else "N/A",
                    "low": round(stock_data[ticker]["Low"].iloc[0], 2) if not stock_data[ticker]["Low"].isna().all() else "N/A",
                    "close": round(stock_data[ticker]["Close"].iloc[0], 2) if not stock_data[ticker]["Close"].isna().all() else "N/A",
                })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")

        return jsonify({"stocks": stocks_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)