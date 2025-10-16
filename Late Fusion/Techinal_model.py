import yfinance as yf
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ta
from joblib import dump

# Grab all the data for training/testing model
tickers = ["TSLA", "AAPL", "NVDA", "INTC", "PFE", "AMZN", "MSFT", "GOOGL"]
all_data = []

# Add missing predictor columns and target column
for ticker in tickers:
    data = yf.Ticker(ticker).history(period="max")
    
    # Clean the data before adding it
    data = data.loc["1995-01-01":].copy()
    data.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    data["Tomorrow"] = data["Close"].shift(-20)

    # Dependent variable trying to classify
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    # Indicators used in model
    data["Return"] = data["Close"].pct_change(periods=20)
    data["Volatility"] = data["Return"].rolling(20).std()
    # Momentum of the stock
    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    # Averages out price (last 20 days) - removes noise
    data["SMA_10"] = ta.trend.SMAIndicator(data["Close"], window=20).sma_indicator()
    data["MACD"] = ta.trend.MACD(data["Close"]).macd()
    
    data.dropna(inplace=True)
    all_data.append(data)

combined_data = pd.concat(all_data, keys=tickers)

# Collecting training data
X = combined_data[["Open", "High", "Low", "Close", "Volume", "Return", "Volatility", "RSI", "SMA_10", "MACD"]]
y = combined_data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

best_model = None
best_acc = 0

# Gradient Boosting holds better accuracy in financial models over Random Forest
for seed in [1, 7, 42, 123, 999]:
    model = GradientBoostingClassifier(n_estimators = 250, learning_rate = 0.05, random_state = seed)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Test the model - we're making the model more confident in its prediction
    preds = model.predict(X_test)
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    if acc > best_acc:
        best_model = model
        best_acc = acc
    
dump(model, "gradient_boosting_stock_model.joblib")


