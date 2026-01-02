import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)

df = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0,
    parse_dates=True,
)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

price_col = "Price in €/MWh"
control_col = "Controlled output requirements in MW"

# Lagged prices
for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)

# Simple rolling volatility (last 4 intervals ≈ 1 hour)
df["price_rolling_std"] = df[price_col].rolling(window=4).std()

# Spike threshold: top 10% of prices
spike_threshold = df[price_col].quantile(0.90)

# Current-interval spike
df["spike"] = (df[price_col] > spike_threshold).astype(int)

# Next-interval spike (this is the target)
df["spike_next"] = df["spike"].shift(-1)

df = df.dropna()

market_features = [
    price_col,
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "price_lag4",
    "price_rolling_std",
    control_col,
]

micro_features = [
    "slope",
    "dev_mean",
    "dev_min",
    "dev_max",
    "mild_excursions",
    "deep_excursions",
    "var",
    "skewness",
    "kurtosis",
    "entropy",
    "max_abs_rocof",
    "mean_abs_rocof",
    "rocof_std",
    "rocof_shock_count",
    "shock_depth",
    "recovery_time",
    "post_shock_var",
]

full_features = market_features + micro_features

X = df[full_features]
y = df["spike_next"].astype(int)

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Spike rate in full data:", y.mean().round(4))
print("Spike rate in train:", y_train.mean().round(4))
print("Spike rate in test:", y_test.mean().round(4))

model_cls = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1,
)

model_cls.fit(X_train, y_train)

y_proba = model_cls.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_proba)
except ValueError:
    auc = float("nan")
pr_auc = average_precision_score(y_test, y_proba)    

print("\n=== Layer C: Market + Micro-signal Spike Classification ===")
print("Accuracy:", round(acc, 4))
print("F1 Score:", round(f1, 4))
print("AUC:", round(auc, 4))
print("PR-AUC:", round(pr_auc, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
