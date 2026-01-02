import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

features = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0,
    parse_dates=True,
)

df = features.copy()
df.index = pd.to_datetime(df.index)
df = df.sort_index()

price_col = 'Price in €/MWh'

df_market = df[[price_col, 'Controlled output requirements in MW']].copy()

for i in [1, 2, 3, 4]:
    df_market[f'price_lag{i}'] = df_market[price_col].shift(i)

df_market['price_roll_mean_4'] = df_market[price_col].rolling(4).mean()
df_market['price_roll_std_4']  = df_market[price_col].rolling(4).std()

df_market['price_change'] = df_market[price_col].diff()

df_market['MW_lag1']  = df_market['Controlled output requirements in MW'].shift(1)
df_market['MW_delta'] = (
    df_market['Controlled output requirements in MW']
    - df_market['MW_lag1']
)

df_market['hour'] = df.index.hour
df_market['dayofweek'] = df.index.dayofweek
df_market['month'] = df.index.month

df_market['simple_return'] = df_market[price_col].pct_change()
df_market = df_market.dropna().copy()

split_idx = int(len(df_market) * 0.8)

train = df_market.iloc[:split_idx].copy()
test  = df_market.iloc[split_idx:].copy()

vol_threshold = train['simple_return'].abs().quantile(0.99)

train['spike_current'] = (
    train['simple_return'].abs() > vol_threshold
).astype(int)

test['spike_current'] = (
    test['simple_return'].abs() > vol_threshold
).astype(int)

train['spike_next'] = train['spike_current'].shift(-1)
test['spike_next']  = test['spike_current'].shift(-1)

train = train.dropna(subset=['spike_next']).copy()
test  = test.dropna(subset=['spike_next']).copy()

train['spike_next'] = train['spike_next'].astype(int)
test['spike_next']  = test['spike_next'].astype(int)

feature_cols = [
    col for col in train.columns
    if col not in [
        price_col,
        'simple_return',
        'spike_current',
        'spike_next'
    ]
]

X_train = train[feature_cols]
y_train = train['spike_next']

X_test = test[feature_cols]
y_test = test['spike_next']

clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

clf.fit(X_train, y_train)
clf.save_model("vol_spike_market_only_xgb.json")
with open("vol_spike_market_only_features.json", "w") as f:
    json.dump(feature_cols, f)
np.save("vol_spike_market_only_return_threshold.npy", vol_threshold)
decision_threshold = 0.5
np.save("vol_spike_market_only_decision_threshold.npy", decision_threshold)

y_pred_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
pr_auc = average_precision_score(y_test, y_pred_prob)

print("\n=== Layer B (Market-Only) Volatility Spike Classification ===")
print("Accuracy :", round(acc, 4))
print("PR-AUC   :", round(pr_auc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("F1       :", round(f1, 4))
