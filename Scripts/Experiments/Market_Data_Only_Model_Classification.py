import pandas as pd
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

features = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0,
    parse_dates=True,
)

df = features.copy()
df.index = pd.to_datetime(df.index)
df = df.sort_index()

df_market = df[['Price in €/MWh', 'Controlled output requirements in MW']].copy()

for i in [1, 2, 3, 4]:
    df_market[f'price_lag{i}'] = df_market['Price in €/MWh'].shift(i)

df_market['price_roll_mean_4'] = df_market['Price in €/MWh'].rolling(4).mean()
df_market['price_roll_std_4']  = df_market['Price in €/MWh'].rolling(4).std()

df_market['price_change'] = df_market['Price in €/MWh'].diff()

df_market['MW_lag1']  = df_market['Controlled output requirements in MW'].shift(1)
df_market['MW_delta'] = (
    df_market['Controlled output requirements in MW']
    - df_market['MW_lag1']
)

df_market['hour'] = df.index.hour
df_market['dayofweek'] = df.index.dayofweek
df_market['month'] = df.index.month
df_market = df_market.dropna()

df_market['next_price'] = df_market['Price in €/MWh'].shift(-1)
df_market = df_market.dropna()  # drop last row with NaN next_price

split_idx = int(len(df_market) * 0.8)
train = df_market.iloc[:split_idx].copy()
test  = df_market.iloc[split_idx:].copy()

spike_threshold = train['next_price'].quantile(0.9)

train['spike'] = (train['next_price'] > spike_threshold).astype(int)
test['spike']  = (test['next_price']  > spike_threshold).astype(int)

feature_cols = [
    col for col in df_market.columns
    if col not in ['Price in €/MWh', 'next_price', 'spike']
]

X_train = train[feature_cols]
y_train = train['spike']

X_test = test[feature_cols]
y_test = test['spike']

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
clf.save_model("price_spike_market_only_xgb.json")
with open("price_spike_market_only_features.json", "w") as f:
    json.dump(feature_cols, f)
np.save("price_spike_market_only_price_threshold.npy", spike_threshold)
decision_threshold = 0.5
np.save("price_spike_market_only_decision_threshold.npy", decision_threshold)

y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
pr_auc = average_precision_score(y_test, y_pred) 

print("\n=== Layer B (Market-Only) Spike Classification ===")
print("Accuracy :", round(acc, 4))
print("PR-AUC:", round(pr_auc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("F1       :", round(f1, 4))
