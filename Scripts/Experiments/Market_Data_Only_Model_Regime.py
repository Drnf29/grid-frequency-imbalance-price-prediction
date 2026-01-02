import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


df = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0,
    parse_dates=True,
)

df.index = pd.to_datetime(df.index)
df = df.sort_index()

price_col = "Price in €/MWh"
control_col = "Controlled output requirements in MW"

missing = [c for c in [price_col, control_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)

df["price_rolling_std"] = df[price_col].rolling(window=4).std()

market_features = [
    price_col,
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "price_lag4",
    "price_rolling_std",
    control_col,
]

df["future_vol"] = df[price_col].shift(-1).rolling(window=4).var()

df["future_vol"] = df[price_col].shift(-1).rolling(window=4).var()

df = df.dropna()
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

low_q = train_df["future_vol"].quantile(0.60)
high_q = train_df["future_vol"].quantile(0.90)

def vol_regime(v):
    if v <= low_q:
        return 0
    elif v <= high_q:
        return 1
    else:
        return 2

train_df["vol_regime"] = train_df["future_vol"].apply(vol_regime)
test_df["vol_regime"]  = test_df["future_vol"].apply(vol_regime)

X_train = train_df[market_features]
y_train = train_df["vol_regime"]

X_test = test_df[market_features]
y_test = test_df["vol_regime"]


print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
model.save_model("vol_regime_market_only_xgb.json")
with open("vol_regime_market_only_features.json", "w") as f:
    json.dump(market_features, f)
np.save("vol_regime_market_only_low_threshold.npy", low_q)
np.save("vol_regime_market_only_high_threshold.npy", high_q)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print("\n=== Market-Only Volatility Regime Classification ===")
print("Accuracy     :", round(acc, 4))
print("Macro F1     :", round(macro_f1, 4))
print("Weighted F1  :", round(weighted_f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDetailed Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low vol", "Medium vol", "High vol"],
        zero_division=0,
    )
)
