import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from experiment_common import (
    CONTROL_COL,
    PRICE_COL,
    add_future_variance_target,
    add_price_lags,
    add_price_rolling_std,
    keep_contiguous_rows,
    load_aggregated_data,
    market_price_features,
    save_feature_list,
    split_train_test,
)

df = load_aggregated_data()

missing = [c for c in [PRICE_COL, CONTROL_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

add_price_lags(df, price_col=PRICE_COL)
add_price_rolling_std(df, price_col=PRICE_COL)
market_features = market_price_features(price_col=PRICE_COL, control_col=CONTROL_COL)

add_future_variance_target(df, source_col=PRICE_COL, horizon=4, out_col="future_vol")
df = keep_contiguous_rows(df, prev_steps=4, next_steps=4)

df = df.dropna()
train_df, test_df = split_train_test(df, train_fraction=0.8)

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
save_feature_list("vol_regime_market_only_features.json", market_features)
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
