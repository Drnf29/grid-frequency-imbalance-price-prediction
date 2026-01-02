import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
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

for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)

df["price_rolling_std"] = df[price_col].rolling(window=4).std()

spike_threshold = df[price_col].quantile(0.90)
df["spike"] = (df[price_col] > spike_threshold).astype(int)
df["spike_next"] = df["spike"].shift(-1)

# Drop NaNs from lags, rolling, and shifted label
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

print("Overall spike rate:", y.mean().round(4))

n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Train size:", X_train.shape, "spike rate:", y_train.mean().round(4))
print("Val size:", X_val.shape, "spike rate:", y_val.mean().round(4))
print("Test size:", X_test.shape, "spike rate:", y_test.mean().round(4))

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print("scale_pos_weight (train):", round(scale_pos_weight, 2))

model_cls = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

model_cls.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_proba = model_cls.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print("\nBest threshold on validation:")
print("Threshold:", round(float(best_thresh), 3))
print("Precision at best F1:", round(float(precisions[best_idx]), 4))
print("Recall at best F1:", round(float(recalls[best_idx]), 4))
print("F1 (val):", round(float(f1_scores[best_idx]), 4))

test_proba = model_cls.predict_proba(X_test)[:, 1]
test_pred = (test_proba > best_thresh).astype(int)

acc = accuracy_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
try:
    auc = roc_auc_score(y_test, test_proba)
except ValueError:
    auc = float("nan")
pr_auc = average_precision_score(y_test, test_proba)

print("\n=== Layer C: Market + Micro-signal Spike Classification (Improved) ===")
print("Accuracy:", round(acc, 4))
print("F1 Score:", round(f1, 4))
print("AUC:", round(auc, 4))
print("PR-AUC:", round(pr_auc, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred, digits=4))
