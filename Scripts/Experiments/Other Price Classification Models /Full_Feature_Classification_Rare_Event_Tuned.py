import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    accuracy_score
)

df = pd.read_csv(
    "../../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0, parse_dates=True
).sort_index()

price_col = "Price in €/MWh"
control_col = "Controlled output requirements in MW"

# time feats
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek

# price lags
for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)
df["price_rolling_std"] = df[price_col].rolling(4).std()

micro_features = [
    "slope","dev_mean","dev_min","dev_max",
    "mild_excursions","deep_excursions",
    "var","skewness","kurtosis","entropy",
    "max_abs_rocof","mean_abs_rocof","rocof_std","rocof_shock_count",
    "shock_depth","recovery_time","post_shock_var",
]

# micro lags
for feat in micro_features:
    for lag in [1, 2, 3, 4]:
        df[f"{feat}_lag{lag}"] = df[feat].shift(lag)

time_features = ["hour", "day_of_week"]
market_features = [
    price_col, "price_lag1", "price_lag2", "price_lag3", "price_lag4",
    "price_rolling_std", control_col
]
full_features = (
    market_features
    + micro_features
    + [f"{f}_lag{l}" for f in micro_features for l in [1,2,3,4]]
    + time_features
)

# ---------- IMPORTANT: build splits BEFORE label, but compute threshold on train rows ----------
df = df.dropna(subset=full_features).copy()

n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

# spike threshold from TRAIN ONLY
spike_threshold = df.iloc[:train_end][price_col].quantile(0.90)

# label, then drop last row that loses spike_next
df["spike_current"] = (df[price_col] > spike_threshold).astype(int)
df["spike_next"] = df["spike_current"].shift(-1)
df = df.dropna(subset=["spike_next"]).copy()
df["spike_next"] = df["spike_next"].astype(int)

# Now recompute n and split points AFTER dropping last row (keeps alignment perfect)
n = len(df)
train_end = int(n * 0.7)
val_end   = int(n * 0.85)

X = df[full_features]
y = df["spike_next"]

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

# class weight (no downsampling)
n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
scale_pos_weight = float(n_neg / max(n_pos, 1))

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "eta": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "gamma": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

model = xgb.train(
    params,
    dtrain,
    num_boost_round=8000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=300,
    verbose_eval=False,
)

val_proba  = model.predict(dval)
test_proba = model.predict(dtest)

# choose threshold using F2 (recall-weighted)
prec, rec, thr = precision_recall_curve(y_val, val_proba)
beta = 2.0
f2 = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
best = np.nanargmax(f2[:-1]) if len(thr) > 0 else 0
best_thr = float(thr[best]) if len(thr) > 0 else 0.5

test_pred = (test_proba >= best_thr).astype(int)
acc = accuracy_score(y_test, test_pred)
print("Accuracy:", round(acc, 4))

print("Spike threshold (train-only):", float(spike_threshold))
print("scale_pos_weight:", scale_pos_weight)
print("Best iteration:", model.best_iteration)
print("Chosen threshold:", round(best_thr, 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_proba), 4))
print("PR-AUC (AP):", round(average_precision_score(y_test, test_proba), 4))
print("\nConfusion:\n", confusion_matrix(y_test, test_pred))
print("\nReport:\n", classification_report(y_test, test_pred, digits=4))
