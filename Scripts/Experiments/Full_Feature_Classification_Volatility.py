import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
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

df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek

for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)

df["price_rolling_std"] = df[price_col].rolling(window=4).std()

# IMPORTANT: Create returns BEFORE dropna
df["simple_return"] = df[price_col].pct_change()

df = df.dropna().copy()

n = len(df)

test_start = int(n * 0.80)
val_start  = int(n * 0.70)

df_train = df.iloc[:val_start].copy()
df_val   = df.iloc[val_start:test_start].copy()
df_test  = df.iloc[test_start:].copy()

vol_threshold = df_train["simple_return"].abs().quantile(0.99)

for subset in [df_train, df_val, df_test]:
    subset["spike_current"] = (
        subset["simple_return"].abs() > vol_threshold
    ).astype(int)

    subset["spike_next"] = subset["spike_current"].shift(-1)

df_train = df_train.dropna(subset=["spike_next"]).copy()
df_val   = df_val.dropna(subset=["spike_next"]).copy()
df_test  = df_test.dropna(subset=["spike_next"]).copy()

df_train["spike_next"] = df_train["spike_next"].astype(int)
df_val["spike_next"]   = df_val["spike_next"].astype(int)
df_test["spike_next"]  = df_test["spike_next"].astype(int)

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

time_features = ["hour", "day_of_week"]

full_features = market_features + micro_features + time_features

X_train_full = df_train[full_features]
y_train_full = df_train["spike_next"]

X_val = df_val[full_features]
y_val = df_val["spike_next"]

X_test = df_test[full_features]
y_test = df_test["spike_next"]

print("Train spike rate:", y_train_full.mean().round(4))
print("Val spike rate:", y_val.mean().round(4))
print("Test spike rate:", y_test.mean().round(4))

pos_idx = y_train_full[y_train_full == 1].index
neg_idx = y_train_full[y_train_full == 0].index

n_pos = len(pos_idx)
neg_target_ratio = 5
n_neg_sample = min(len(neg_idx), neg_target_ratio * n_pos)

neg_sample_idx = np.random.RandomState(42).choice(
    neg_idx, size=n_neg_sample, replace=False
)

train_idx_balanced = np.concatenate([pos_idx, neg_sample_idx])
train_idx_balanced = np.sort(train_idx_balanced)

X_train = X_train_full.loc[train_idx_balanced]
y_train = y_train_full.loc[train_idx_balanced]

print("Balanced train spike rate:", y_train.mean().round(4))

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'seed': 42,
}

evals = [(dtrain, 'train'), (dval, 'val')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=False,
)

print("Best iteration:", model.best_iteration)

val_proba = model.predict(xgb.DMatrix(X_val))

precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

model.save_model("vol_spike_full_xgb.json")

with open("vol_spike_full_features.json", "w") as f:
    json.dump(full_features, f)

np.save("vol_spike_full_return_threshold.npy", vol_threshold)
np.save("vol_spike_full_decision_threshold.npy", best_thresh)

print("\nBest threshold (val):", round(float(best_thresh), 4))
print("Val F1:", round(float(f1_scores[best_idx]), 4))

test_proba = model.predict(xgb.DMatrix(X_test))
test_pred = (test_proba > best_thresh).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec  = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)

print("\n=== Layer C Tuned: Full Feature Volatility Spike Classification ===")
print("Accuracy:", round(acc, 4))
print("F1 Score:", round(f1, 4))
print("AUC:", round(auc, 4))
print("PR-AUC:", round(pr_auc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred, digits=4))

# ==========================================================
# ABLATION ANALYSIS (Volatility Spike Classification)
# ==========================================================

print("\n================ ABLATION ANALYSIS (VOL SPIKES) ================")

# Frequency microstructure:
level_features = [
    "slope",
    "dev_mean",
    "dev_min",
    "dev_max",
    "var",
    "skewness",
    "kurtosis",
    "entropy",
]

excursion_features = [
    "mild_excursions",
    "deep_excursions",
]

rocof_features = [
    "max_abs_rocof",
    "mean_abs_rocof",
    "rocof_std",
    "rocof_shock_count",
]

shock_recovery_features = [
    "shock_depth",
    "recovery_time",
    "post_shock_var",
]

all_micro = (
    level_features +
    excursion_features +
    rocof_features +
    shock_recovery_features
)

market_features = [
    price_col,
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "price_lag4",
    "price_rolling_std",
    control_col,
]

ablation_sets = {
    "ALL_FEATURES": market_features + all_micro + time_features,

    "NO_MICRO": market_features + time_features,

    "NO_LEVEL": market_features
                + excursion_features
                + rocof_features
                + shock_recovery_features
                + time_features,

    "NO_EXCURSIONS": market_features
                     + level_features
                     + rocof_features
                     + shock_recovery_features
                     + time_features,

    "NO_ROCOF": market_features
                + level_features
                + excursion_features
                + shock_recovery_features
                + time_features,

    "NO_SHOCK_RECOVERY": market_features
                         + level_features
                         + excursion_features
                         + rocof_features
                         + time_features,

    "NO_TIME": market_features + all_micro,

    "MARKET_ONLY": market_features,

    "MICRO_ONLY": all_micro + time_features,
}

results_ablation = []

for name, feat_list in ablation_sets.items():

    X_train_full_ab = df_train[feat_list]
    X_val_ab = df_val[feat_list]
    X_test_ab = df_test[feat_list]

    X_train_ab = X_train_full_ab.loc[train_idx_balanced]
    y_train_ab = y_train_full.loc[train_idx_balanced]

    dtrain_ab = xgb.DMatrix(X_train_ab, label=y_train_ab)
    dval_ab   = xgb.DMatrix(X_val_ab, label=y_val)

    model_ab = xgb.train(
        params,
        dtrain_ab,
        num_boost_round=2000,
        evals=[(dtrain_ab, 'train'), (dval_ab, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # Threshold tuning on validation
    val_proba_ab = model_ab.predict(xgb.DMatrix(X_val_ab))
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba_ab)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh_ab = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    test_proba_ab = model_ab.predict(xgb.DMatrix(X_test_ab))
    test_pred_ab = (test_proba_ab > best_thresh_ab).astype(int)

    acc = accuracy_score(y_test, test_pred_ab)
    f1  = f1_score(y_test, test_pred_ab)
    auc = roc_auc_score(y_test, test_proba_ab)
    pr_auc = average_precision_score(y_test, test_proba_ab)

    results_ablation.append((name, len(feat_list), acc, f1, auc, pr_auc))

df_ablation = pd.DataFrame(
    results_ablation,
    columns=["Config", "NumFeatures", "Accuracy", "F1", "AUC", "PR_AUC"]
)

baseline_auc = df_ablation.loc[df_ablation["Config"]=="ALL_FEATURES","AUC"].values[0]
baseline_pr  = df_ablation.loc[df_ablation["Config"]=="ALL_FEATURES","PR_AUC"].values[0]

df_ablation["Delta_AUC"] = df_ablation["AUC"] - baseline_auc
df_ablation["Delta_PR_AUC"] = df_ablation["PR_AUC"] - baseline_pr

print("\n=== Volatility Ablation Summary ===")
print(df_ablation.sort_values("Delta_AUC"))

# ==========================================================
# TEMPORAL GENERALISATION (Volatility Spike Classification)
# ==========================================================

print("\n================ TEMPORAL GENERALISATION (VOL SPIKES) ================")

df_temp = df.copy()
df_temp["year"] = df_temp.index.year

scenarios = [
    ([2012], 2013),
    ([2012], [2013, 2014]),
    ([2012], [2013, 2014, 2015]),
    ([2012], [2013, 2014, 2015, 2016]),
    ([2012, 2013], 2014),
    ([2012, 2013], [2014, 2015]),
    ([2012, 2013], [2014, 2015, 2016]),
    ([2012, 2013, 2014], 2015),
    ([2012, 2013, 2014], [2015, 2016]),
    ([2012, 2013, 2014, 2015], 2016),
]

results_temporal = []

for train_years, test_years in scenarios:

    train_mask = df_temp["year"].isin(train_years)

    if isinstance(test_years, (list, tuple)):
        test_mask = df_temp["year"].isin(test_years)
        test_label = ",".join(str(y_) for y_ in test_years)
    else:
        test_mask = df_temp["year"] == test_years
        test_label = str(test_years)

    df_train_temp = df_temp.loc[train_mask].copy()
    df_test_temp  = df_temp.loc[test_mask].copy()

    # --- Define threshold from training only ---
    vol_threshold_temp = df_train_temp["simple_return"].abs().quantile(0.99)

    df_train_temp["spike_current"] = (
        df_train_temp["simple_return"].abs() > vol_threshold_temp
    ).astype(int)

    df_test_temp["spike_current"] = (
        df_test_temp["simple_return"].abs() > vol_threshold_temp
    ).astype(int)

    df_train_temp["spike_next"] = df_train_temp["spike_current"].shift(-1)
    df_test_temp["spike_next"]  = df_test_temp["spike_current"].shift(-1)

    df_train_temp = df_train_temp.dropna(subset=["spike_next"])
    df_test_temp  = df_test_temp.dropna(subset=["spike_next"])

    df_train_temp["spike_next"] = df_train_temp["spike_next"].astype(int)
    df_test_temp["spike_next"]  = df_test_temp["spike_next"].astype(int)

    X_train_temp = df_train_temp[full_features]
    y_train_temp = df_train_temp["spike_next"]

    X_test_temp = df_test_temp[full_features]
    y_test_temp = df_test_temp["spike_next"]

    # Balance training
    pos_idx = y_train_temp[y_train_temp == 1].index
    neg_idx = y_train_temp[y_train_temp == 0].index

    n_pos = len(pos_idx)
    n_neg_sample = min(len(neg_idx), 5 * n_pos)

    neg_sample_idx = np.random.RandomState(42).choice(
        neg_idx, size=n_neg_sample, replace=False
    )

    train_idx_bal = np.sort(np.concatenate([pos_idx, neg_sample_idx]))

    X_train_bal = X_train_temp.loc[train_idx_bal]
    y_train_bal = y_train_temp.loc[train_idx_bal]

    dtrain_temp = xgb.DMatrix(X_train_bal, label=y_train_bal)

    model_temp = xgb.train(
        params,
        dtrain_temp,
        num_boost_round=2000,
        verbose_eval=False,
    )

    test_proba_temp = model_temp.predict(xgb.DMatrix(X_test_temp))
    test_pred_temp = (test_proba_temp > 0.5).astype(int)

    acc = accuracy_score(y_test_temp, test_pred_temp)
    f1  = f1_score(y_test_temp, test_pred_temp)
    auc = roc_auc_score(y_test_temp, test_proba_temp)
    pr_auc = average_precision_score(y_test_temp, test_proba_temp)

    results_temporal.append((train_years, test_label, acc, f1, auc, pr_auc))

df_temporal = pd.DataFrame(
    results_temporal,
    columns=["TrainYears", "TestYears", "Accuracy", "F1", "AUC", "PR_AUC"]
)

print("\n=== Temporal Volatility Generalisation Summary ===")
print(df_temporal)
