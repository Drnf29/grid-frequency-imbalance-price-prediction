import pandas as pd
import numpy as np
import xgboost as xgb
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
from experiment_common import (
    ALL_MICRO_FEATURES,
    CONTROL_COL,
    EXCURSION_FEATURES,
    LEVEL_FEATURES,
    MICRO_FEATURES,
    PRICE_COL,
    ROCOF_FEATURES,
    SHOCK_RECOVERY_FEATURES,
    TIME_FEATURES,
    add_hour_and_day,
    add_price_lags,
    add_price_rolling_std,
    balanced_binary_index,
    keep_contiguous_rows,
    load_aggregated_data,
    market_price_features,
    save_feature_list,
    split_train_val_test,
)

df = load_aggregated_data()

add_hour_and_day(df, hour_col="hour", day_col="day_of_week")
add_price_lags(df, price_col=PRICE_COL)
add_price_rolling_std(df, price_col=PRICE_COL)

# IMPORTANT: Create returns BEFORE dropna
df["simple_return"] = df[PRICE_COL].pct_change()
df = keep_contiguous_rows(df, prev_steps=4, next_steps=1)

df = df.dropna().copy()

df_train, df_val, df_test = split_train_val_test(
    df,
    train_fraction=0.7,
    test_start_fraction=0.8,
)

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

market_features = market_price_features(price_col=PRICE_COL, control_col=CONTROL_COL)
micro_features = MICRO_FEATURES
time_features = TIME_FEATURES
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

train_idx_balanced = balanced_binary_index(
    y_train_full,
    neg_to_pos_ratio=5,
    random_state=42,
)

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

save_feature_list("vol_spike_full_features.json", full_features)

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
level_features = LEVEL_FEATURES
excursion_features = EXCURSION_FEATURES
rocof_features = ROCOF_FEATURES
shock_recovery_features = SHOCK_RECOVERY_FEATURES
all_micro = ALL_MICRO_FEATURES
market_features = market_price_features(price_col=PRICE_COL, control_col=CONTROL_COL)

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
    train_idx_bal = balanced_binary_index(
        y_train_temp,
        neg_to_pos_ratio=5,
        random_state=42,
    )

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
