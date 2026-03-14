import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
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

# next price (THIS is what market-only script uses)
df["next_price"] = df[PRICE_COL].shift(-1)
df = keep_contiguous_rows(df, prev_steps=4, next_steps=1)

# drop NaNs from lags/rolling/next_price
df = df.dropna().copy()

market_features = market_price_features(price_col=PRICE_COL, control_col=CONTROL_COL)
micro_features = MICRO_FEATURES
time_features = TIME_FEATURES
full_features = market_features + micro_features + time_features

df_train, df_val, df_test = split_train_val_test(
    df,
    train_fraction=0.7,
    test_start_fraction=0.8,
)

spike_threshold = df_train["next_price"].quantile(0.90)

for subset in (df_train, df_val, df_test):
    subset["spike_next"] = (subset["next_price"] > spike_threshold).astype(int)

print("Train spike rate:", df_train["spike_next"].mean().round(4))
print("Val spike rate  :", df_val["spike_next"].mean().round(4))
print("Test spike rate :", df_test["spike_next"].mean().round(4))

X_train_full = df_train[full_features]
y_train_full = df_train["spike_next"]

X_val = df_val[full_features]
y_val = df_val["spike_next"]

X_test = df_test[full_features]
y_test = df_test["spike_next"]

train_idx_balanced = balanced_binary_index(
    y_train_full,
    neg_to_pos_ratio=5,
    random_state=42,
)

X_train = X_train_full.loc[train_idx_balanced]
y_train = y_train_full.loc[train_idx_balanced]

print("Balanced train spike rate:", y_train.mean().round(4))

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "verbosity": 0,
    "seed": 42,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=100,
    verbose_eval=False,
)

print("Best iteration:", model.best_iteration)

val_proba = model.predict(xgb.DMatrix(X_val))
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print("Best decision threshold (val):", round(float(best_thresh), 4))

model.save_model("price_spike_full_xgb.json")
save_feature_list("price_spike_full_features.json", full_features)
np.save("price_spike_full_price_threshold.npy", spike_threshold)
np.save("price_spike_full_decision_threshold.npy", best_thresh)

test_proba = model.predict(xgb.DMatrix(X_test))
test_pred = (test_proba > best_thresh).astype(int)

acc = accuracy_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)

print("\n=== Full Feature Price Spike Classification (consistent definition) ===")
print("Accuracy:", round(acc, 4))
print("F1      :", round(f1, 4))
print("AUC     :", round(auc, 4))
print("PR-AUC  :", round(pr_auc, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred, digits=4))

print("\n=== ABLATION ANALYSIS ===")

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

    # --- Build matrices ---
    X_train_full_ab = df_train[feat_list]
    X_val_ab = df_val[feat_list]
    X_test_ab = df_test[feat_list]

    # --- Apply SAME balancing logic ---
    X_train_ab = X_train_full_ab.loc[train_idx_balanced]
    y_train_ab = y_train_full.loc[train_idx_balanced]

    dtrain_ab = xgb.DMatrix(X_train_ab, label=y_train_ab)
    dval_ab   = xgb.DMatrix(X_val_ab, label=y_val)

    model_ab = xgb.train(
        params,
        dtrain_ab,
        num_boost_round=2000,
        evals=[(dtrain_ab, "train"), (dval_ab, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # --- Threshold tuning on validation ---
    val_proba_ab = model_ab.predict(xgb.DMatrix(X_val_ab))
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba_ab)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh_ab = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # --- Test evaluation ---
    test_proba_ab = model_ab.predict(xgb.DMatrix(X_test_ab))
    test_pred_ab = (test_proba_ab > best_thresh_ab).astype(int)

    acc = accuracy_score(y_test, test_pred_ab)
    f1 = f1_score(y_test, test_pred_ab)
    auc = roc_auc_score(y_test, test_proba_ab)
    pr_auc = average_precision_score(y_test, test_proba_ab)

    results_ablation.append((name, len(feat_list), acc, f1, auc, pr_auc))

# ---- Summary ----
df_ablation = pd.DataFrame(
    results_ablation,
    columns=["Config", "NumFeatures", "Accuracy", "F1", "AUC", "PR_AUC"]
)

baseline_auc = df_ablation.loc[df_ablation["Config"]=="ALL_FEATURES","AUC"].values[0]
baseline_f1  = df_ablation.loc[df_ablation["Config"]=="ALL_FEATURES","F1"].values[0]

df_ablation["Delta_AUC"] = df_ablation["AUC"] - baseline_auc
df_ablation["Delta_F1"]  = df_ablation["F1"] - baseline_f1

print("\n=== Ablation Summary (Sorted by ΔAUC) ===")
print(df_ablation.sort_values("Delta_AUC"))

print("\n=== TEMPORAL GENERALISATION ===")

df["year"] = df.index.year
years = df["year"]

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

    train_mask = years.isin(train_years)

    if isinstance(test_years, (list, tuple, np.ndarray, pd.Series)):
        test_mask = years.isin(test_years)
        test_label = ",".join(str(y_) for y_ in test_years)
    else:
        test_mask = years == test_years
        test_label = str(test_years)

    df_train_temp = df.loc[train_mask].copy()
    df_test_temp  = df.loc[test_mask].copy()

    # --- Define spike threshold ONLY from training period ---
    spike_threshold_temp = df_train_temp["next_price"].quantile(0.90)

    df_train_temp["spike_next"] = (df_train_temp["next_price"] > spike_threshold_temp).astype(int)
    df_test_temp["spike_next"]  = (df_test_temp["next_price"] > spike_threshold_temp).astype(int)

    X_train_temp = df_train_temp[full_features]
    y_train_temp = df_train_temp["spike_next"]

    X_test_temp = df_test_temp[full_features]
    y_test_temp = df_test_temp["spike_next"]

    # --- Balance training set ---
    train_idx_balanced_temp = balanced_binary_index(
        y_train_temp,
        neg_to_pos_ratio=5,
        random_state=42,
    )

    X_train_bal = X_train_temp.loc[train_idx_balanced_temp]
    y_train_bal = y_train_temp.loc[train_idx_balanced_temp]

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
    f1 = f1_score(y_test_temp, test_pred_temp)
    auc = roc_auc_score(y_test_temp, test_proba_temp)
    pr_auc = average_precision_score(y_test_temp, test_proba_temp)

    results_temporal.append((train_years, test_label, acc, f1, auc, pr_auc))

# ---- Summary ----
df_temporal = pd.DataFrame(
    results_temporal,
    columns=["TrainYears", "TestYears", "Accuracy", "F1", "AUC", "PR_AUC"]
)

print("\n=== Temporal Generalisation Summary ===")
print(df_temporal)
