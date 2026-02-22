import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from experiment_common import (
    ALL_MICRO_FEATURES,
    CONTROL_COL,
    EXCURSION_FEATURES,
    LEVEL_FEATURES,
    MICRO_FEATURES,
    PRICE_COL,
    ROCOF_FEATURES,
    SHOCK_RECOVERY_FEATURES,
    keep_contiguous_rows,
    load_aggregated_data,
    market_return_features,
    save_feature_list,
)

df = load_aggregated_data()

df["simple_return"] = df[PRICE_COL].pct_change()
df["simple_return"] = df["simple_return"].replace([np.inf, -np.inf], np.nan)

# Lag return features
for lag in [1, 2, 3, 4]:
    df[f"return_lag{lag}"] = df["simple_return"].shift(lag)

# Rolling return features
df["return_roll_std"] = df["simple_return"].rolling(window=4).std()
df["return_roll_mean"] = df["simple_return"].rolling(window=4).mean()

# Market + micro features (same micro features as before)
micro_features = MICRO_FEATURES
market_features = market_return_features(control_col=CONTROL_COL)

full_features = market_features + micro_features

df["y_next"] = df["simple_return"].shift(-1)
df = keep_contiguous_rows(df, prev_steps=4, next_steps=1)

df = df.dropna(subset=full_features + ["y_next"]).copy()

X_full = df[full_features]
y_full = df["y_next"]

split_idx = int(len(df) * 0.8)

X_train_full = X_full.iloc[:split_idx]
y_train_full = y_full.iloc[:split_idx]
X_test_full  = X_full.iloc[split_idx:]
y_test_full  = y_full.iloc[split_idx:]

print("Train size:", X_train_full.shape, "Test size:", X_test_full.shape)

USE_CLIPPING = False

if USE_CLIPPING:
    thresholds = [None, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
else:
    thresholds = [None]

results = []

for THRESH in thresholds:
    if THRESH is None:
        y_train = y_train_full.copy()
        y_test = y_test_full.copy()
        label = "None"
    else:
        y_train = y_train_full.clip(-THRESH, THRESH)
        y_test = y_test_full.clip(-THRESH, THRESH)
        label = str(THRESH)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_full, y_train)

    y_pred = model.predict(X_test_full)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append((label, mae, rmse))

print("\n=== Return Regression Threshold Search ===")
print("Threshold\tMAE\t\tRMSE")
for label, mae, rmse in results:
    print(f"{label:8s}\t{mae:.6f}\t{rmse:.6f}")

best_label, best_mae, best_rmse = min(results, key=lambda x: x[2])

print("\nBest by RMSE:")
print(f"Threshold: {best_label}")
print(f"MAE      : {best_mae:.6f}")
print(f"RMSE     : {best_rmse:.6f}")

if best_label == "None":
    y_train_best = y_train_full
    y_test_best = y_test_full
else:
    best_thresh = float(best_label)
    y_train_best = y_train_full.clip(-best_thresh, best_thresh)
    y_test_best = y_test_full.clip(-best_thresh, best_thresh)

final_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

final_model.fit(X_train_full, y_train_best)
final_model.save_model("return_regression_full_xgb.json")

save_feature_list("return_regression_full_features.json", full_features)
    
y_pred_best = final_model.predict(X_test_full)

final_mae = mean_absolute_error(y_test_best, y_pred_best)
final_rmse = np.sqrt(mean_squared_error(y_test_best, y_pred_best))

print("\n=== Final Return Regression Model ===")
print("MAE :", round(final_mae, 6))
print("RMSE:", round(final_rmse, 6))

# ==========================================================
# ABLATION ANALYSIS (Return Regression)
# ==========================================================

print("\n================ ABLATION ANALYSIS (RETURN REGRESSION) ================")

# Frequency microstructure:
level_features = LEVEL_FEATURES
excursion_features = EXCURSION_FEATURES
rocof_features = ROCOF_FEATURES
shock_recovery_features = SHOCK_RECOVERY_FEATURES
all_micro = ALL_MICRO_FEATURES
market_features = market_return_features(control_col=CONTROL_COL)

ablation_sets = {
    "ALL_FEATURES": market_features + all_micro,
    "NO_MICRO": market_features,
    "NO_LEVEL": market_features + excursion_features + rocof_features + shock_recovery_features,
    "NO_EXCURSIONS": market_features + level_features + rocof_features + shock_recovery_features,
    "NO_ROCOF": market_features + level_features + excursion_features + shock_recovery_features,
    "NO_SHOCK_RECOVERY": market_features + level_features + excursion_features + rocof_features,
    "MARKET_ONLY": market_features,
    "MICRO_ONLY": all_micro,
}

results_ablation = []

for name, feat_list in ablation_sets.items():

    X_train_ab = df.iloc[:split_idx][feat_list]
    X_test_ab  = df.iloc[split_idx:][feat_list]

    if best_label == "None":
        y_train_ab = y_train_full
        y_test_ab  = y_test_full
    else:
        best_thresh = float(best_label)
        y_train_ab = y_train_full.clip(-best_thresh, best_thresh)
        y_test_ab  = y_test_full.clip(-best_thresh, best_thresh)

    model_ab = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model_ab.fit(X_train_ab, y_train_ab)
    y_pred_ab = model_ab.predict(X_test_ab)

    mae = mean_absolute_error(y_test_ab, y_pred_ab)
    rmse = np.sqrt(mean_squared_error(y_test_ab, y_pred_ab))

    results_ablation.append((name, len(feat_list), mae, rmse))

df_ablation = pd.DataFrame(
    results_ablation,
    columns=["Config", "NumFeatures", "MAE", "RMSE"]
)

base_mae = df_ablation.loc[df_ablation["Config"]=="ALL_FEATURES","MAE"].values[0]
df_ablation["Delta_MAE"] = df_ablation["MAE"] - base_mae

print("\n=== Return Regression Ablation Summary (sorted by Delta_MAE) ===")
print(df_ablation.sort_values("Delta_MAE"))

# ==========================================================
# TEMPORAL GENERALISATION (Return Regression)
# ==========================================================

print("\n================ TEMPORAL GENERALISATION (RETURN REGRESSION) ================")

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

    if isinstance(test_years, (list, tuple, np.ndarray)):
        test_mask = df_temp["year"].isin(test_years)
        test_label = ",".join(str(y_) for y_ in test_years)
    else:
        test_mask = df_temp["year"] == test_years
        test_label = str(test_years)

    train_df_t = df_temp.loc[train_mask].copy()
    test_df_t  = df_temp.loc[test_mask].copy()

    X_train_t = train_df_t[full_features]
    X_test_t  = test_df_t[full_features]

    y_train_t = train_df_t["y_next"]
    y_test_t  = test_df_t["y_next"]

    if best_label != "None":
        best_thresh = float(best_label)
        y_train_t = y_train_t.clip(-best_thresh, best_thresh)
        y_test_t  = y_test_t.clip(-best_thresh, best_thresh)

    model_t = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model_t.fit(X_train_t, y_train_t)
    y_pred_t = model_t.predict(X_test_t)

    mae = mean_absolute_error(y_test_t, y_pred_t)
    rmse = np.sqrt(mean_squared_error(y_test_t, y_pred_t))

    results_temporal.append((train_years, test_label, mae, rmse))

df_temporal = pd.DataFrame(
    results_temporal,
    columns=["TrainYears", "TestYears", "MAE", "RMSE"]
)

print("\n=== Temporal Generalisation Summary (Return Regression) ===")
print(df_temporal)
