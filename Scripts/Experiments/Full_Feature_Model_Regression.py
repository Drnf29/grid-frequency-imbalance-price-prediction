import numpy as np
import pandas as pd
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
    add_price_lags,
    add_price_rolling_std,
    keep_contiguous_rows,
    load_aggregated_data,
    market_price_features,
    save_feature_list,
)

df = load_aggregated_data()

# Lagged prices (you can adjust lags)
add_price_lags(df, price_col=PRICE_COL)

# Simple rolling volatility (over last 4 intervals = 1 hour)
add_price_rolling_std(df, price_col=PRICE_COL)

df["y_next"] = df[PRICE_COL].shift(-1)
df = keep_contiguous_rows(df, prev_steps=4, next_steps=1)

# Drop rows with NaNs introduced by shifting/rolling
df = df.dropna()

# Market features (Layer B)
market_features = market_price_features(price_col=PRICE_COL, control_col=CONTROL_COL)

# Micro-signal features (from your frequency engineering)
micro_features = MICRO_FEATURES

full_features = market_features + micro_features

# Filter the tail ends of the data
X = df[full_features]
y = df["y_next"]

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print("Train size:", X_train.shape, "Test size:", X_test.shape)

model_C = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

model_C.fit(X_train, y_train)
model_C.save_model("price_regression_full_xgb.json")
save_feature_list("price_regression_full_features.json", full_features)

y_pred = model_C.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Layer C: Market + Micro-signal Regression ===")
print("MAE :", mae)
print("RMSE:", rmse)

# ==========================================================
# ABLATION ANALYSIS (Price Regression)
# ==========================================================

print("\n================ ABLATION ANALYSIS (PRICE REGRESSION) ================")

# Frequency microstructure:
level_features = LEVEL_FEATURES
excursion_features = EXCURSION_FEATURES
rocof_features = ROCOF_FEATURES
shock_recovery_features = SHOCK_RECOVERY_FEATURES
all_micro = ALL_MICRO_FEATURES
market_features = market_price_features(price_col=PRICE_COL, control_col=CONTROL_COL)

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
    y_train_ab = df.iloc[:split_idx]["y_next"]

    X_test_ab = df.iloc[split_idx:][feat_list]
    y_test_ab = df.iloc[split_idx:]["y_next"]

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

print("\n=== Price Regression Ablation Summary (sorted by Delta_MAE) ===")
print(df_ablation.sort_values("Delta_MAE"))

# ==========================================================
# TEMPORAL GENERALISATION (Price Regression)
# ==========================================================

print("\n================ TEMPORAL GENERALISATION (PRICE REGRESSION) ================")

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
    y_train_t = train_df_t["y_next"]

    X_test_t  = test_df_t[full_features]
    y_test_t  = test_df_t["y_next"]

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

print("\n=== Temporal Generalisation Summary (Price Regression) ===")
print(df_temporal)
