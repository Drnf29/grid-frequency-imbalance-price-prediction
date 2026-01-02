import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0,
    parse_dates=True,
)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

price_col = "Price in €/MWh"
control_col = "Controlled output requirements in MW"

# Lagged prices (you can adjust lags)
for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)

# Simple rolling volatility (over last 4 intervals = 1 hour)
df["price_rolling_std"] = df[price_col].rolling(window=4).std()

df["y_next"] = df[price_col].shift(-1)

# Drop rows with NaNs introduced by shifting/rolling
df = df.dropna()

# Market features (Layer B)
market_features = [
    price_col,
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "price_lag4",
    "price_rolling_std",
    control_col,
]

# Micro-signal features (from your frequency engineering)
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
with open("price_regression_full_features.json", "w") as f:
    json.dump(full_features, f)

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

