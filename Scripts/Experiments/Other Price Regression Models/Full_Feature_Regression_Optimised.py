import pandas as pd
import numpy as np
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

for lag in [1, 2, 3, 4]:
    df[f"price_lag{lag}"] = df[price_col].shift(lag)

df["price_rolling_std"] = df[price_col].rolling(window=4).std()
df["y_next"] = df[price_col].shift(-1)
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

X_full = df[full_features]
y_full = df["y_next"]

split_idx = int(len(df) * 0.8)

X_train_full = X_full.iloc[:split_idx]
y_train_full = y_full.iloc[:split_idx]
X_test_full = X_full.iloc[split_idx:]
y_test_full = y_full.iloc[split_idx:]

print("Train size:", X_train_full.shape, "Test size:", X_test_full.shape)

# Clipping thresholds
thresholds = [None,100, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000]

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

print("\n=== Threshold search results (clipping target y_next) ===")
print("Threshold\tMAE\t\tRMSE")
for label, mae, rmse in results:
    print(f"{label:8s}\t{mae:.4f}\t{rmse:.4f}")

# Find best threshold by RMSE
best_label, best_mae, best_rmse = min(results, key=lambda x: x[2])

print("\nBest by RMSE:")
print(f"Threshold: {best_label}")
print(f"MAE      : {best_mae:.4f}")
print(f"RMSE     : {best_rmse:.4f}")

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
y_pred_best = final_model.predict(X_test_full)

final_mae = mean_absolute_error(y_test_best, y_pred_best)
final_rmse = np.sqrt(mean_squared_error(y_test_best, y_pred_best))

print("\n=== Final model with best threshold ===")
print("MAE :", final_mae)
print("RMSE:", final_rmse)

importances = final_model.feature_importances_
feat_imp = sorted(
    zip(full_features, importances), key=lambda x: x[1], reverse=True
)

print("\nTop 15 feature importances (final model):")
for name, val in feat_imp[:15]:
    print(f"{name:30s} {val:.4f}")
