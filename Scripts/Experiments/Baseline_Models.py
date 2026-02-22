import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from experiment_common import (
    PRICE_COL,
    add_future_variance_target,
    keep_contiguous_rows,
    load_aggregated_data,
)

features = load_aggregated_data()

df = features.copy()
df = df.sort_index()
df = keep_contiguous_rows(df, prev_steps=1)

price_col = PRICE_COL
df["pred_last_value"] = df[price_col].shift(1)

window = 4
df["rolling_mean"] = df[price_col].rolling(window=window, min_periods=1).mean()
df["pred_rolling_mean"] = df["rolling_mean"].shift(1)

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

y_train = train[price_col]
y_test = test[price_col]


arima_model = ARIMA(y_train.values, order=(3, 1, 1)).fit()
arima_forecast = arima_model.forecast(steps=len(y_test))

def evaluate_regression(y_true_arr, y_pred_arr):
    """
    y_true_arr, y_pred_arr: 1D numpy arrays (same length)
    Drop NaNs from y_pred_arr, align by position.
    """
    mask = ~np.isnan(y_pred_arr)
    yt = y_true_arr[mask]
    yp = y_pred_arr[mask]

    mae = mean_absolute_error(yt, yp)
    mse = mean_squared_error(yt, yp)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "RMSE": rmse,
    }


print("\n--- Regression Baselines ---")

results_reg = {}

results_reg["Last Value"] = evaluate_regression(
    y_test.values,
    test["pred_last_value"].values,
)

results_reg["Rolling Mean"] = evaluate_regression(
    y_test.values,
    test["pred_rolling_mean"].values,
)

results_reg["ARIMA"] = evaluate_regression(
    y_test.values,
    np.array(arima_forecast),
)

for name, res in results_reg.items():
    print(f"\n{name}:")
    for metric, value in res.items():
        print(f"  {metric}: {value:.4f}")

spike_threshold = y_train.quantile(0.90)
df["is_spike"] = (df[price_col] > spike_threshold).astype(int)

y_train_cls = df["is_spike"].iloc[:split_idx]
y_test_cls = df["is_spike"].iloc[split_idx:]

test_cls = test.copy()

majority_class = y_train_cls.mode()[0]
test_cls["pred_majority"] = majority_class

p_spike = y_train_cls.mean()
rng = np.random.default_rng(seed=42)
test_cls["pred_random"] = rng.binomial(1, p_spike, size=len(test_cls))

test_cls["last_price"] = test_cls[price_col].shift(1)
test_cls["pred_price_rule"] = (test_cls["last_price"] > spike_threshold).astype(int)


def eval_classification(y_true, y_pred):
    """
    y_true, y_pred: 1D pandas Series (same index & length)
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


print("\n--- Classification Baselines ---")

results_cls = {}
results_cls["Majority"] = eval_classification(
    y_test_cls, test_cls["pred_majority"]
)
results_cls["Random"] = eval_classification(
    y_test_cls, test_cls["pred_random"]
)
results_cls["Last Price Rule"] = eval_classification(
    y_test_cls, test_cls["pred_price_rule"]
)

for name, res in results_cls.items():
    print(f"\n{name}:")
    for metric, value in res.items():
        print(f"  {metric}: {value:.4f}")

print("\n--- Regime Classification Baselines ---")

df_reg = df.copy()
add_future_variance_target(df_reg, source_col=price_col, horizon=4, out_col="future_vol")
df_reg = df_reg.dropna()

split_idx_reg = int(len(df_reg) * 0.8)
train_reg = df_reg.iloc[:split_idx_reg].copy()
test_reg = df_reg.iloc[split_idx_reg:].copy()

low_q = train_reg["future_vol"].quantile(0.60)
high_q = train_reg["future_vol"].quantile(0.90)

def vol_regime(v):
    if v <= low_q:
        return 0
    elif v <= high_q:
        return 1
    else:
        return 2

train_reg["vol_regime"] = train_reg["future_vol"].apply(vol_regime)
test_reg["vol_regime"] = test_reg["future_vol"].apply(vol_regime)

y_train_reg = train_reg["vol_regime"]
y_test_reg = test_reg["vol_regime"]

majority_reg = y_train_reg.mode()[0]
test_reg["pred_majority"] = majority_reg

class_probs = (
    y_train_reg.value_counts(normalize=True)
    .reindex([0, 1, 2], fill_value=0.0)
    .values
)
rng = np.random.default_rng(seed=42)
test_reg["pred_random"] = rng.choice(
    [0, 1, 2],
    size=len(test_reg),
    p=class_probs
)

test_reg["pred_persistence"] = test_reg["vol_regime"].shift(1)

mask = ~test_reg["pred_persistence"].isna()

def eval_multi(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
    }

results_regime = {}

results_regime["Majority"] = eval_multi(
    y_test_reg, test_reg["pred_majority"]
)

results_regime["Random"] = eval_multi(
    y_test_reg, test_reg["pred_random"]
)

results_regime["Persistence"] = eval_multi(
    y_test_reg[mask],
    test_reg.loc[mask, "pred_persistence"]
)

for name, res in results_regime.items():
    print(f"\n{name}:")
    for metric, value in res.items():
        print(f"  {metric}: {value:.4f}")

print("\n--- Classification Baselines (Volatility Spikes) ---")

df_vol = df[[price_col]].copy()
df_vol["simple_return"] = df_vol[price_col].pct_change()
df_vol["simple_return"] = df_vol["simple_return"].replace([np.inf, -np.inf], np.nan)
df_vol = keep_contiguous_rows(df_vol, prev_steps=1, next_steps=1)
df_vol = df_vol.dropna(subset=["simple_return"]).copy()

split_idx_vol = int(len(df_vol) * 0.8)
train_vol = df_vol.iloc[:split_idx_vol].copy()
test_vol = df_vol.iloc[split_idx_vol:].copy()

vol_threshold = train_vol["simple_return"].abs().quantile(0.99)

train_vol["spike_current"] = (train_vol["simple_return"].abs() > vol_threshold).astype(int)
test_vol["spike_current"] = (test_vol["simple_return"].abs() > vol_threshold).astype(int)

train_vol["spike_next"] = train_vol["spike_current"].shift(-1)
test_vol["spike_next"] = test_vol["spike_current"].shift(-1)

train_vol = train_vol.dropna(subset=["spike_next"]).copy()
test_vol = test_vol.dropna(subset=["spike_next"]).copy()

train_vol["spike_next"] = train_vol["spike_next"].astype(int)
test_vol["spike_next"] = test_vol["spike_next"].astype(int)

y_train_vol = train_vol["spike_next"]
y_test_vol = test_vol["spike_next"]

majority_vol = y_train_vol.mode()[0]
test_vol["pred_majority"] = majority_vol

p_spike_vol = y_train_vol.mean()
rng = np.random.default_rng(seed=42)
test_vol["pred_random"] = rng.binomial(1, p_spike_vol, size=len(test_vol))

test_vol["last_return"] = test_vol["simple_return"].shift(1)
test_vol["pred_last_price"] = (test_vol["last_return"].abs() > vol_threshold).astype(int)

results_vol_cls = {}
results_vol_cls["Majority"] = eval_classification(y_test_vol, test_vol["pred_majority"])
results_vol_cls["Random"] = eval_classification(y_test_vol, test_vol["pred_random"])
results_vol_cls["Last Price"] = eval_classification(y_test_vol, test_vol["pred_last_price"])

for name, res in results_vol_cls.items():
    print(f"\n{name}:")
    for metric, value in res.items():
        print(f"  {metric}: {value:.4f}")

print("\n--- Return Regression Baselines ---")

df_ret = df.copy()

# 1) Simple returns (can produce inf when previous price is 0)
df_ret["simple_return"] = df_ret[price_col].pct_change()

# 2) Replace inf/-inf with NaN
df_ret["simple_return"] = df_ret["simple_return"].replace([np.inf, -np.inf], np.nan)

# 3) Next-period return target
df_ret["next_return"] = df_ret["simple_return"].shift(-1)
df_ret = keep_contiguous_rows(df_ret, prev_steps=1, next_steps=1)

# 4) Drop bad rows BEFORE split
df_ret = df_ret.dropna(subset=["simple_return", "next_return"]).copy()

# 5) Split
split_idx_ret = int(len(df_ret) * 0.8)
train_ret = df_ret.iloc[:split_idx_ret].copy()
test_ret  = df_ret.iloc[split_idx_ret:].copy()

y_train_ret = train_ret["next_return"].values
y_test_ret  = test_ret["next_return"].values

test_ret["pred_last_return"] = test_ret["simple_return"].shift(1)

window = 4
df_ret["rolling_ret_mean"] = df_ret["simple_return"].rolling(window).mean()
df_ret["pred_rolling_ret"] = df_ret["rolling_ret_mean"].shift(1)
test_ret["pred_rolling_ret"] = df_ret["pred_rolling_ret"].iloc[split_idx_ret:].values

use_arima = True
arima_ret_forecast = None

if use_arima:
    try:
        arima_ret_model = ARIMA(y_train_ret, order=(1, 0, 0)).fit()
        arima_ret_forecast = arima_ret_model.forecast(steps=len(y_test_ret))
    except Exception as e:
        print("\nARIMA Return failed (returns often break ARIMA). Skipping ARIMA baseline.")
        print("Error:", repr(e))
        use_arima = False

def evaluate_return_regression(y_true_arr, y_pred_arr):
    # Drop NaNs + infs in predictions AND targets
    y_true_arr = np.asarray(y_true_arr, dtype=float)
    y_pred_arr = np.asarray(y_pred_arr, dtype=float)

    mask = (
        np.isfinite(y_true_arr) &
        np.isfinite(y_pred_arr)
    )
    yt = y_true_arr[mask]
    yp = y_pred_arr[mask]

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return mae, rmse

results_ret_reg = {}

results_ret_reg["Last Return"] = evaluate_return_regression(
    y_test_ret,
    test_ret["pred_last_return"].values
)

results_ret_reg["Rolling Mean Return"] = evaluate_return_regression(
    y_test_ret,
    test_ret["pred_rolling_ret"].values
)

if use_arima and arima_ret_forecast is not None:
    results_ret_reg["ARIMA Return"] = evaluate_return_regression(
        y_test_ret,
        np.array(arima_ret_forecast)
    )

for name, (mae, rmse) in results_ret_reg.items():
    print(f"\n{name}:")
    print(f"  MAE : {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
