import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


PRICE_COL = "Price in €/MWh"
MW_COL = "Controlled output requirements in MW"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Aggregated Data" / "germany_2012_2016_aggregated.csv"

TAB5_TASKS = {
    "Price Regression": {
        "task_type": "regression",
        "description": "Predict next 15-minute imbalance price (market-only vs full features).",
        "market_model": "Market Only/Price Regression/price_regression_market_only_xgb.json",
        "market_features": "Market Only/Price Regression/price_regression_market_only_features.json",
        "full_model": "Full Feature/Price Regression/price_regression_full_xgb.json",
        "full_features": "Full Feature/Price Regression/price_regression_full_features.json",
    },
    "Return Regression": {
        "task_type": "regression",
        "description": "Predict next 15-minute simple return (market-only vs full features).",
        "market_model": "Market Only/Return Regression/return_regression_market_only_xgb.json",
        "market_features": "Market Only/Return Regression/return_regression_market_only_features.json",
        "full_model": "Full Feature/Return Regression/return_regression_full_xgb.json",
        "full_features": "Full Feature/Return Regression/return_regression_full_features.json",
    },
    "Price Spike Classification": {
        "task_type": "binary",
        "description": "Classify whether next-interval price exceeds the top-10% training threshold.",
        "market_model": "Market Only/Price Spike Classification/price_spike_market_only_xgb.json",
        "market_features": "Market Only/Price Spike Classification/price_spike_market_only_features.json",
        "full_model": "Full Feature/Price Spike Classification/price_spike_full_xgb.json",
        "full_features": "Full Feature/Price Spike Classification/price_spike_full_features.json",
        "market_decision_threshold": "Market Only/Price Spike Classification/price_spike_market_only_decision_threshold.npy",
        "market_label_threshold": "Market Only/Price Spike Classification/price_spike_market_only_price_threshold.npy",
        "full_decision_threshold": "Full Feature/Price Spike Classification/price_spike_full_decision_threshold.npy",
        "full_label_threshold": "Full Feature/Price Spike Classification/price_spike_full_price_threshold.npy",
    },
    "Volatility Regime Classification": {
        "task_type": "multiclass",
        "description": "Classify low/medium/high future volatility regimes over the next hour.",
        "market_model": "Market Only/Regime Classification/vol_regime_market_only_xgb.json",
        "market_features": "Market Only/Regime Classification/vol_regime_market_only_features.json",
        "full_model": "Full Feature/Regime Classification/vol_regime_full_xgb.json",
        "full_features": "Full Feature/Regime Classification/vol_regime_full_features.json",
        "market_low_threshold": "Market Only/Regime Classification/vol_regime_market_only_low_threshold.npy",
        "market_high_threshold": "Market Only/Regime Classification/vol_regime_market_only_high_threshold.npy",
        "full_low_threshold": "Full Feature/Regime Classification/vol_regime_full_low_threshold.npy",
        "full_high_threshold": "Full Feature/Regime Classification/vol_regime_full_high_threshold.npy",
    },
    "Volatility Spike Classification": {
        "task_type": "binary",
        "description": "Classify top-1% absolute-return spike events for the next interval.",
        "market_model": "Market Only/Volatility Classification/vol_spike_market_only_xgb.json",
        "market_features": "Market Only/Volatility Classification/vol_spike_market_only_features.json",
        "full_model": "Full Feature/Volatility Classification/vol_spike_full_xgb.json",
        "full_features": "Full Feature/Volatility Classification/vol_spike_full_features.json",
        "market_decision_threshold": "Market Only/Volatility Classification/vol_spike_market_only_decision_threshold.npy",
        "market_label_threshold": "Market Only/Volatility Classification/vol_spike_market_only_return_threshold.npy",
        "full_decision_threshold": "Full Feature/Volatility Classification/vol_spike_full_decision_threshold.npy",
        "full_label_threshold": "Full Feature/Volatility Classification/vol_spike_full_return_threshold.npy",
    },
}


st.set_page_config(
    page_title="Quant Risk Analysis Dashboard",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.app-header {
  background: linear-gradient(90deg, #0f172a 0%, #1f2937 60%, #334155 100%);
  padding: 1rem 1.2rem;
  border-radius: 12px;
  color: #f8fafc;
  margin-bottom: 1rem;
}
.section-note {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-header">
  <h2 style="margin:0;">Quant Risk Analysis Dashboard</h2>
  <p style="margin:0.25rem 0 0 0; opacity:0.9;">
    German balancing market, 2012-2016, with model comparison and tail-risk views
  </p>
</div>
""",
    unsafe_allow_html=True,
)


def contiguous_mask(
    ts: pd.Series,
    *,
    step: str = "15min",
    prev_steps: int = 0,
    next_steps: int = 0,
) -> pd.Series:
    series = pd.Series(pd.to_datetime(ts).values, index=ts.index)
    delta = pd.to_timedelta(step)
    mask = pd.Series(True, index=ts.index)

    for lag in range(1, prev_steps + 1):
        mask &= series.diff(lag).eq(delta * lag)

    for lead in range(1, next_steps + 1):
        mask &= series.shift(-lead).sub(series).eq(delta * lead)

    return mask


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    idx_name = df.index.name
    df = df.reset_index()

    if "timestamp" not in df.columns:
        rename_map = {
            "index": "timestamp",
            "Timestamp": "timestamp",
        }
        if idx_name:
            rename_map[idx_name] = "timestamp"
        df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    df["simple_return"] = df[PRICE_COL].pct_change()
    df["simple_return"] = df["simple_return"].replace([np.inf, -np.inf], np.nan)

    df["log_return"] = np.nan
    positive_mask = df[PRICE_COL] > 0
    df.loc[positive_mask, "log_return"] = np.log(df.loc[positive_mask, PRICE_COL]).diff()

    df["vol_1h"] = df["simple_return"].rolling(window=4).std()
    df["vol_1d"] = df["simple_return"].rolling(window=96).std()

    vol_series = df["vol_1d"]
    valid_vol = vol_series.dropna()
    if len(valid_vol) > 0:
        low_q = valid_vol.quantile(0.60)
        high_q = valid_vol.quantile(0.90)
    else:
        low_q = np.nan
        high_q = np.nan

    def assign_regime(v: float) -> float:
        if np.isnan(v):
            return np.nan
        if v <= low_q:
            return 0
        if v <= high_q:
            return 1
        return 2

    df["vol_regime"] = vol_series.apply(assign_regime)
    regime_map = {0: "Low", 1: "Medium", 2: "High"}
    df["vol_regime_label"] = df["vol_regime"].map(regime_map)

    returns = df["simple_return"]
    if returns.notna().any():
        var_99 = np.nanquantile(returns, 0.01)
        es_99 = returns[returns <= var_99].mean()
    else:
        var_99 = np.nan
        es_99 = np.nan

    df["VaR_99"] = var_99
    df["ES_99"] = es_99

    window = 96
    df["VaR_99_rolling"] = returns.rolling(window).quantile(0.01)
    df["ES_99_rolling"] = returns.rolling(window).apply(
        lambda x: x[x <= np.nanquantile(x, 0.01)].mean(),
        raw=False,
    )

    n = len(df)
    train_end = int(n * 0.7)
    spike_threshold = df.iloc[:train_end][PRICE_COL].quantile(0.90)
    df["spike_current"] = (df[PRICE_COL] > spike_threshold).astype(int)
    df["is_spike"] = df["spike_current"].shift(-1)
    df = df.dropna(subset=["is_spike"]).copy()
    df["is_spike"] = df["is_spike"].astype(int)

    return df


def ensure_engineered_columns(df_in: pd.DataFrame, needed_cols: list[str]) -> pd.DataFrame:
    df2 = df_in.sort_values("timestamp").copy()

    base_missing = [c for c in ["timestamp", PRICE_COL, MW_COL] if c not in df2.columns]
    if base_missing:
        raise ValueError(f"Dashboard data missing base columns: {base_missing}")

    if "simple_return" in needed_cols and "simple_return" not in df2.columns:
        df2["simple_return"] = df2[PRICE_COL].pct_change()
        df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)

    if "hour" in needed_cols and "hour" not in df2.columns:
        df2["hour"] = pd.to_datetime(df2["timestamp"]).dt.hour
    if "dayofweek" in needed_cols and "dayofweek" not in df2.columns:
        df2["dayofweek"] = pd.to_datetime(df2["timestamp"]).dt.dayofweek
    if "day_of_week" in needed_cols and "day_of_week" not in df2.columns:
        df2["day_of_week"] = pd.to_datetime(df2["timestamp"]).dt.dayofweek
    if "month" in needed_cols and "month" not in df2.columns:
        df2["month"] = pd.to_datetime(df2["timestamp"]).dt.month

    lag_cols = [c for c in needed_cols if c.startswith("price_lag")]
    if lag_cols:
        max_lag = max(int(c.replace("price_lag", "")) for c in lag_cols if c.replace("price_lag", "").isdigit())
        for i in range(1, max_lag + 1):
            col = f"price_lag{i}"
            if col in needed_cols and col not in df2.columns:
                df2[col] = df2[PRICE_COL].shift(i)

    return_lag_cols = [c for c in needed_cols if c.startswith("return_lag")]
    if return_lag_cols:
        if "simple_return" not in df2.columns:
            df2["simple_return"] = df2[PRICE_COL].pct_change()
            df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)
        max_lag = max(int(c.replace("return_lag", "")) for c in return_lag_cols if c.replace("return_lag", "").isdigit())
        for i in range(1, max_lag + 1):
            col = f"return_lag{i}"
            if col in needed_cols and col not in df2.columns:
                df2[col] = df2["simple_return"].shift(i)

    if "price_roll_mean_4" in needed_cols and "price_roll_mean_4" not in df2.columns:
        df2["price_roll_mean_4"] = df2[PRICE_COL].rolling(4).mean()
    if "price_roll_std_4" in needed_cols and "price_roll_std_4" not in df2.columns:
        df2["price_roll_std_4"] = df2[PRICE_COL].rolling(4).std()
    if "price_rolling_std" in needed_cols and "price_rolling_std" not in df2.columns:
        df2["price_rolling_std"] = df2[PRICE_COL].rolling(4).std()

    if "price_change" in needed_cols and "price_change" not in df2.columns:
        df2["price_change"] = df2[PRICE_COL].diff()

    if "return_roll_mean_4" in needed_cols and "return_roll_mean_4" not in df2.columns:
        if "simple_return" not in df2.columns:
            df2["simple_return"] = df2[PRICE_COL].pct_change()
            df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)
        df2["return_roll_mean_4"] = df2["simple_return"].rolling(4).mean()
    if "return_roll_std_4" in needed_cols and "return_roll_std_4" not in df2.columns:
        if "simple_return" not in df2.columns:
            df2["simple_return"] = df2[PRICE_COL].pct_change()
            df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)
        df2["return_roll_std_4"] = df2["simple_return"].rolling(4).std()
    if "return_roll_mean" in needed_cols and "return_roll_mean" not in df2.columns:
        if "simple_return" not in df2.columns:
            df2["simple_return"] = df2[PRICE_COL].pct_change()
            df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)
        df2["return_roll_mean"] = df2["simple_return"].rolling(4).mean()
    if "return_roll_std" in needed_cols and "return_roll_std" not in df2.columns:
        if "simple_return" not in df2.columns:
            df2["simple_return"] = df2[PRICE_COL].pct_change()
            df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)
        df2["return_roll_std"] = df2["simple_return"].rolling(4).std()

    if "MW_lag1" in needed_cols and "MW_lag1" not in df2.columns:
        df2["MW_lag1"] = df2[MW_COL].shift(1)
    if "MW_delta" in needed_cols and "MW_delta" not in df2.columns:
        if "MW_lag1" not in df2.columns:
            df2["MW_lag1"] = df2[MW_COL].shift(1)
        df2["MW_delta"] = df2[MW_COL] - df2["MW_lag1"]

    return df2


def add_future_variance_target(
    df_in: pd.DataFrame,
    *,
    source_col: str = PRICE_COL,
    horizon: int = 4,
    out_col: str = "future_vol",
) -> pd.DataFrame:
    df2 = df_in.copy()
    future_cols = [df2[source_col].shift(-step) for step in range(1, horizon + 1)]
    future_frame = pd.concat(future_cols, axis=1)
    df2[out_col] = future_frame.var(axis=1, ddof=1)
    next_contiguous = contiguous_mask(df2["timestamp"], prev_steps=0, next_steps=horizon)
    df2.loc[~next_contiguous, out_col] = np.nan
    return df2


def load_booster(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


@st.cache_resource
def load_task_artifacts(base_dir: str, task_name: str) -> dict:
    config = TAB5_TASKS[task_name]
    models_dir = Path(base_dir)

    with (models_dir / config["market_features"]).open() as f:
        market_features = json.load(f)
    with (models_dir / config["full_features"]).open() as f:
        full_features = json.load(f)

    artifacts = {
        "market_model": load_booster(models_dir / config["market_model"]),
        "market_features": market_features,
        "full_model": load_booster(models_dir / config["full_model"]),
        "full_features": full_features,
    }

    for key, rel_path in config.items():
        if key.endswith("_threshold"):
            artifacts[key] = float(np.load(models_dir / rel_path))

    return artifacts


def model_feature_gain(model: xgb.Booster, top_n: int = 15) -> pd.DataFrame:
    scores = model.get_score(importance_type="gain")
    if not scores:
        return pd.DataFrame(columns=["feature", "gain"])
    gain = pd.DataFrame(scores.items(), columns=["feature", "gain"]).sort_values("gain", ascending=False)
    return gain.head(top_n)


def booster_predict(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(xgb.DMatrix(X)))


def binary_single_row_contributions(model: xgb.Booster, row_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    contrib = np.asarray(model.predict(xgb.DMatrix(row_df), pred_contribs=True))[0]
    bias = float(contrib[-1])
    contrib_values = contrib[:-1]
    out = pd.DataFrame(
        {
            "feature": row_df.columns,
            "contribution": contrib_values,
        }
    )
    out["abs_contribution"] = out["contribution"].abs()
    out = out.sort_values("abs_contribution", ascending=False)
    return out, bias


def binary_metric_row(name: str, y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "Model": name,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "PR-AUC": np.nan,
            "ROC-AUC": np.nan,
        }

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC": np.nan,
        "ROC-AUC": np.nan,
    }

    if y_true.nunique() > 1:
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)

    return metrics


def multiclass_metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "Model": name,
            "Accuracy": np.nan,
            "Precision (Macro)": np.nan,
            "Recall (Macro)": np.nan,
            "F1 (Macro)": np.nan,
            "Precision (Weighted)": np.nan,
            "Recall (Weighted)": np.nan,
            "F1 (Weighted)": np.nan,
        }

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def regression_metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "Model": name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(y_true) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan

    return {
        "Model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_regression_variant(
    *,
    df_in: pd.DataFrame,
    model: xgb.Booster,
    features: list[str],
    target_kind: str,
) -> dict:
    needed_cols = sorted(set(features + [PRICE_COL, "simple_return"]))
    df_model = ensure_engineered_columns(df_in.copy(), needed_cols)
    missing_features = [c for c in features if c not in df_model.columns]
    if missing_features:
        raise ValueError(f"Missing model features in dashboard data: {missing_features}")
    continuity = contiguous_mask(df_model["timestamp"], prev_steps=4, next_steps=1)
    df_model = df_model.loc[continuity].copy()

    if target_kind == "next_price":
        df_model["target"] = df_model[PRICE_COL].shift(-1)
    elif target_kind == "next_return":
        df_model["target"] = df_model["simple_return"].shift(-1)
    else:
        raise ValueError(f"Unsupported regression target: {target_kind}")

    df_model = df_model.dropna(subset=features + ["target"]).copy()
    split_idx = int(len(df_model) * 0.8)
    df_test = df_model.iloc[split_idx:].copy()

    if len(df_test) == 0:
        raise ValueError("No out-of-sample rows available after preprocessing.")

    y_pred = booster_predict(model, df_test[features])
    preds = pd.DataFrame(
        {
            "timestamp": df_test["timestamp"],
            "actual": df_test["target"],
            "pred": y_pred,
        }
    )
    preds["error"] = preds["actual"] - preds["pred"]

    return {
        "preds": preds,
        "rows_total": int(len(df_model)),
        "rows_test": int(len(df_test)),
        "split_idx": int(split_idx),
    }


def evaluate_binary_variant(
    *,
    df_in: pd.DataFrame,
    model: xgb.Booster,
    features: list[str],
    target_kind: str,
    label_threshold: float,
    decision_threshold: float,
) -> dict:
    needed_cols = sorted(set(features + [PRICE_COL] + (["simple_return"] if target_kind == "vol_spike" else [])))
    df_model = ensure_engineered_columns(df_in.copy(), needed_cols)
    missing_features = [c for c in features if c not in df_model.columns]
    if missing_features:
        raise ValueError(f"Missing model features in dashboard data: {missing_features}")
    continuity = contiguous_mask(df_model["timestamp"], prev_steps=4, next_steps=1)
    df_model = df_model.loc[continuity].copy()
    drop_cols = features + [PRICE_COL]
    if target_kind == "vol_spike":
        drop_cols = drop_cols + ["simple_return"]
    df_model = df_model.dropna(subset=drop_cols).copy()

    split_idx = int(len(df_model) * 0.8)
    df_test = df_model.iloc[split_idx:].copy()

    if target_kind == "price_spike":
        df_test["next_price"] = df_test[PRICE_COL].shift(-1)
        df_test = df_test.dropna(subset=["next_price"]).copy()
        y_true = (df_test["next_price"] > label_threshold).astype(int)
    elif target_kind == "vol_spike":
        df_test["spike_current"] = (df_test["simple_return"].abs() > label_threshold).astype(int)
        df_test["target"] = df_test["spike_current"].shift(-1)
        df_test = df_test.dropna(subset=["target"]).copy()
        y_true = df_test["target"].astype(int)
    else:
        raise ValueError(f"Unsupported binary target: {target_kind}")

    if len(df_test) == 0:
        raise ValueError("No out-of-sample rows available after preprocessing.")

    raw_prob = booster_predict(model, df_test[features])
    if raw_prob.ndim == 2:
        y_prob = raw_prob[:, 1]
    else:
        y_prob = raw_prob

    y_pred = (y_prob > decision_threshold).astype(int)

    preds = pd.DataFrame(
        {
            "timestamp": df_test["timestamp"],
            "actual": y_true.values,
            "pred": y_pred,
            "prob": y_prob,
        }
    )
    X_test = df_test[features].copy()
    X_test.insert(0, "timestamp", df_test["timestamp"].values)

    return {
        "preds": preds,
        "X_test": X_test,
        "rows_total": int(len(df_model)),
        "rows_test": int(len(df_test)),
        "split_idx": int(split_idx),
    }


def evaluate_regime_variant(
    *,
    df_in: pd.DataFrame,
    model: xgb.Booster,
    features: list[str],
    low_threshold: float,
    high_threshold: float,
) -> dict:
    needed_cols = sorted(set(features + [PRICE_COL]))
    df_model = ensure_engineered_columns(df_in.copy(), needed_cols)
    missing_features = [c for c in features if c not in df_model.columns]
    if missing_features:
        raise ValueError(f"Missing model features in dashboard data: {missing_features}")
    df_model = add_future_variance_target(df_model, source_col=PRICE_COL, horizon=4, out_col="future_vol")
    continuity = contiguous_mask(df_model["timestamp"], prev_steps=4, next_steps=4)
    df_model = df_model.loc[continuity].copy()
    df_model = df_model.dropna(subset=features + ["future_vol"]).copy()

    split_idx = int(len(df_model) * 0.8)
    df_test = df_model.iloc[split_idx:].copy()

    if len(df_test) == 0:
        raise ValueError("No out-of-sample rows available after preprocessing.")

    y_true = np.where(
        df_test["future_vol"] <= low_threshold,
        0,
        np.where(df_test["future_vol"] <= high_threshold, 1, 2),
    )

    probs = booster_predict(model, df_test[features])
    if probs.ndim == 1:
        y_pred = probs.astype(int)
        probs = np.full((len(y_pred), 3), np.nan)
    else:
        y_pred = np.argmax(probs, axis=1)

    preds = pd.DataFrame(
        {
            "timestamp": df_test["timestamp"],
            "actual": y_true.astype(int),
            "pred": y_pred.astype(int),
            "prob_low": probs[:, 0],
            "prob_medium": probs[:, 1],
            "prob_high": probs[:, 2],
        }
    )

    return {
        "preds": preds,
        "rows_total": int(len(df_model)),
        "rows_test": int(len(df_test)),
        "split_idx": int(split_idx),
    }


df = load_data(str(DATA_PATH))

st.sidebar.header("Controls")
dates = df["timestamp"].dt.to_pydatetime()
start, end = st.sidebar.slider(
    "Select time range",
    min_value=dates[0],
    max_value=dates[-1],
    value=(dates[0], dates[-1]),
)

df_view = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Rows in Window", f"{len(df_view):,}")
c2.metric("Mean Price (EUR/MWh)", f"{df_view[PRICE_COL].mean():.2f}")
c3.metric("Spike Rate", f"{df_view['is_spike'].mean():.2%}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Market Overview",
        "Tail Risk",
        "Volatility Regimes",
        "Spikes",
        "Models",
        "Uncertainty",
    ]
)

with tab1:
    st.subheader("Market Overview")

    fig_price = px.line(
        df_view,
        x="timestamp",
        y=PRICE_COL,
        title="Imbalance Price Over Time",
    )
    fig_returns = px.line(
        df_view,
        x="timestamp",
        y="simple_return",
        title="Simple Returns Over Time",
    )
    fig_vol = px.line(
        df_view,
        x="timestamp",
        y=["vol_1h", "vol_1d"],
        title="Rolling Volatility (1h and 1d)",
    )

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(fig_price, use_container_width=True)
    col_b.plotly_chart(fig_returns, use_container_width=True)
    st.plotly_chart(fig_vol, use_container_width=True)


with tab2:
    st.subheader("Return Distribution and Tail Risk")

    returns_view = df_view["simple_return"].dropna()
    if len(returns_view) <= 50:
        st.warning("Not enough data in selected time range to compute tail risk.")
    else:
        var_99_view = np.quantile(returns_view, 0.01)
        es_99_view = returns_view[returns_view <= var_99_view].mean()

        st.markdown(
            f"""
<div class="section-note">
<b>VaR (99%)</b>: {var_99_view:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
<b>Expected Shortfall (99%)</b>: {es_99_view:.4f}
</div>
""",
            unsafe_allow_html=True,
        )

        fig_full = px.histogram(
            returns_view,
            nbins=200,
            title="Full Return Distribution",
        )
        fig_full.add_vline(
            x=var_99_view,
            line_color="red",
            annotation_text="VaR 99%",
            annotation_position="top right",
        )
        fig_full.add_vline(
            x=es_99_view,
            line_color="darkorange",
            annotation_text="ES 99%",
            annotation_position="top left",
        )
        st.plotly_chart(fig_full, use_container_width=True)

        lower_bound = np.quantile(returns_view, 0.025)
        upper_bound = np.quantile(returns_view, 0.975)
        returns_zoom = returns_view[(returns_view >= lower_bound) & (returns_view <= upper_bound)]
        fig_zoom = px.histogram(
            returns_zoom,
            nbins=150,
            title="Central 95% of Returns (Zoomed View)",
        )
        st.plotly_chart(fig_zoom, use_container_width=True)

        fig_tail_time = px.line(
            df_view,
            x="timestamp",
            y=["VaR_99_rolling", "ES_99_rolling"],
            render_mode="webgl",
            title="Rolling VaR and Expected Shortfall (1-day Window)",
        )
        st.plotly_chart(fig_tail_time, use_container_width=True)


with tab3:
    st.subheader("Volatility Regimes")

    regime_filter = st.selectbox("Volatility regime", ["All", "Low", "Medium", "High"], index=0)
    df_regime_view = df_view.dropna(subset=["vol_regime", "vol_regime_label"]).copy()
    if regime_filter != "All":
        df_regime_view = df_regime_view[df_regime_view["vol_regime_label"] == regime_filter]

    if len(df_regime_view) == 0:
        st.info("No regime data in selected range.")
    else:
        fig_regime = px.scatter(
            df_regime_view,
            x="timestamp",
            y="vol_regime_label",
            color="vol_regime_label",
            title="Volatility Regime Over Time",
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        transition_matrix = pd.crosstab(
            df_regime_view["vol_regime"].shift(1),
            df_regime_view["vol_regime"],
            normalize="index",
        )
        if transition_matrix.size > 0:
            fig_heatmap = px.imshow(
                transition_matrix,
                text_auto=True,
                aspect="auto",
                title="Transition Probability Matrix",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        regime_dist = df_regime_view["vol_regime_label"].value_counts(normalize=True).rename("Share")
        risk_by_regime = df_view.groupby("vol_regime_label")["ES_99_rolling"].mean().rename("Avg ES_99_rolling")

        left, right = st.columns(2)
        left_df = regime_dist.reset_index()
        left_df.columns = ["Regime", "Share"]
        right_df = risk_by_regime.reset_index()
        right_df.columns = ["Regime", "Avg ES_99_rolling"]
        left.dataframe(left_df, use_container_width=True, hide_index=True)
        right.dataframe(right_df, use_container_width=True, hide_index=True)


with tab4:
    st.subheader("Spike Events and Jump Behaviour")

    fig_price_spikes = px.line(
        df_view,
        x="timestamp",
        y=PRICE_COL,
        title="Price with Spike Events",
    )
    spike_points = df_view[df_view["is_spike"] == 1]
    fig_price_spikes.add_scatter(
        x=spike_points["timestamp"],
        y=spike_points[PRICE_COL],
        mode="markers",
        marker=dict(color="red", size=6),
        name="Spike",
    )
    st.plotly_chart(fig_price_spikes, use_container_width=True)

    spike_rate = (
        df_view.groupby("vol_regime_label")["is_spike"]
        .mean()
        .sort_index()
        .rename("Spike Probability")
    )
    spike_df = spike_rate.reset_index()
    spike_df.columns = ["Regime", "Spike Probability"]
    st.dataframe(spike_df, use_container_width=True, hide_index=True)

    inter_arrival = spike_points["timestamp"].diff().dt.total_seconds().div(3600).dropna()
    if len(inter_arrival) > 0:
        fig_interarrival = px.histogram(
            inter_arrival,
            nbins=100,
            title="Distribution of Spike Inter-Arrival Times (Hours)",
        )
        st.plotly_chart(fig_interarrival, use_container_width=True)
    else:
        st.info("Not enough spikes in selected time range.")

    inter_df = spike_points.copy()
    inter_df["inter_arrival"] = inter_df["timestamp"].diff().dt.total_seconds().div(3600)
    inter_df = inter_df.dropna(subset=["inter_arrival"])
    if len(inter_df) > 0:
        fig_regime_inter = px.box(
            inter_df,
            x="vol_regime_label",
            y="inter_arrival",
            title="Inter-Arrival Time by Regime",
        )
        st.plotly_chart(fig_regime_inter, use_container_width=True)


with tab5:
    st.subheader("Model Comparison: Market-Only vs Full-Feature")

    task_names = list(TAB5_TASKS.keys())
    default_idx = task_names.index("Volatility Spike Classification")
    selected_task = st.selectbox("Model family", task_names, index=default_idx)
    selected_config = TAB5_TASKS[selected_task]
    st.caption(selected_config["description"])

    try:
        artifacts = load_task_artifacts(str(BASE_DIR), selected_task)
    except Exception as exc:
        st.error(f"Could not load model artifacts for Tab 5 ({selected_task}): {exc}")
        st.stop()

    market_model = artifacts["market_model"]
    market_features = artifacts["market_features"]
    full_model = artifacts["full_model"]
    full_features = artifacts["full_features"]

    setup_payload = {
        "Task": selected_task,
        "Continuity filter": "15-min step, prev_steps=4, next_steps=1 (regime uses next_steps=4)",
    }

    if selected_task in {"Price Regression", "Return Regression"}:
        target_kind = "next_price" if selected_task == "Price Regression" else "next_return"
        y_title = "Price (EUR/MWh)" if selected_task == "Price Regression" else "Simple Return"

        try:
            market_eval = evaluate_regression_variant(
                df_in=df,
                model=market_model,
                features=market_features,
                target_kind=target_kind,
            )
            full_eval = evaluate_regression_variant(
                df_in=df,
                model=full_model,
                features=full_features,
                target_kind=target_kind,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        market_window = market_eval["preds"][
            (market_eval["preds"]["timestamp"] >= start) & (market_eval["preds"]["timestamp"] <= end)
        ].copy()
        full_window = full_eval["preds"][
            (full_eval["preds"]["timestamp"] >= start) & (full_eval["preds"]["timestamp"] <= end)
        ].copy()

        if len(market_window) == 0 and len(full_window) == 0:
            st.info("No model-evaluation rows in selected time range.")
            st.stop()

        col_m, col_f = st.columns(2)
        if len(market_window) > 0:
            fig_market = px.line(
                market_window,
                x="timestamp",
                y=["actual", "pred"],
                title=f"Market-Only {selected_task}: Actual vs Predicted",
                labels={"value": y_title, "variable": "Series"},
            )
            col_m.plotly_chart(fig_market, use_container_width=True)
        if len(full_window) > 0:
            fig_full = px.line(
                full_window,
                x="timestamp",
                y=["actual", "pred"],
                title=f"Full-Feature {selected_task}: Actual vs Predicted",
                labels={"value": y_title, "variable": "Series"},
            )
            col_f.plotly_chart(fig_full, use_container_width=True)

        merged_preds = (
            market_window[["timestamp", "pred"]]
            .rename(columns={"pred": "pred_market"})
            .merge(
                full_window[["timestamp", "pred"]].rename(columns={"pred": "pred_full"}),
                on="timestamp",
                how="inner",
            )
        )
        if len(merged_preds) > 0:
            merged_preds["prediction_diff"] = merged_preds["pred_full"] - merged_preds["pred_market"]
            fig_diff = px.line(
                merged_preds,
                x="timestamp",
                y="prediction_diff",
                title="Prediction Difference (Full - Market)",
            )
            st.plotly_chart(fig_diff, use_container_width=True)

        metrics_rows = [
            regression_metric_row("Market-Only", market_window["actual"], market_window["pred"]),
            regression_metric_row("Full-Feature", full_window["actual"], full_window["pred"]),
        ]
        metrics_df = pd.DataFrame(metrics_rows)
        st.subheader("Out-of-Sample Metrics (Selected Window)")
        st.dataframe(
            metrics_df.style.format(
                {
                    "MAE": "{:.6f}",
                    "RMSE": "{:.6f}",
                    "R2": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        setup_payload.update(
            {
                "Market rows used": market_eval["rows_total"],
                "Market test rows (pre window)": market_eval["rows_test"],
                "Full rows used": full_eval["rows_total"],
                "Full test rows (pre window)": full_eval["rows_test"],
            }
        )

    elif selected_task in {"Price Spike Classification", "Volatility Spike Classification"}:
        target_kind = "price_spike" if selected_task == "Price Spike Classification" else "vol_spike"

        market_dec_thresh = artifacts["market_decision_threshold"]
        market_label_thresh = artifacts["market_label_threshold"]
        full_dec_thresh = artifacts["full_decision_threshold"]
        full_label_thresh = artifacts["full_label_threshold"]

        try:
            market_eval = evaluate_binary_variant(
                df_in=df,
                model=market_model,
                features=market_features,
                target_kind=target_kind,
                label_threshold=market_label_thresh,
                decision_threshold=market_dec_thresh,
            )
            full_eval = evaluate_binary_variant(
                df_in=df,
                model=full_model,
                features=full_features,
                target_kind=target_kind,
                label_threshold=full_label_thresh,
                decision_threshold=full_dec_thresh,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        market_mask = (market_eval["preds"]["timestamp"] >= start) & (market_eval["preds"]["timestamp"] <= end)
        full_mask = (full_eval["preds"]["timestamp"] >= start) & (full_eval["preds"]["timestamp"] <= end)

        market_window = market_eval["preds"][market_mask].copy()
        full_window = full_eval["preds"][full_mask].copy()
        market_x_window = market_eval["X_test"][
            (market_eval["X_test"]["timestamp"] >= start) & (market_eval["X_test"]["timestamp"] <= end)
        ].copy()
        full_x_window = full_eval["X_test"][
            (full_eval["X_test"]["timestamp"] >= start) & (full_eval["X_test"]["timestamp"] <= end)
        ].copy()

        if len(market_window) == 0 and len(full_window) == 0:
            st.info("No model-evaluation rows in selected time range.")
            st.stop()

        task_key = selected_task.lower().replace(" ", "_")
        st.subheader("Threshold Lab")
        th_col_m, th_col_f = st.columns(2)
        market_thr = th_col_m.slider(
            "Market-Only Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(market_dec_thresh),
            step=0.001,
            key=f"{task_key}_market_threshold",
        )
        full_thr = th_col_f.slider(
            "Full-Feature Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(full_dec_thresh),
            step=0.001,
            key=f"{task_key}_full_threshold",
        )

        market_window["pred"] = (market_window["prob"] > market_thr).astype(int)
        full_window["pred"] = (full_window["prob"] > full_thr).astype(int)

        col_m, col_f = st.columns(2)
        if len(market_window) > 0:
            fig_prob_market = px.line(
                market_window,
                x="timestamp",
                y=["prob", "actual", "pred"],
                title=f"Market-Only {selected_task}: Probabilities and Labels",
                labels={"value": "Value", "variable": "Series"},
            )
            for tr in fig_prob_market.data:
                if tr.name in {"actual", "pred"}:
                    tr.update(mode="markers", marker=dict(size=4))
            col_m.plotly_chart(fig_prob_market, use_container_width=True)

        if len(full_window) > 0:
            fig_prob_full = px.line(
                full_window,
                x="timestamp",
                y=["prob", "actual", "pred"],
                title=f"Full-Feature {selected_task}: Probabilities and Labels",
                labels={"value": "Value", "variable": "Series"},
            )
            for tr in fig_prob_full.data:
                if tr.name in {"actual", "pred"}:
                    tr.update(mode="markers", marker=dict(size=4))
            col_f.plotly_chart(fig_prob_full, use_container_width=True)

        merged_prob = (
            market_window[["timestamp", "prob"]]
            .rename(columns={"prob": "prob_market"})
            .merge(
                full_window[["timestamp", "prob"]].rename(columns={"prob": "prob_full"}),
                on="timestamp",
                how="inner",
            )
        )
        if len(merged_prob) > 0:
            merged_prob["prob_diff"] = merged_prob["prob_full"] - merged_prob["prob_market"]
            fig_diff = px.line(
                merged_prob,
                x="timestamp",
                y="prob_diff",
                title="Probability Difference (Full - Market)",
            )
            st.plotly_chart(fig_diff, use_container_width=True)

        metrics_rows = [
            binary_metric_row("Market-Only", market_window["actual"], market_window["prob"], market_window["pred"]),
            binary_metric_row("Full-Feature", full_window["actual"], full_window["prob"], full_window["pred"]),
        ]
        metrics_df = pd.DataFrame(metrics_rows)
        st.subheader("Out-of-Sample Metrics (Selected Window)")
        st.dataframe(
            metrics_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "PR-AUC": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        positive_summary = pd.DataFrame(
            [
                {
                    "Model": "Market-Only",
                    "Actual Positive Rate": float(market_window["actual"].mean()),
                    "Predicted Positive Rate": float(market_window["pred"].mean()),
                    "Actual Positives": int(market_window["actual"].sum()),
                    "Predicted Positives": int(market_window["pred"].sum()),
                    "Rows": int(len(market_window)),
                },
                {
                    "Model": "Full-Feature",
                    "Actual Positive Rate": float(full_window["actual"].mean()),
                    "Predicted Positive Rate": float(full_window["pred"].mean()),
                    "Actual Positives": int(full_window["actual"].sum()),
                    "Predicted Positives": int(full_window["pred"].sum()),
                    "Rows": int(len(full_window)),
                },
            ]
        )
        st.subheader("Positive Event Summary (Selected Window)")
        st.dataframe(
            positive_summary.style.format(
                {
                    "Actual Positive Rate": "{:.4%}",
                    "Predicted Positive Rate": "{:.4%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        evt_col_m, evt_col_f = st.columns(2)
        market_events = pd.concat(
            [
                pd.DataFrame(
                    {
                        "timestamp": market_window.loc[market_window["actual"] == 1, "timestamp"],
                        "Series": "Actual=1",
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": market_window.loc[market_window["pred"] == 1, "timestamp"],
                        "Series": "Pred=1",
                    }
                ),
            ],
            ignore_index=True,
        )
        full_events = pd.concat(
            [
                pd.DataFrame(
                    {
                        "timestamp": full_window.loc[full_window["actual"] == 1, "timestamp"],
                        "Series": "Actual=1",
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": full_window.loc[full_window["pred"] == 1, "timestamp"],
                        "Series": "Pred=1",
                    }
                ),
            ],
            ignore_index=True,
        )

        if len(market_events) > 0:
            fig_market_events = px.scatter(
                market_events,
                x="timestamp",
                y="Series",
                color="Series",
                title="Market-Only Positive Event Times",
            )
            evt_col_m.plotly_chart(fig_market_events, use_container_width=True)
        if len(full_events) > 0:
            fig_full_events = px.scatter(
                full_events,
                x="timestamp",
                y="Series",
                color="Series",
                title="Full-Feature Positive Event Times",
            )
            evt_col_f.plotly_chart(fig_full_events, use_container_width=True)

        st.subheader("Why Did It Predict This?")
        explain_pool = (
            market_window[["timestamp", "actual", "pred", "prob"]]
            .rename(
                columns={
                    "actual": "actual_market",
                    "pred": "pred_market",
                    "prob": "prob_market",
                }
            )
            .merge(
                full_window[["timestamp", "actual", "pred", "prob"]].rename(
                    columns={
                        "actual": "actual_full",
                        "pred": "pred_full",
                        "prob": "prob_full",
                    }
                ),
                on="timestamp",
                how="inner",
            )
            .sort_values("timestamp")
        )

        if len(explain_pool) == 0:
            st.info("No overlapping timestamps between market-only and full-feature predictions for explanation.")
        else:
            focus_mode = st.selectbox(
                "Candidate timestamps",
                [
                    "Actual spikes only",
                    "Top Full-Feature probability",
                    "Top Market-Only probability",
                    "Largest probability gap",
                    "All (chronological sample)",
                ],
                key=f"{task_key}_focus_mode",
            )

            if focus_mode == "Actual spikes only":
                explain_candidates = explain_pool[
                    (explain_pool["actual_market"] == 1) | (explain_pool["actual_full"] == 1)
                ].copy()
            elif focus_mode == "Top Full-Feature probability":
                explain_candidates = explain_pool.sort_values("prob_full", ascending=False).head(800).copy()
            elif focus_mode == "Top Market-Only probability":
                explain_candidates = explain_pool.sort_values("prob_market", ascending=False).head(800).copy()
            elif focus_mode == "Largest probability gap":
                explain_candidates = explain_pool.copy()
                explain_candidates["prob_gap_abs"] = (
                    explain_candidates["prob_full"] - explain_candidates["prob_market"]
                ).abs()
                explain_candidates = explain_candidates.sort_values("prob_gap_abs", ascending=False).head(800).copy()
            else:
                explain_candidates = explain_pool.copy()

            if len(explain_candidates) == 0:
                st.info("No rows available for this explanation filter.")
            else:
                if len(explain_candidates) > 1200:
                    stride = int(np.ceil(len(explain_candidates) / 1200))
                    explain_candidates = explain_candidates.iloc[::stride].copy()

                pick_idx = st.slider(
                    "Pick candidate row",
                    min_value=0,
                    max_value=int(len(explain_candidates) - 1),
                    value=int(min(20, len(explain_candidates) - 1)),
                    step=1,
                    key=f"{task_key}_explain_row_idx",
                )
                selected_row = explain_candidates.iloc[pick_idx]
                ts_selected = pd.to_datetime(selected_row["timestamp"])

                st.caption(
                    f"Timestamp: {ts_selected} | "
                    f"Market prob={selected_row['prob_market']:.4f}, pred={int(selected_row['pred_market'])}, actual={int(selected_row['actual_market'])} | "
                    f"Full prob={selected_row['prob_full']:.4f}, pred={int(selected_row['pred_full'])}, actual={int(selected_row['actual_full'])}"
                )

                top_n = st.slider(
                    "Top features to show",
                    min_value=5,
                    max_value=20,
                    value=10,
                    step=1,
                    key=f"{task_key}_explain_top_n",
                )

                market_row = market_x_window.loc[
                    market_x_window["timestamp"] == ts_selected,
                    market_features,
                ].head(1)
                full_row = full_x_window.loc[
                    full_x_window["timestamp"] == ts_selected,
                    full_features,
                ].head(1)

                if len(market_row) == 0 or len(full_row) == 0:
                    st.info("Could not locate feature rows for the selected timestamp.")
                else:
                    m_contrib, _ = binary_single_row_contributions(market_model, market_row)
                    f_contrib, _ = binary_single_row_contributions(full_model, full_row)

                    extra_features = [f for f in full_features if f not in market_features]
                    extra_net = float(
                        f_contrib.loc[f_contrib["feature"].isin(extra_features), "contribution"].sum()
                    )

                    explain_col_m, explain_col_f = st.columns(2)
                    explain_col_m.metric("Market Threshold", f"{market_thr:.3f}")
                    explain_col_m.metric("Market Probability", f"{float(selected_row['prob_market']):.4f}")
                    explain_col_m.metric("Market Prediction", str(int(selected_row["pred_market"])))

                    explain_col_f.metric("Full Threshold", f"{full_thr:.3f}")
                    explain_col_f.metric("Full Probability", f"{float(selected_row['prob_full']):.4f}")
                    explain_col_f.metric("Full Prediction", str(int(selected_row["pred_full"])))
                    explain_col_f.caption(
                        f"Net contribution of extra (grid-frequency) features: {extra_net:.4f} log-odds"
                    )

                    fig_m_contrib = px.bar(
                        m_contrib.head(top_n).sort_values("contribution"),
                        x="contribution",
                        y="feature",
                        orientation="h",
                        title="Market-Only Top Feature Contributions",
                        color="contribution",
                        color_continuous_scale="RdBu",
                    )
                    fig_f_contrib = px.bar(
                        f_contrib.head(top_n).sort_values("contribution"),
                        x="contribution",
                        y="feature",
                        orientation="h",
                        title="Full-Feature Top Feature Contributions",
                        color="contribution",
                        color_continuous_scale="RdBu",
                    )
                    explain_col_m.plotly_chart(fig_m_contrib, use_container_width=True)
                    explain_col_f.plotly_chart(fig_f_contrib, use_container_width=True)

                    st.caption(
                        "Contribution values are in log-odds units from XGBoost (pred_contribs). "
                        "Positive values push spike probability up, negative values push it down."
                    )

        setup_payload.update(
            {
                "Market rows used": market_eval["rows_total"],
                "Market test rows (pre window)": market_eval["rows_test"],
                "Full rows used": full_eval["rows_total"],
                "Full test rows (pre window)": full_eval["rows_test"],
                "Market saved decision threshold": market_dec_thresh,
                "Market active decision threshold": market_thr,
                "Market label threshold": market_label_thresh,
                "Full saved decision threshold": full_dec_thresh,
                "Full active decision threshold": full_thr,
                "Full label threshold": full_label_thresh,
            }
        )

    else:
        market_low_threshold = artifacts["market_low_threshold"]
        market_high_threshold = artifacts["market_high_threshold"]
        full_low_threshold = artifacts["full_low_threshold"]
        full_high_threshold = artifacts["full_high_threshold"]

        try:
            market_eval = evaluate_regime_variant(
                df_in=df,
                model=market_model,
                features=market_features,
                low_threshold=market_low_threshold,
                high_threshold=market_high_threshold,
            )
            full_eval = evaluate_regime_variant(
                df_in=df,
                model=full_model,
                features=full_features,
                low_threshold=full_low_threshold,
                high_threshold=full_high_threshold,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        market_window = market_eval["preds"][
            (market_eval["preds"]["timestamp"] >= start) & (market_eval["preds"]["timestamp"] <= end)
        ].copy()
        full_window = full_eval["preds"][
            (full_eval["preds"]["timestamp"] >= start) & (full_eval["preds"]["timestamp"] <= end)
        ].copy()

        if len(market_window) == 0 and len(full_window) == 0:
            st.info("No model-evaluation rows in selected time range.")
            st.stop()

        col_m, col_f = st.columns(2)

        if len(market_window) > 0:
            market_plot = market_window.melt(
                id_vars="timestamp",
                value_vars=["actual", "pred"],
                var_name="Series",
                value_name="Regime",
            )
            fig_market = px.scatter(
                market_plot,
                x="timestamp",
                y="Regime",
                color="Series",
                title="Market-Only Regime: Actual vs Predicted",
            )
            fig_market.update_yaxes(tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"])
            col_m.plotly_chart(fig_market, use_container_width=True)

            fig_market_prob = px.line(
                market_window,
                x="timestamp",
                y=["prob_low", "prob_medium", "prob_high"],
                title="Market-Only Class Probabilities",
            )
            col_m.plotly_chart(fig_market_prob, use_container_width=True)

        if len(full_window) > 0:
            full_plot = full_window.melt(
                id_vars="timestamp",
                value_vars=["actual", "pred"],
                var_name="Series",
                value_name="Regime",
            )
            fig_full = px.scatter(
                full_plot,
                x="timestamp",
                y="Regime",
                color="Series",
                title="Full-Feature Regime: Actual vs Predicted",
            )
            fig_full.update_yaxes(tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"])
            col_f.plotly_chart(fig_full, use_container_width=True)

            fig_full_prob = px.line(
                full_window,
                x="timestamp",
                y=["prob_low", "prob_medium", "prob_high"],
                title="Full-Feature Class Probabilities",
            )
            col_f.plotly_chart(fig_full_prob, use_container_width=True)

        metrics_rows = [
            multiclass_metric_row("Market-Only", market_window["actual"], market_window["pred"]),
            multiclass_metric_row("Full-Feature", full_window["actual"], full_window["pred"]),
        ]
        metrics_df = pd.DataFrame(metrics_rows)
        st.subheader("Out-of-Sample Metrics (Selected Window)")
        st.dataframe(
            metrics_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision (Macro)": "{:.4f}",
                    "Recall (Macro)": "{:.4f}",
                    "F1 (Macro)": "{:.4f}",
                    "Precision (Weighted)": "{:.4f}",
                    "Recall (Weighted)": "{:.4f}",
                    "F1 (Weighted)": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        labels = [0, 1, 2]
        cm_market = confusion_matrix(market_window["actual"], market_window["pred"], labels=labels)
        cm_full = confusion_matrix(full_window["actual"], full_window["pred"], labels=labels)
        cm_market_df = pd.DataFrame(
            cm_market,
            index=["Actual Low", "Actual Medium", "Actual High"],
            columns=["Pred Low", "Pred Medium", "Pred High"],
        )
        cm_full_df = pd.DataFrame(
            cm_full,
            index=["Actual Low", "Actual Medium", "Actual High"],
            columns=["Pred Low", "Pred Medium", "Pred High"],
        )

        cm_col_m, cm_col_f = st.columns(2)
        cm_col_m.dataframe(cm_market_df, use_container_width=True)
        cm_col_f.dataframe(cm_full_df, use_container_width=True)

        setup_payload.update(
            {
                "Market rows used": market_eval["rows_total"],
                "Market test rows (pre window)": market_eval["rows_test"],
                "Full rows used": full_eval["rows_total"],
                "Full test rows (pre window)": full_eval["rows_test"],
                "Market low threshold": market_low_threshold,
                "Market high threshold": market_high_threshold,
                "Full low threshold": full_low_threshold,
                "Full high threshold": full_high_threshold,
            }
        )

    with st.expander("Model Setup and Saved Artifact Values", expanded=False):
        st.write(setup_payload)
        st.write(
            {
                "Market feature count": len(market_features),
                "Full feature count": len(full_features),
            }
        )

    m_gain = model_feature_gain(market_model, top_n=12)
    f_gain = model_feature_gain(full_model, top_n=12)
    col_m_gain, col_f_gain = st.columns(2)
    if len(m_gain) > 0:
        fig_m_gain = px.bar(
            m_gain.sort_values("gain"),
            x="gain",
            y="feature",
            orientation="h",
            title="Market-Only Top Feature Gain",
        )
        col_m_gain.plotly_chart(fig_m_gain, use_container_width=True)
    if len(f_gain) > 0:
        fig_f_gain = px.bar(
            f_gain.sort_values("gain"),
            x="gain",
            y="feature",
            orientation="h",
            title="Full-Feature Top Feature Gain",
        )
        col_f_gain.plotly_chart(fig_f_gain, use_container_width=True)


with tab6:
    st.subheader("Uncertainty")

    df_unc = df_view.dropna(subset=["simple_return"]).copy()
    if len(df_unc) < 50:
        st.info("Not enough data in selected window for uncertainty diagnostics.")
    else:
        df_unc["abs_return"] = df_unc["simple_return"].abs()
        df_unc["abs_return_roll_1d"] = df_unc["abs_return"].rolling(96).mean()
        df_unc["abs_return_roll_1w"] = df_unc["abs_return"].rolling(96 * 7).mean()

        fig_unc = px.line(
            df_unc,
            x="timestamp",
            y=["abs_return_roll_1d", "abs_return_roll_1w"],
            title="Rolling Absolute Return (Uncertainty Proxy)",
        )
        st.plotly_chart(fig_unc, use_container_width=True)

        fig_abs = px.histogram(
            df_unc["abs_return"],
            nbins=120,
            title="Absolute Return Distribution",
        )
        st.plotly_chart(fig_abs, use_container_width=True)

        fig_joint = px.scatter(
            df_unc,
            x="vol_1h",
            y="abs_return",
            color="vol_regime_label",
            title="Intrahour Volatility vs Absolute Return",
        )
        st.plotly_chart(fig_joint, use_container_width=True)
