from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd

PRICE_COL = "Price in €/MWh"
CONTROL_COL = "Controlled output requirements in MW"
DATA_FILE = "germany_2012_2016_aggregated.csv"
PRICE_LAGS = (1, 2, 3, 4)

MICRO_FEATURES = [
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

TIME_FEATURES = ["hour", "day_of_week"]

LEVEL_FEATURES = [
    "slope",
    "dev_mean",
    "dev_min",
    "dev_max",
    "var",
    "skewness",
    "kurtosis",
    "entropy",
]

EXCURSION_FEATURES = [
    "mild_excursions",
    "deep_excursions",
]

ROCOF_FEATURES = [
    "max_abs_rocof",
    "mean_abs_rocof",
    "rocof_std",
    "rocof_shock_count",
]

SHOCK_RECOVERY_FEATURES = [
    "shock_depth",
    "recovery_time",
    "post_shock_var",
]

ALL_MICRO_FEATURES = (
    LEVEL_FEATURES
    + EXCURSION_FEATURES
    + ROCOF_FEATURES
    + SHOCK_RECOVERY_FEATURES
)


def default_data_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "Aggregated Data" / DATA_FILE


def load_aggregated_data(data_path: Path | str | None = None) -> pd.DataFrame:
    csv_path = Path(data_path) if data_path else default_data_path()
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def add_price_lags(
    df: pd.DataFrame,
    *,
    price_col: str = PRICE_COL,
    lags: Sequence[int] = PRICE_LAGS,
) -> None:
    for lag in lags:
        df[f"price_lag{lag}"] = df[price_col].shift(lag)


def add_price_rolling_std(
    df: pd.DataFrame,
    *,
    price_col: str = PRICE_COL,
    window: int = 4,
    out_col: str = "price_rolling_std",
) -> None:
    df[out_col] = df[price_col].rolling(window=window).std()


def contiguous_index_mask(
    index: pd.Index,
    *,
    step: str | pd.Timedelta = "15min",
    prev_steps: int = 0,
    next_steps: int = 0,
) -> pd.Series:
    ts = pd.Series(index, index=index)
    delta = pd.to_timedelta(step)
    mask = pd.Series(True, index=index)

    for lag in range(1, prev_steps + 1):
        mask &= ts.diff(lag).eq(delta * lag)

    for lead in range(1, next_steps + 1):
        mask &= ts.shift(-lead).sub(ts).eq(delta * lead)

    return mask


def keep_contiguous_rows(
    df: pd.DataFrame,
    *,
    step: str | pd.Timedelta = "15min",
    prev_steps: int = 0,
    next_steps: int = 0,
) -> pd.DataFrame:
    mask = contiguous_index_mask(
        df.index,
        step=step,
        prev_steps=prev_steps,
        next_steps=next_steps,
    )
    return df.loc[mask].copy()


def add_future_variance_target(
    df: pd.DataFrame,
    *,
    source_col: str = PRICE_COL,
    horizon: int = 4,
    out_col: str = "future_vol",
) -> None:
    future_cols = [df[source_col].shift(-step) for step in range(1, horizon + 1)]
    future_frame = pd.concat(future_cols, axis=1)
    df[out_col] = future_frame.var(axis=1, ddof=1)
    next_contiguous = contiguous_index_mask(df.index, next_steps=horizon)
    df.loc[~next_contiguous, out_col] = np.nan


def add_hour_and_day(
    df: pd.DataFrame,
    *,
    hour_col: str = "hour",
    day_col: str = "day_of_week",
) -> None:
    df[hour_col] = df.index.hour
    df[day_col] = df.index.dayofweek


def market_price_features(
    *,
    price_col: str = PRICE_COL,
    control_col: str = CONTROL_COL,
) -> list[str]:
    return [
        price_col,
        "price_lag1",
        "price_lag2",
        "price_lag3",
        "price_lag4",
        "price_rolling_std",
        control_col,
    ]


def market_return_features(
    *,
    control_col: str = CONTROL_COL,
) -> list[str]:
    return [
        "return_lag1",
        "return_lag2",
        "return_lag3",
        "return_lag4",
        "return_roll_std",
        "return_roll_mean",
        control_col,
    ]


def split_train_test(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def split_train_val_test(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    test_start_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    val_start = int(n * train_fraction)
    test_start = int(n * test_start_fraction)
    return (
        df.iloc[:val_start].copy(),
        df.iloc[val_start:test_start].copy(),
        df.iloc[test_start:].copy(),
    )


def balanced_binary_index(
    y: pd.Series,
    *,
    neg_to_pos_ratio: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_pos = len(pos_idx)
    n_neg_sample = min(len(neg_idx), neg_to_pos_ratio * n_pos)

    neg_sample_idx = np.random.RandomState(random_state).choice(
        neg_idx,
        size=n_neg_sample,
        replace=False,
    )
    return np.sort(np.concatenate([pos_idx, neg_sample_idx]))


def save_feature_list(path: str | Path, features: Iterable[str]) -> None:
    with Path(path).open("w") as fp:
        json.dump(list(features), fp)


def build_market_only_price_frame(
    df: pd.DataFrame,
    *,
    price_col: str = PRICE_COL,
    control_col: str = CONTROL_COL,
) -> pd.DataFrame:
    df_market = df[[price_col, control_col]].copy()

    for i in PRICE_LAGS:
        df_market[f"price_lag{i}"] = df_market[price_col].shift(i)

    df_market["price_roll_mean_4"] = df_market[price_col].rolling(4).mean()
    df_market["price_roll_std_4"] = df_market[price_col].rolling(4).std()
    df_market["price_change"] = df_market[price_col].diff()

    df_market["MW_lag1"] = df_market[control_col].shift(1)
    df_market["MW_delta"] = df_market[control_col] - df_market["MW_lag1"]

    df_market["hour"] = df.index.hour
    df_market["dayofweek"] = df.index.dayofweek
    df_market["month"] = df.index.month

    return df_market


def build_market_only_return_frame(
    df: pd.DataFrame,
    *,
    price_col: str = PRICE_COL,
    control_col: str = CONTROL_COL,
) -> pd.DataFrame:
    df_market = df[[price_col, control_col]].copy()

    df_market["simple_return"] = df_market[price_col].pct_change()
    df_market["simple_return"] = df_market["simple_return"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for i in PRICE_LAGS:
        df_market[f"return_lag{i}"] = df_market["simple_return"].shift(i)

    df_market["return_roll_mean_4"] = df_market["simple_return"].rolling(4).mean()
    df_market["return_roll_std_4"] = df_market["simple_return"].rolling(4).std()

    df_market["MW_lag1"] = df_market[control_col].shift(1)
    df_market["MW_delta"] = df_market[control_col] - df_market["MW_lag1"]

    df_market["hour"] = df.index.hour
    df_market["dayofweek"] = df.index.dayofweek
    df_market["month"] = df.index.month

    return df_market
