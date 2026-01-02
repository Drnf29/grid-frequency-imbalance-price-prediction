import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

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
df["future_vol"] = (
    df[price_col]
    .shift(-1)
    .rolling(window=4)
    .var()
)
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

features = market_features + micro_features

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

low_q = train_df["future_vol"].quantile(0.60)
high_q = train_df["future_vol"].quantile(0.90)

def vol_regime(v):
    if v <= low_q:
        return 0
    elif v <= high_q:
        return 1
    else:
        return 2

train_df["vol_regime"] = train_df["future_vol"].apply(vol_regime)
test_df["vol_regime"]  = test_df["future_vol"].apply(vol_regime)

print("Train volatility regime distribution:")
print(train_df["vol_regime"].value_counts(normalize=True))

print("Test volatility regime distribution:")
print(test_df["vol_regime"].value_counts(normalize=True))

X_train = train_df[features]
y_train = train_df["vol_regime"]
X_test = test_df[features]
y_test = test_df["vol_regime"]

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

model.save_model("vol_regime_full_xgb.json")

with open("vol_regime_full_features.json", "w") as f:
    json.dump(features, f)

np.save("vol_regime_full_low_threshold.npy", low_q)
np.save("vol_regime_full_high_threshold.npy", high_q)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print("\n=== Volatility Regime Metrics ===")
print(f"Accuracy:     {acc:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")

print("\nDetailed Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low vol", "Medium vol", "High vol"],
        zero_division=0,
    )
)

# ==========================================================
# ABLATION ANALYSIS (Volatility Regime Classification)
# ==========================================================

print("\n================ ABLATION ANALYSIS (VOL REGIME) ================")

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

    "NO_LEVEL": market_features
                + excursion_features
                + rocof_features
                + shock_recovery_features,

    "NO_EXCURSIONS": market_features
                     + level_features
                     + rocof_features
                     + shock_recovery_features,

    "NO_ROCOF": market_features
                + level_features
                + excursion_features
                + shock_recovery_features,

    "NO_SHOCK_RECOVERY": market_features
                         + level_features
                         + excursion_features
                         + rocof_features,

    "MARKET_ONLY": market_features,

    "MICRO_ONLY": all_micro,
}

results_ablation = []

for name, feat_list in ablation_sets.items():

    X_train_ab = train_df[feat_list]
    y_train_ab = train_df["vol_regime"]

    X_test_ab = test_df[feat_list]
    y_test_ab = test_df["vol_regime"]

    model_ab = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
    )

    model_ab.fit(X_train_ab, y_train_ab)

    y_pred_ab = model_ab.predict(X_test_ab)

    acc = accuracy_score(y_test_ab, y_pred_ab)
    macro_f1 = f1_score(y_test_ab, y_pred_ab, average="macro")
    weighted_f1 = f1_score(y_test_ab, y_pred_ab, average="weighted")

    results_ablation.append((name, len(feat_list), acc, macro_f1, weighted_f1))

df_ablation = pd.DataFrame(
    results_ablation,
    columns=["Config", "NumFeatures", "Accuracy", "Macro_F1", "Weighted_F1"]
)

base_macro = df_ablation.loc[df_ablation["Config"]=="ALL_FEATURES","Macro_F1"].values[0]
df_ablation["Delta_Macro_F1"] = df_ablation["Macro_F1"] - base_macro

print("\n=== Volatility Regime Ablation Summary (sorted by ΔMacro_F1) ===")
print(df_ablation.sort_values("Delta_Macro_F1"))


# ==========================================================
# TEMPORAL GENERALISATION (Volatility Regime Classification)
# ==========================================================

print("\n================ TEMPORAL GENERALISATION (VOL REGIME) ================")

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

def assign_regime(series, low_q, high_q):
    # vectorised: 0 low, 1 medium, 2 high
    out = np.ones(len(series), dtype=int)
    out[series <= low_q] = 0
    out[series > high_q] = 2
    return out

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

    # thresholds from TRAIN ONLY
    low_q_t = train_df_t["future_vol"].quantile(0.60)
    high_q_t = train_df_t["future_vol"].quantile(0.90)

    train_df_t["vol_regime"] = assign_regime(train_df_t["future_vol"].values, low_q_t, high_q_t)
    test_df_t["vol_regime"]  = assign_regime(test_df_t["future_vol"].values,  low_q_t, high_q_t)

    X_train_t = train_df_t[features]
    y_train_t = train_df_t["vol_regime"]
    X_test_t  = test_df_t[features]
    y_test_t  = test_df_t["vol_regime"]

    model_t = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
    )

    model_t.fit(X_train_t, y_train_t)
    y_pred_t = model_t.predict(X_test_t)

    acc = accuracy_score(y_test_t, y_pred_t)
    macro_f1 = f1_score(y_test_t, y_pred_t, average="macro")
    weighted_f1 = f1_score(y_test_t, y_pred_t, average="weighted")

    results_temporal.append((train_years, test_label, acc, macro_f1, weighted_f1))

df_temporal = pd.DataFrame(
    results_temporal,
    columns=["TrainYears", "TestYears", "Accuracy", "Macro_F1", "Weighted_F1"]
)

print("\n=== Temporal Generalisation Summary (Vol Regime) ===")
print(df_temporal)
