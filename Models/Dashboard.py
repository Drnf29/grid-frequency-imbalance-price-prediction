import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import xgboost as xgb

@st.cache_data
def load_data():
    df = pd.read_csv(
        "../Aggregated Data/germany_2012_2016_aggregated.csv",
        index_col=0,
        parse_dates=True
    )

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.reset_index(names="timestamp")

    df["simple_return"] = df["Price in €/MWh"].pct_change()
    df["simple_return"] = df["simple_return"].replace([np.inf, -np.inf], np.nan)

    df["log_return"] = np.nan
    positive_mask = df["Price in €/MWh"] > 0
    df.loc[positive_mask, "log_return"] = np.log(
        df.loc[positive_mask, "Price in €/MWh"]
    ).diff()

    df["vol_1h"] = df["simple_return"].rolling(window=4).std()
    df["vol_1d"] = df["simple_return"].rolling(window=96).std()

    vol_series = df["vol_1d"]

    low_q = vol_series.quantile(0.60)
    high_q = vol_series.quantile(0.90)

    def assign_regime(v):
        if np.isnan(v):
            return np.nan
        elif v <= low_q:
            return 0   # Low
        elif v <= high_q:
            return 1   # Medium
        else:
            return 2   # High

    df["vol_regime"] = vol_series.apply(assign_regime)

    regime_map = {0: "Low", 1: "Medium", 2: "High"}
    df["vol_regime_label"] = df["vol_regime"].map(regime_map)

    returns = df["simple_return"]

    df["VaR_99"] = np.nan
    df["ES_99"] = np.nan

    var_99 = np.nanquantile(returns, 0.01)
    es_99 = returns[returns <= var_99].mean()

    df.loc[:, "VaR_99"] = var_99
    df.loc[:, "ES_99"] = es_99

    window = 96  

    df["VaR_99_rolling"] = returns.rolling(window).quantile(0.01)

    df["ES_99_rolling"] = returns.rolling(window).apply(
        lambda x: x[x <= np.nanquantile(x, 0.01)].mean(),
        raw=False
    )

    price_col = "Price in €/MWh"

    n = len(df)
    train_end = int(n * 0.7)

    spike_threshold = df.iloc[:train_end][price_col].quantile(0.90)

    df["spike_current"] = (df[price_col] > spike_threshold).astype(int)
    df["is_spike"] = df["spike_current"].shift(-1)

    df = df.dropna(subset=["is_spike"]).copy()
    df["is_spike"] = df["is_spike"].astype(int)

    return df

df = load_data()

# Time slider
dates = df["timestamp"].dt.to_pydatetime()

start, end = st.slider(
    "Select time range",
    min_value=dates[0],
    max_value=dates[-1],
    value=(dates[0], dates[-1]),
)

df_view = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

# Different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Overview",
    "Tail Risk",
    "Volatility Regimes",
    "Spikes",
    "Models",
    "Uncertainty"
])

with tab1:
    fig_price = px.line(
        df_view,
        x="timestamp",
        y="Price in €/MWh",
        title="Imbalance Price Over Time"
    )

    fig_returns = px.line(
        df_view,
        x="timestamp",
        y="simple_return",
        title="Simple Returns Over Time"
    )

    fig_vol = px.line(
        df_view,
        x="timestamp",
        y=["vol_1h", "vol_1d"],
        title="Rolling Volatility (1h and 1d)"
    )

    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_returns, use_container_width=True)
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.subheader("Return Distribution and Tail Risk")

    returns_view = df_view["simple_return"].dropna()

    if len(returns_view) > 50:

        var_99_view = np.quantile(returns_view, 0.01)
        es_99_view = returns_view[returns_view <= var_99_view].mean()

        fig_full = px.histogram(
            returns_view,
            nbins=200,
            title="Full Return Distribution"
        )

        fig_full.add_vline(
            x=var_99_view,
            line_color="red",
            annotation_text="VaR 99%",
            annotation_position="top right"
        )

        fig_full.add_vline(
            x=es_99_view,
            line_color="purple",
            annotation_text="ES 99%",
            annotation_position="top left"
        )

        st.plotly_chart(fig_full, use_container_width=True)

        st.markdown(
            f"""
            **VaR (99%)**: {var_99_view:.4f}  
            **Expected Shortfall (99%)**: {es_99_view:.4f}
            """
        )

        lower_bound = np.quantile(returns_view, 0.025)
        upper_bound = np.quantile(returns_view, 0.975)

        returns_zoom = returns_view[
            (returns_view >= lower_bound) &
            (returns_view <= upper_bound)
        ]

        fig_zoom = px.histogram(
            returns_zoom,
            nbins=150,
            title="Central 95% of Returns (Zoomed View)"
        )

        st.plotly_chart(fig_zoom, use_container_width=True)

        st.subheader("Rolling Tail Risk")

        fig_tail_time = px.line(
            df_view,
            x="timestamp",
            y=["VaR_99_rolling", "ES_99_rolling"],
            render_mode="webgl",
            title="Rolling VaR and Expected Shortfall (1-day window)"
        )

        st.plotly_chart(fig_tail_time, use_container_width=True)

    else:
        st.warning("Not enough data in selected time range to compute tail risk.")

with tab3:
    st.subheader("Volatility Regimes Over Time")

    regime = st.selectbox("Volatility regime", ["All", "Low", "Mid", "High"])

    df_regime_view = df_view.dropna(subset=["vol_regime"])

    fig_regime = px.scatter(
        df_regime_view,
        x="timestamp",
        y="vol_regime_label",
        color="vol_regime_label",
        title="Volatility Regime Over Time"
    )

    st.plotly_chart(fig_regime, use_container_width=True)

    st.subheader("Regime Transition Matrix")

    regimes = df_regime_view["vol_regime"]

    transition_matrix = pd.crosstab(
        regimes.shift(1),
        regimes,
        normalize="index"
    )

    fig_heatmap = px.imshow(
        transition_matrix,
        text_auto=True,
        aspect="auto",
        title="Transition Probability Matrix"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Regime Distribution")

    regime_dist = df_regime_view["vol_regime_label"].value_counts(normalize=True)

    st.write(regime_dist)

    risk_by_regime = df_view.groupby("vol_regime_label")["ES_99_rolling"].mean()
    st.subheader("Average Expected Shortfall by Regime:")
    st.write(risk_by_regime)

with tab4:
    st.subheader("Spike Events and Jump Behaviour")

    fig_price_spikes = px.line(
        df_view,
        x="timestamp",
        y="Price in €/MWh",
        title="Price with Spike Events"
    )

    spike_points = df_view[df_view["is_spike"] == 1]

    fig_price_spikes.add_scatter(
        x=spike_points["timestamp"],
        y=spike_points["Price in €/MWh"],
        mode="markers",
        marker=dict(color="red", size=6),
        name="Spike"
    )

    st.plotly_chart(fig_price_spikes, use_container_width=True)

    st.subheader("Spike Probability by Regime")

    spike_rate = (
        df_view
        .groupby("vol_regime_label")["is_spike"]
        .mean()
        .sort_index()
    )

    st.write(spike_rate)

    st.subheader("Spike Inter-Arrival Times (Hours)")

    spike_times = spike_points["timestamp"]
    inter_arrival = (
        spike_times.diff()
        .dt.total_seconds() / 3600
    ).dropna()

    if len(inter_arrival) > 0:

        fig_interarrival = px.histogram(
            inter_arrival,
            nbins=100,
            title="Distribution of Spike Inter-Arrival Times"
        )

        st.plotly_chart(fig_interarrival, use_container_width=True)

    else:
        st.info("Not enough spikes in selected time range.")

    st.subheader("Inter-Arrival Times by Regime")

    inter_df = spike_points.copy()
    inter_df["inter_arrival"] = (
        inter_df["timestamp"]
        .diff()
        .dt.total_seconds() / 3600
    )

    inter_df = inter_df.dropna(subset=["inter_arrival"])

    if len(inter_df) > 0:

        fig_regime_inter = px.box(
            inter_df,
            x="vol_regime_label",
            y="inter_arrival",
            title="Inter-Arrival Time by Regime"
        )

        st.plotly_chart(fig_regime_inter, use_container_width=True)

    else:
        st.info("Not enough regime-separated spike data.")

with tab5:
    st.subheader("Volatility Spike Classification – Market vs Micro")

    from sklearn.metrics import average_precision_score, roc_auc_score

    # ---------------------------
    # Helpers: build missing engineered columns
    # ---------------------------
    def ensure_engineered_columns(df_in: pd.DataFrame, needed_cols: list) -> pd.DataFrame:
        df2 = df_in.sort_values("timestamp").copy()

        price_col = "Price in €/MWh"
        mw_col = "Controlled output requirements in MW"

        # Make sure base columns exist
        base_missing = [c for c in ["timestamp", price_col, mw_col] if c not in df2.columns]
        if base_missing:
            st.error(f"Dashboard data missing base columns: {base_missing}")
            st.stop()

        # simple_return (used by your vol-spike label + some models)
        if "simple_return" in needed_cols and "simple_return" not in df2.columns:
            df2["simple_return"] = df2[price_col].pct_change()
            df2["simple_return"] = df2["simple_return"].replace([np.inf, -np.inf], np.nan)

        # time features (cover both naming conventions)
        if "hour" in needed_cols and "hour" not in df2.columns:
            df2["hour"] = pd.to_datetime(df2["timestamp"]).dt.hour
        if "dayofweek" in needed_cols and "dayofweek" not in df2.columns:
            df2["dayofweek"] = pd.to_datetime(df2["timestamp"]).dt.dayofweek
        if "day_of_week" in needed_cols and "day_of_week" not in df2.columns:
            df2["day_of_week"] = pd.to_datetime(df2["timestamp"]).dt.dayofweek
        if "month" in needed_cols and "month" not in df2.columns:
            df2["month"] = pd.to_datetime(df2["timestamp"]).dt.month

        # price lags
        lag_cols = [c for c in needed_cols if c.startswith("price_lag")]
        if lag_cols:
            max_lag = max(int(c.replace("price_lag", "")) for c in lag_cols if c.replace("price_lag", "").isdigit())
            for i in range(1, max_lag + 1):
                col = f"price_lag{i}"
                if col in needed_cols and col not in df2.columns:
                    df2[col] = df2[price_col].shift(i)

        # rolling price stats
        if "price_roll_mean_4" in needed_cols and "price_roll_mean_4" not in df2.columns:
            df2["price_roll_mean_4"] = df2[price_col].rolling(4).mean()
        if "price_roll_std_4" in needed_cols and "price_roll_std_4" not in df2.columns:
            df2["price_roll_std_4"] = df2[price_col].rolling(4).std()

        # alt name used in some scripts
        if "price_rolling_std" in needed_cols and "price_rolling_std" not in df2.columns:
            df2["price_rolling_std"] = df2[price_col].rolling(4).std()

        # price change
        if "price_change" in needed_cols and "price_change" not in df2.columns:
            df2["price_change"] = df2[price_col].diff()

        # MW lag + delta
        if "MW_lag1" in needed_cols and "MW_lag1" not in df2.columns:
            df2["MW_lag1"] = df2[mw_col].shift(1)
        if "MW_delta" in needed_cols and "MW_delta" not in df2.columns:
            if "MW_lag1" not in df2.columns:
                df2["MW_lag1"] = df2[mw_col].shift(1)
            df2["MW_delta"] = df2[mw_col] - df2["MW_lag1"]

        return df2

    # ---------------------------
    # 1) Load model feature lists + thresholds
    # ---------------------------
    market_model = xgb.XGBClassifier()
    market_model.load_model("Market Only/Volatility Classification/vol_spike_market_only_xgb.json")
    with open("Market Only/Volatility Classification/vol_spike_market_only_features.json") as f:
        market_features = json.load(f)
    market_dec_thresh = float(np.load("Market Only/Volatility Classification/vol_spike_market_only_decision_threshold.npy"))

    micro_model = xgb.XGBClassifier()
    micro_model.load_model("Full Feature/Volatility Classification/vol_spike_full_xgb.json")
    with open("Full Feature/Volatility Classification/vol_spike_full_features.json") as f:
        micro_features = json.load(f)
    micro_dec_thresh = float(np.load("Full Feature/Volatility Classification/vol_spike_full_decision_threshold.npy"))

    # union of required columns (so we build all missing engineered market columns once)
    required_cols = sorted(set(["simple_return"] + market_features + micro_features))

    # ---------------------------
    # 2) Build a model-ready dataframe with engineered columns
    # ---------------------------
    df_model = ensure_engineered_columns(df.copy(), required_cols)

    # Drop rows where required engineered cols are NA (lags/rolling)
    df_model = df_model.dropna(subset=[c for c in required_cols if c in df_model.columns]).copy()

    # ---------------------------
    # 3) Recreate vol-spike label exactly (train-only threshold)
    # ---------------------------
    split_idx = int(len(df_model) * 0.8)
    df_train = df_model.iloc[:split_idx].copy()
    df_test  = df_model.iloc[split_idx:].copy()

    vol_threshold = df_train["simple_return"].abs().quantile(0.99)

    for subset in (df_train, df_test):
        subset["spike_current"] = (subset["simple_return"].abs() > vol_threshold).astype(int)
        subset["spike_next"] = subset["spike_current"].shift(-1)

    df_test = df_test.dropna(subset=["spike_next"]).copy()
    df_test["actual"] = df_test["spike_next"].astype(int)

    # ---------------------------
    # 4) Final safety: check feature availability
    # ---------------------------
    miss_market = [c for c in market_features if c not in df_test.columns]
    miss_micro  = [c for c in micro_features  if c not in df_test.columns]

    if miss_market:
        st.error(f"Market model features missing from dashboard df: {miss_market}")
        st.stop()

    if miss_micro:
        st.error(
            "Full-feature model expects micro-signal columns not found in your dashboard CSV.\n"
            f"Missing: {miss_micro}\n\n"
            "Fix: point the dashboard to the SAME aggregated CSV you trained the full-feature model on "
            "(the one that includes slope/dev_mean/.../post_shock_var)."
        )
        st.stop()

    # ---------------------------
    # 5) Predict
    # ---------------------------
    X_market = df_test[market_features].copy()
    prob_market = market_model.predict_proba(X_market)[:, 1]
    pred_market = (prob_market > market_dec_thresh).astype(int)

    X_micro = df_test[micro_features].copy()
    prob_micro = micro_model.predict_proba(X_micro)[:, 1]
    pred_micro = (prob_micro > micro_dec_thresh).astype(int)

    # ---------------------------
    # 6) Build df for plots (slider window)
    # ---------------------------
    df_preds = pd.DataFrame({
        "timestamp": df_test["timestamp"],
        "actual": df_test["actual"],
        "prob_market": prob_market,
        "prob_micro": prob_micro,
        "pred_market": pred_market,
        "pred_micro": pred_micro,
    })

    df_preds = df_preds[
        (df_preds["timestamp"] >= start) &
        (df_preds["timestamp"] <= end)
    ].copy()

    # ---------------------------
    # 7) Plots
    # ---------------------------
    st.subheader("Predicted Probability (Overlay)")
    fig_prob = px.line(
        df_preds,
        x="timestamp",
        y=["prob_market", "prob_micro"],
        title="Volatility Spike Probability: Market vs Micro"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("Actual Volatility Spikes (Next Interval)")
    fig_actual = px.line(df_preds, x="timestamp", y="actual", title="Actual Volatility Spike (spike_next)")
    st.plotly_chart(fig_actual, use_container_width=True)

    st.subheader("Micro − Market Probability Difference")
    df_preds["prob_diff"] = df_preds["prob_micro"] - df_preds["prob_market"]
    fig_diff = px.line(df_preds, x="timestamp", y="prob_diff", title="Probability Difference (Micro - Market)")
    st.plotly_chart(fig_diff, use_container_width=True)

    # ---------------------------
    # 8) Metrics
    # ---------------------------
    st.subheader("Out-of-sample metrics (within selected window)")
    if df_preds["actual"].nunique() > 1:
        st.write("Market PR-AUC:", round(average_precision_score(df_preds["actual"], df_preds["prob_market"]), 4))
        st.write("Micro  PR-AUC:", round(average_precision_score(df_preds["actual"], df_preds["prob_micro"]), 4))
        st.write("Market ROC-AUC:", round(roc_auc_score(df_preds["actual"], df_preds["prob_market"]), 4))
        st.write("Micro  ROC-AUC:", round(roc_auc_score(df_preds["actual"], df_preds["prob_micro"]), 4))
    else:
        st.info("Not enough class variation in this time window to compute AUC/PR-AUC.")
