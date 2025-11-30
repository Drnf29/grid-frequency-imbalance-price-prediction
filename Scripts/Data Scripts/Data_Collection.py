import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

years = ['2012','2013','2014','2015','2016']

def compute_features(window_df):

    dev = window_df["Deviation"].dropna().values / 1000.0
    N = len(dev)

    if N == 0:
        return None

    # Guard for flat signals 
    if np.std(dev) == 0:
        return None

    t = np.arange(N)

    slope = np.polyfit(t, dev, 1)[0]
    dev_mean = dev.mean()
    dev_min = dev.min()
    dev_max = dev.max()

    mild_mask = np.abs(dev) > 0.02
    deep_mask = np.abs(dev) > 0.05
    mild_excursions = mild_mask.sum()
    deep_excursions = deep_mask.sum()

    var = dev.var()
    skewness = skew(dev)
    kurt_val = kurtosis(dev)

    counts, _ = np.histogram(dev, bins=20)
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()


    rocof = np.diff(dev)
    max_abs_rocof = np.max(np.abs(rocof))
    mean_abs_rocof = np.mean(np.abs(rocof))
    rocof_std = rocof.std()
    rocof_shock_count = (np.abs(rocof) > 0.02).sum()

    shock_idx = np.argmin(dev)
    shock_depth = -dev[shock_idx]

    recovery_time = None
    for i in range(shock_idx, N):
        if abs(dev[i]) <= 0.02:
            recovery_time = i - shock_idx
            break

    if recovery_time is None:
        recovery_time = N

    post_end = min(shock_idx + 60, N)
    post_shock_var = dev[shock_idx:post_end].var()

    return {
        "slope": slope,
        "dev_mean": dev_mean,
        "dev_min": dev_min,
        "dev_max": dev_max,
        "mild_excursions": mild_excursions,
        "deep_excursions": deep_excursions,
        "var": var,
        "skewness": skewness,
        "kurtosis": kurt_val,
        "entropy": entropy,
        "max_abs_rocof": max_abs_rocof,
        "mean_abs_rocof": mean_abs_rocof,
        "rocof_std": rocof_std,
        "rocof_shock_count": rocof_shock_count,
        "shock_depth": shock_depth,
        "recovery_time": recovery_time,
        "post_shock_var": post_shock_var
    }


all_features = []

for year in years:
    for i in range(1, 13):

        path = f"../../Grid Frequency Data/germany_{year}_{i:02d}.csv.zip"

        try:
            df = pd.read_csv(path, index_col=0)
            print("Opened file:", path)
        except FileNotFoundError:
            print("Missing file:", path)
            continue

        df.index = pd.to_datetime(df.index)
        df.index.name = "Timestamp"

        df = df.rename(columns={"Frequency": "Deviation"})
        df["Frequency"] = 50.0 + df["Deviation"] / 1000.0

        # ---- MANUAL 15-MINUTE LOOP ----
        start = df.index.min()
        end = df.index.max()

        t = start

        while t < end:
            t_end = t + pd.Timedelta(minutes=15)

            window = df.loc[t:t_end]

            if len(window) == 0:
                t = t_end
                continue

            feats = compute_features(window)

            if feats is not None:
                feats["Timestamp"] = t
                all_features.append(feats)

            t = t_end


# Convert to final dataframe
features_15m_all = pd.DataFrame(all_features)
features_15m_all.set_index("Timestamp", inplace=True)
features_15m_all.sort_index(inplace=True)

features_15m_all.to_csv("germany_2012_2016_15min_features.csv")

print("DONE.")
