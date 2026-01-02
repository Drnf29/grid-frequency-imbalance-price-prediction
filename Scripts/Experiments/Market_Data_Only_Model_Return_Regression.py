import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

df = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_aggregated.csv",
    index_col=0,
    parse_dates=True,
)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

price_col = 'Price in €/MWh'
mw_col = 'Controlled output requirements in MW'

df_market = df[[price_col, mw_col]].copy()

df_market["simple_return"] = df_market[price_col].pct_change()

# Replace inf values caused by pct_change around zero
df_market["simple_return"] = df_market["simple_return"].replace([np.inf, -np.inf], np.nan)

for i in [1, 2, 3, 4]:
    df_market[f'return_lag{i}'] = df_market["simple_return"].shift(i)

df_market['return_roll_mean_4'] = df_market["simple_return"].rolling(4).mean()
df_market['return_roll_std_4']  = df_market["simple_return"].rolling(4).std()

df_market['MW_lag1']  = df_market[mw_col].shift(1)
df_market['MW_delta'] = df_market[mw_col] - df_market['MW_lag1']

df_market['hour'] = df.index.hour
df_market['dayofweek'] = df.index.dayofweek
df_market['month'] = df.index.month

df_market["target"] = df_market["simple_return"].shift(-1)

df_market = df_market.dropna().copy()

split_idx = int(len(df_market) * 0.8)

train = df_market.iloc[:split_idx]
test  = df_market.iloc[split_idx:]

X_train = train.drop(columns=[price_col, "simple_return", "target"])
y_train = train["target"]

X_test = test.drop(columns=[price_col, "simple_return", "target"])
y_test = test["target"]

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)
model.save_model("return_regression_market_only_xgb.json")

feature_cols = X_train.columns.tolist()
with open("return_regression_market_only_features.json", "w") as f:
    json.dump(feature_cols, f)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Layer B (Market-Only) Return Regression ===")
print("MAE :", round(mae, 6))
print("RMSE:", round(rmse, 6))
