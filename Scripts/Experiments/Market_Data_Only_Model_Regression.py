import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from experiment_common import (
    PRICE_COL,
    build_market_only_price_frame,
    keep_contiguous_rows,
    load_aggregated_data,
    save_feature_list,
    split_train_test,
)

df = load_aggregated_data()
df_market = build_market_only_price_frame(df)
df_market = df_market.dropna()
df_market["target"] = df_market[PRICE_COL].shift(-1)
df_market = keep_contiguous_rows(df_market, prev_steps=4, next_steps=1)
df_market = df_market.dropna()   # remove final NaN target row

train, test = split_train_test(df_market, train_fraction=0.8)

X_train = train.drop(columns=[PRICE_COL, "target"])
y_train = train["target"]

X_test = test.drop(columns=[PRICE_COL, "target"])
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
model.save_model("price_regression_market_only_xgb.json")

feature_cols = X_train.columns.tolist()
save_feature_list("price_regression_market_only_features.json", feature_cols)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Layer B (Market-Only) Regression ===")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
