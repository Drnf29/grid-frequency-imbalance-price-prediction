import pandas as pd
import numpy as numpy

features = pd.read_csv(
    "../../Aggregated Data/germany_2012_2016_15min_features.csv",
    index_col=0
)

features.index = pd.to_datetime(features.index)
features.index.name = "Timestamp"

rebap = pd.read_csv(
    "../../Aggregated Data/balanceGroupDeviation_2012_2016.csv",
    sep=";"
)

rebap["Timestamp"] = pd.to_datetime(rebap["Date"] + " " + rebap["Start Time"])
rebap = rebap.set_index("Timestamp")
rebap = rebap.drop(columns=["Date", "Start Time", "End Time"])

merged = features.join(rebap, how="inner")

merged.to_csv("germany_2012_2016_aggregated.csv")
