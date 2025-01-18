import pandas as pd
import ta

df = pd.DataFrame({
    "Open": [1, 2, 3, 4, 5],
    "High": [2, 3, 4, 5, 6],
    "Low": [1, 1, 2, 3, 4],
    "Close": [2, 3, 4, 3, 5],
    "Volume": [100, 200, 300, 400, 500]
})

df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
print(df.head())