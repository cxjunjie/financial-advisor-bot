import pandas as pd
from ta.momentum import RSIIndicator

FEATURE_COLS = ["sma_10", "sma_50", "rsi_14"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_50"] = out["Close"].rolling(50).mean()
    out["rsi_14"] = RSIIndicator(close=out["Close"], window=14).rsi()
    return out.dropna()

def latest_feature_row(df_feat: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([df_feat[FEATURE_COLS].iloc[-1]], columns=FEATURE_COLS)