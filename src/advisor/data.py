import pandas as pd
import yfinance as yf

def download_ohlcv(ticker: str, start="2020-01-01", end=None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for ticker '{ticker}'.")
    
    # Flatten the data
    if isinstance(df.columns, pd.MultiIndex):
    # Take only the first level (Open, High, Low, Close, Volume)
        df.columns = [col[0] for col in df.columns]

    df.index = pd.to_datetime(df.index)
    return df