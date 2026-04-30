TICKERS = {
    # ===== Equity =====
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "EUROSTOXX50": "^STOXX50E",
    "NIKKEI225": "^N225",
    "MSCI_EM": "EEM",      # iShares MSCI Emerging Markets ETF (replaces ^EEM)

    # ===== Volatility =====
    "VIX": "^VIX",

    # ===== Rates =====
    "RATE_3M": "^IRX",   # ÷10
    "RATE_10Y": "^TNX",  # ÷10
    "IEF": "IEF",        # 7–10Y US Treasury ETF

    # ===== FX =====
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    # "USDCNH": "USDCNH=X",

    # ===== Credit =====
    "HYG": "HYG",

    # ===== Commodities =====
    "COMMODITY": "^SPGSCI",
}

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

START = "1980-01-01"
# Use the most recent trading day (today or yesterday if weekend/holiday)
END = datetime.now().strftime("%Y-%m-%d")

print(f"Downloading data from {START} to {END}...")

prices = yf.download(
    list(TICKERS.values()),
    start=START,
    end=END,
    auto_adjust=True,
    progress=True
)


TICKER_RENAME = {
    "DX-Y.NYB": "DXY",
    "EURUSD=X": "EURUSD",
    "USDJPY=X": "USDJPY",
    "^GSPC": "SP500",
    "^IRX": "RATE_3M",
    "^N225": "NIKKEI225",
    "^NDX": "NASDAQ100",
    "^SPGSCI": "COMMODITY",
    "^STOXX50E": "EUROSTOXX50",
    "^TNX": "RATE_10Y",
    "^VIX": "VIX",
}
prices = prices.rename(columns=TICKER_RENAME, level="Ticker")

# prices = prices.sort_index().ffill()

prices.loc[:, (slice(None), "RATE_3M")] = prices.loc[:, (slice(None), "RATE_3M")] / 100
prices.loc[:, (slice(None), "RATE_10Y")] = prices.loc[:, (slice(None), "RATE_10Y")] / 100

prices.to_csv("data_yahoo.csv")
print(f"\nData saved to data_yahoo.csv")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")
print(f"Shape: {prices.shape}")

# prices[("Close", "YIELD_CURVE")] = prices[("Close", "RATE_10Y")] - prices[("Close", "RATE_3M")]

close_prices = prices["Close"]
returns = close_prices.pct_change().dropna()

features = pd.concat(
    [
        returns,
        returns.rolling(5).std().add_suffix("_vol5"),
        returns.rolling(20).std().add_suffix("_vol20"),
    ],
    axis=1
).dropna()

features.to_csv("market_features.csv")