"""Synthetic OHLCV for tests only; never used by application data loading."""
import numpy as np
import pandas as pd

from btc_forecast.data import Snapshot, fingerprint


def daily_frame(n=360):
    rng = np.random.default_rng(42)
    close = 20000 * np.exp(np.cumsum(rng.normal(0.001, 0.012, n)))
    return pd.DataFrame({"Datetime": pd.date_range("2020-01-01", periods=n, tz="UTC"),
        "Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(10000, 50000, n).astype(float)})


def snapshot(n=360):
    frame = daily_frame(n)
    return Snapshot(frame, {"source": "SYNTHETIC TEST FIXTURE — NOT MARKET DATA", "symbol": "BTC-USD",
        "interval": "1d", "fingerprint": fingerprint(frame), "fetched_at": "2021-01-01T00:00:00+00:00"})
