"""Causal features and direct multi-horizon targets; no future imputation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES = ["Close", "Returns", "Volatility", "MA20", "RSI"]
RF_FEATURES = FEATURES + ["MACD", "BBWidth", "VolumeChange", "Weekday"]


def feature_frame(df):
    close = df["Close"]
    returns = close.pct_change(fill_method=None)
    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    ma = close.rolling(20).mean()
    result = pd.DataFrame({
        "Close": close, "Returns": returns, "Volatility": returns.rolling(20).std(),
        "MA20": ma, "RSI": rsi,
        "MACD": close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean(),
        "BBWidth": 4 * close.rolling(20).std() / ma,
        # log1p handles zero-volume days without division by zero.
        "VolumeChange": np.log1p(df["Volume"]).diff(),
        "Weekday": pd.to_datetime(df["Datetime"]).dt.dayofweek / 6,
    }, index=df.index)
    return result.replace([np.inf, -np.inf], np.nan).dropna()


@dataclass
class Samples:
    sequences: np.ndarray
    tabular: np.ndarray
    targets: np.ndarray
    origins: np.ndarray
    target_positions: np.ndarray


def training_samples(df, lookback=60, horizon=7):
    features = feature_frame(df)
    sequences, tabular, targets, origins, positions = [], [], [], [], []
    for j in range(lookback - 1, len(features)):
        origin = int(features.index[j])
        if origin + horizon >= len(df):
            break
        window = features.iloc[j - lookback + 1:j + 1]
        sequences.append(window[FEATURES].to_numpy())
        tabular.append(np.r_[window.Close.to_numpy(), features.iloc[j][RF_FEATURES].to_numpy()])
        targets.append(df.Close.iloc[origin + 1:origin + horizon + 1].to_numpy())
        origins.append(origin)
        positions.append(np.arange(origin + 1, origin + horizon + 1))
    if not sequences:
        raise ValueError("Insufficient history after feature warm-up, lookback, and seven-day targets.")
    return Samples(np.asarray(sequences), np.asarray(tabular), np.asarray(targets),
                   np.asarray(origins), np.asarray(positions))


def latest_inputs(history, lookback=60):
    features = feature_frame(history)
    if len(features) < lookback or features.index[-1] != history.index[-1]:
        raise ValueError("Latest history has insufficient complete feature rows.")
    window = features.iloc[-lookback:]
    return (window[FEATURES].to_numpy()[None, ...],
            np.r_[window.Close.to_numpy(), window.iloc[-1][RF_FEATURES].to_numpy()][None, ...])
