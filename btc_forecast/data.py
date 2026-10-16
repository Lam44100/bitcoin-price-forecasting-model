"""Validated completed UTC candles and immutable, fingerprinted local snapshots."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "cache"
MIN_ROWS = 240


class DataError(ValueError):
    """An actionable data validation or retrieval failure."""


@dataclass
class Snapshot:
    frame: pd.DataFrame
    metadata: dict
    path: Path | None = None


def validate_daily(frame: pd.DataFrame, *, now=None, min_rows=MIN_ROWS) -> pd.DataFrame:
    if frame.empty:
        raise DataError("No candles returned. Refresh later or load a saved daily snapshot.")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise DataError(f"Missing columns: {', '.join(sorted(missing))}.")
    df = frame[COLUMNS].copy()
    try:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataError("Datetime must contain valid UTC daily timestamps.") from exc
    if df["Datetime"].isna().any():
        raise DataError("Datetime contains missing timestamps.")
    if (df["Datetime"] != df["Datetime"].dt.normalize()).any():
        raise DataError("Expected midnight UTC daily candles; intraday data is unsupported.")
    if df["Datetime"].duplicated().any():
        raise DataError("Duplicate daily timestamps found. Resolve duplicates at the source.")
    today = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    today = today.tz_localize("UTC") if today.tzinfo is None else today.tz_convert("UTC")
    if (df["Datetime"] > today.normalize()).any():
        raise DataError("Data contains future-dated candles.")
    df = df.loc[df["Datetime"] < today.normalize()].sort_values("Datetime").reset_index(drop=True)
    if len(df) < min_rows:
        raise DataError(f"Need at least {min_rows} completed daily candles; received {len(df)}.")
    if not df["Datetime"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise DataError("Missing daily candles found. Refresh or supply a continuous daily snapshot; gaps are not filled.")
    for col in COLUMNS[1:]:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise DataError(f"{col} contains nonnumeric values.") from exc
    if not np.isfinite(df[COLUMNS[1:]].to_numpy()).all():
        raise DataError("OHLCV contains missing or infinite values; no prices have been imputed.")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any() or (df["Volume"] < 0).any():
        raise DataError("Prices must be positive and volume nonnegative.")
    if (df["High"] < df[["Open", "Low", "Close"]].max(axis=1)).any() or (df["Low"] > df[["Open", "High", "Close"]].min(axis=1)).any():
        raise DataError("Invalid OHLC candle: low/high do not enclose open and close.")
    return df


def canonical_csv(df: pd.DataFrame) -> str:
    return df[COLUMNS].to_csv(index=False, float_format="%.10f", date_format="%Y-%m-%dT%H:%M:%SZ", lineterminator="\n")


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_csv(df).encode()).hexdigest()


def save_snapshot(frame, directory=DEFAULT_CACHE, *, source="Yahoo Finance via yfinance", now=None):
    df = validate_daily(frame, now=now)
    digest = fingerprint(df)
    fetched = pd.Timestamp.now(tz="UTC")
    metadata = {"symbol": "BTC-USD", "interval": "1d", "timezone": "UTC", "source": source,
                "fetched_at": fetched.isoformat(), "start": df.Datetime.iloc[0].isoformat(),
                "end": df.Datetime.iloc[-1].isoformat(), "rows": len(df), "fingerprint": digest}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"btc-usd-{fetched.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:10]}"
    path = directory / f"{stem}.csv"
    path.write_text(canonical_csv(df), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary = directory / f".latest-{uuid.uuid4().hex}.tmp"
    temporary.write_text(path.name, encoding="utf-8")
    temporary.replace(directory / "latest.txt")
    return Snapshot(df, metadata, path)


def load_snapshot(path=None, directory=DEFAULT_CACHE):
    directory = Path(directory)
    if path is None:
        pointer = directory / "latest.txt"
        if not pointer.exists():
            raise DataError("No saved snapshot. Use Refresh daily data while online, or import a daily CSV.")
        filename = pointer.read_text(encoding="utf-8").strip()
        if Path(filename).name != filename:
            raise DataError("Invalid local cache pointer.")
        path = directory / filename
    path = Path(path)
    try:
        df = validate_daily(pd.read_csv(path, float_precision="round_trip"))
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {
            "source": "User-supplied CSV (provenance not verified)", "symbol": "BTC-USD",
            "interval": "1d", "fetched_at": None}
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot load snapshot: {exc}") from exc
    digest = fingerprint(df)
    if metadata.get("fingerprint", digest) != digest:
        raise DataError("Snapshot fingerprint mismatch. The CSV or its metadata has changed.")
    metadata.update(fingerprint=digest, rows=len(df), start=df.Datetime.iloc[0].isoformat(), end=df.Datetime.iloc[-1].isoformat())
    return Snapshot(df, metadata, path)


def fetch_daily(directory=DEFAULT_CACHE):
    # No synthetic data, interval substitution, or automatic stale-data fallback.
    import yfinance as yf
    try:
        frame = yf.Ticker("BTC-USD").history(period="5y", interval="1d", auto_adjust=False, timeout=20)
        frame = frame.reset_index().rename(columns={"Date": "Datetime"})
        return save_snapshot(frame, directory)
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Yahoo daily download failed: {exc}. Retry later or load a saved snapshot.") from exc
