import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from btc_forecast.data import DataError, fetch_daily, fingerprint, load_snapshot, save_snapshot, validate_daily
from tests.fixtures import daily_frame


class DataTests(unittest.TestCase):
    def test_invalid_data_is_rejected(self):
        good = daily_frame()
        cases = {"empty": good.iloc[:0], "short": good.iloc[:50], "gap": good.drop(index=50),
                 "duplicate": pd.concat([good, good.iloc[-1:]]), "column": good.drop(columns="Volume")}
        for name, field, value in [("negative", "Close", -1), ("nan", "Open", np.nan),
                                   ("infinity", "Volume", np.inf), ("bad_high", "High", 1),
                                   ("negative_volume", "Volume", -10), ("bad_time", "Datetime", pd.NaT)]:
            frame = good.copy()
            frame.loc[20, field] = value
            cases[name] = frame
        intraday = good.copy()
        intraday.Datetime += pd.Timedelta(hours=1)
        cases["intraday"] = intraday
        for name, frame in cases.items():
            with self.subTest(name=name), self.assertRaises(DataError):
                validate_daily(frame)

    def test_current_utc_candle_removed(self):
        frame = daily_frame()
        result = validate_daily(frame, now=frame.Datetime.iloc[-1] + pd.Timedelta(hours=12))
        self.assertEqual(len(result), len(frame) - 1)
        self.assertEqual(result.Datetime.iloc[-1], frame.Datetime.iloc[-2])

    def test_snapshot_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = save_snapshot(daily_frame(), directory, source="synthetic test")
            loaded = load_snapshot(directory=directory)
            self.assertEqual(saved.metadata["fingerprint"], fingerprint(loaded.frame))
            self.assertEqual(saved.metadata["source"], loaded.metadata["source"])
            df = pd.read_csv(saved.path)
            df.loc[30, "Volume"] += 1
            df.to_csv(saved.path, index=False)
            with self.assertRaisesRegex(DataError, "fingerprint"):
                load_snapshot(saved.path)

    def test_empty_download_never_substitutes_an_interval(self):
        with tempfile.TemporaryDirectory() as directory, patch("yfinance.Ticker") as ticker:
            ticker.return_value.history.return_value = pd.DataFrame()
            with self.assertRaises(DataError):
                fetch_daily(directory)
            self.assertEqual(ticker.return_value.history.call_count, 1)
            self.assertEqual(ticker.return_value.history.call_args.kwargs["interval"], "1d")
            self.assertFalse((Path(directory) / "latest.txt").exists())


if __name__ == "__main__":
    unittest.main()
