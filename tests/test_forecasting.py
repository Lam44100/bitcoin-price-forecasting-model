import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from btc_forecast.config import ExperimentConfig, experiment_key
from btc_forecast.evaluation import evaluation_origins, load_result, run_experiment, save_result, split_bounds
from btc_forecast.features import FEATURES, feature_frame, latest_inputs, training_samples
from btc_forecast.models import Arima, LSTM, LastPrice, RandomForest
from tests.fixtures import daily_frame, snapshot


class ForecastingTests(unittest.TestCase):
    def test_causal_features_and_targets(self):
        frame = daily_frame()
        train_end, validation_end, _ = split_bounds(len(frame))
        changed = frame.copy()
        changed.loc[train_end:, ["Open", "High", "Low", "Close", "Volume"]] *= 100
        pd.testing.assert_frame_equal(feature_frame(frame.iloc[:train_end]), feature_frame(changed).loc[:train_end - 1])
        a = training_samples(frame.iloc[:train_end])
        b = training_samples(changed.iloc[:train_end])
        np.testing.assert_array_equal(a.sequences, b.sequences)
        np.testing.assert_array_equal(a.targets, b.targets)
        self.assertLess(a.target_positions.max(), train_end)
        for origin, positions, targets in zip(a.origins, a.target_positions, a.targets):
            np.testing.assert_array_equal(positions, np.arange(origin + 1, origin + 8))
            np.testing.assert_array_equal(targets, frame.Close.iloc[positions])
        origins = evaluation_origins(validation_end, len(frame))
        self.assertEqual(origins[0], validation_end - 1)
        self.assertTrue(all(b - a == 7 for a, b in zip(origins, origins[1:])))
        self.assertLess(origins[-1] + 7, len(frame))

    def test_latest_inputs_end_at_latest_observation(self):
        frame = daily_frame()
        sequences, tabular = latest_inputs(frame)
        self.assertEqual(sequences.shape, (1, 60, len(FEATURES)))
        self.assertEqual(sequences[0, -1, 0], frame.Close.iloc[-1])
        self.assertEqual(tabular[0, 59], frame.Close.iloc[-1])

    def test_flat_prices_and_zero_volume_have_finite_features(self):
        frame = daily_frame()
        frame[["Open", "High", "Low", "Close"]] = 100
        frame.Volume = 0
        features = feature_frame(frame)
        self.assertGreater(len(features), 0)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertTrue((features.RSI == 50).all())

    def test_models_forecast_smoke_and_lstm_training_scalers(self):
        frame = daily_frame(240)
        train = frame.iloc[:180]
        config = ExperimentConfig(epochs=1, trees=5)
        for cls in [LastPrice, RandomForest, Arima, LSTM]:
            with self.subTest(model=cls.__name__):
                model = cls(config).fit(train)
                future = model.forecast(frame)
                self.assertEqual(len(future.prices), 7)
                self.assertTrue(np.isfinite(future.prices).all())
                self.assertEqual(future.dates[0], frame.Datetime.iloc[-1] + pd.Timedelta(days=1))
                if cls is LSTM:
                    np.testing.assert_allclose(model.feature_scaler.data_max_, feature_frame(train)[FEATURES].max())
                    self.assertEqual(model.target_scaler.data_max_[0], train.Close.max())

    def test_arima_updates_state_without_refitting_or_noise(self):
        frame = daily_frame()
        model = Arima(ExperimentConfig(), order=(1, 1, 0)).fit(frame.iloc[:250])
        params = model.model.params.copy()
        before = model.forecast(frame.iloc[:250])
        repeated = model.forecast(frame.iloc[:250])
        np.testing.assert_array_equal(before.prices, repeated.prices)
        model.forecast(frame.iloc[:280])
        np.testing.assert_array_equal(params, model.model.params)
        again = model.forecast(frame.iloc[:280])
        native = model.model.get_forecast(steps=7)
        np.testing.assert_allclose(again.prices, native.predicted_mean)
        np.testing.assert_allclose(again.lower, np.asarray(native.conf_int())[:, 0])
        np.testing.assert_allclose(again.upper, np.asarray(native.conf_int())[:, 1])
        with self.assertRaises(ValueError):
            model.forecast(frame.iloc[:270])

    def test_pipeline_matched_targets_and_saved_manifest(self):
        data = snapshot()
        config = ExperimentConfig(epochs=1, trees=5, models=("Last price", "Random Forest", "ARIMA"))
        result = run_experiment(data, config)
        baseline = result.predictions[result.predictions.model == "Last price"]
        for _, group in result.predictions.groupby("model"):
            pd.testing.assert_frame_equal(group[["origin", "target_date", "actual", "horizon"]].reset_index(drop=True),
                baseline[["origin", "target_date", "actual", "horizon"]].reset_index(drop=True))
        expected = np.abs(baseline.actual - baseline.prediction).mean()
        actual = result.metrics.query("model == 'Last price' and horizon == 'Overall'").MAE.iloc[0]
        self.assertAlmostEqual(expected, actual)
        self.assertTrue((result.forecasts.origin == data.frame.Datetime.iloc[-1]).all())
        self.assertEqual(result.manifest["experiment_key"], experiment_key(data.metadata["fingerprint"], config))
        with tempfile.TemporaryDirectory() as directory:
            path = save_result(result, data, directory)
            loaded = load_result(path, result.manifest["experiment_key"])
            self.assertEqual(loaded.manifest["selected_parameters"], json.loads(json.dumps(result.manifest["selected_parameters"])))
            np.testing.assert_allclose(loaded.predictions.prediction, result.predictions.prediction)
            with self.assertRaises(ValueError):
                load_result(path, "wrong-key")


if __name__ == "__main__":
    unittest.main()
