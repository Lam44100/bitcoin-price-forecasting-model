"""Model adapters: fit past observations, forecast from a stated origin."""
from dataclasses import dataclass
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from .features import FEATURES, feature_frame, latest_inputs, training_samples


@dataclass
class Forecast:
    dates: pd.DatetimeIndex
    prices: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


class Adapter:
    def __init__(self, config, **params):
        self.config = config
        self.params = params

    def fit(self, history):
        self.fit_end = history.Datetime.iloc[-1]
        return self

    def pack(self, history, prices, lower=None, upper=None):
        prices = np.asarray(prices, dtype=float).reshape(-1)
        if len(prices) != self.config.horizon or not np.isfinite(prices).all():
            raise ValueError("Model produced a nonfinite or incomplete seven-day forecast.")
        dates = pd.date_range(history.Datetime.iloc[-1] + pd.Timedelta(days=1), periods=self.config.horizon, freq="D")
        return Forecast(dates, prices, lower, upper)


class LastPrice(Adapter):
    def forecast(self, history):
        return self.pack(history, np.repeat(history.Close.iloc[-1], self.config.horizon))


class RandomForest(Adapter):
    def fit(self, history):
        super().fit(history)
        samples = training_samples(history, self.config.lookback, self.config.horizon)
        self.model = RandomForestRegressor(n_estimators=self.config.trees,
            max_depth=self.params.get("max_depth", 12), min_samples_leaf=2,
            n_jobs=2, random_state=self.config.seed)
        self.model.fit(samples.tabular, samples.targets)
        return self

    def forecast(self, history):
        _, inputs = latest_inputs(history, self.config.lookback)
        return self.pack(history, self.model.predict(inputs)[0])


class LSTM(Adapter):
    def fit(self, history):
        super().fit(history)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf
        try:
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except RuntimeError:
            pass  # Runtime may already be initialized by another training run.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.config.seed)
        tf.config.experimental.enable_op_determinism()
        samples = training_samples(history, self.config.lookback, self.config.horizon)
        # Only observations supplied to fit() participate in either scaler.
        self.feature_scaler = MinMaxScaler().fit(feature_frame(history)[FEATURES].to_numpy())
        self.target_scaler = MinMaxScaler().fit(history[["Close"]].to_numpy())
        X = self.feature_scaler.transform(samples.sequences.reshape(-1, len(FEATURES))).reshape(samples.sequences.shape)
        y = self.target_scaler.transform(samples.targets.reshape(-1, 1)).reshape(samples.targets.shape)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.config.lookback, len(FEATURES))),
            tf.keras.layers.LSTM(self.params.get("units", 32)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(self.config.horizon),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        self.model.fit(X, y, epochs=self.config.epochs, batch_size=32, shuffle=False, verbose=0)
        return self

    def forecast(self, history):
        inputs, _ = latest_inputs(history, self.config.lookback)
        scaled = self.feature_scaler.transform(inputs.reshape(-1, len(FEATURES))).reshape(inputs.shape)
        prediction = self.model(scaled, training=False).numpy().reshape(-1, 1)
        return self.pack(history, self.target_scaler.inverse_transform(prediction).ravel())


class Arima(Adapter):
    def fit(self, history):
        super().fit(history)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model = ARIMA(history.Close.to_numpy(), order=tuple(self.params.get("order", (1, 1, 0)))).fit()
        self.fit_warnings = [str(w.message) for w in caught]
        if not self.model.mle_retvals.get("converged", True):
            raise ValueError("ARIMA optimizer did not converge for this order.")
        self.seen_end = self.fit_end
        return self

    def forecast(self, history):
        if history.Datetime.iloc[-1] < self.seen_end:
            raise ValueError("ARIMA origins must advance chronologically.")
        new = history.loc[history.Datetime > self.seen_end, "Close"]
        if len(new):
            # extend filters observations with fixed coefficients; it does not refit.
            self.model = self.model.extend(new.to_numpy())
            self.seen_end = history.Datetime.iloc[-1]
        prediction = self.model.get_forecast(steps=self.config.horizon)
        interval = np.asarray(prediction.conf_int(alpha=0.05))
        return self.pack(history, prediction.predicted_mean, interval[:, 0], interval[:, 1])


def candidates(name):
    return {
        "Last price": [{}],
        "Random Forest": [{"max_depth": 6}, {"max_depth": 12}],
        "LSTM": [{"units": 32}, {"units": 64}],
        "ARIMA": [{"order": order} for order in [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]],
    }[name]


def create_model(name, config, params=None):
    adapter = {"Last price": LastPrice, "Random Forest": RandomForest, "ARIMA": Arima, "LSTM": LSTM}[name]
    return adapter(config, **(params or {}))
