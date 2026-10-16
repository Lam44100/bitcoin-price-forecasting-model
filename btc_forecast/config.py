"""Shared, serializable experiment configuration."""
from dataclasses import asdict, dataclass
import hashlib
import json

MODELS = ("Last price", "Random Forest", "ARIMA", "LSTM")


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: int = 7
    lookback: int = 60
    seed: int = 42
    epochs: int = 12
    trees: int = 200
    models: tuple[str, ...] = MODELS

    def __post_init__(self):
        if self.horizon != 7 or self.lookback != 60:
            raise ValueError("This study uses a 60-day window and seven-day horizon.")
        if self.epochs < 1 or self.trees < 1 or self.seed < 0:
            raise ValueError("Epochs and trees must be positive; seed must be nonnegative.")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Select a nonempty set of unique models.")
        if set(self.models) - set(MODELS):
            raise ValueError("Unknown model requested.")

    def as_dict(self):
        return asdict(self)


def experiment_key(fingerprint: str, config: ExperimentConfig) -> str:
    payload = json.dumps({"data": fingerprint, "config": config.as_dict()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
