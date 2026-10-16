import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from btc_forecast.app import sync_result_state
from btc_forecast.config import ExperimentConfig
from btc_forecast.data import DataError, ROOT
from btc_forecast.evaluation import run_experiment
from tests.fixtures import snapshot


def widget(elements, label):
    return next(element for element in elements if element.label == label)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = snapshot()
        cls.result = run_experiment(cls.data, ExperimentConfig(epochs=1, trees=5, models=("Last price", "Random Forest")))

    def test_no_data_startup_does_not_download(self):
        with patch("btc_forecast.app.load_snapshot", side_effect=DataError("No saved snapshot")), patch("btc_forecast.app.fetch_daily") as fetch:
            app = AppTest.from_file(str(ROOT / "BTCPredictionModel.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("No saved snapshot" in x.value for x in app.info))
            fetch.assert_not_called()

    def test_presentation_reruns_preserve_result_config_change_invalidates(self):
        with patch("btc_forecast.app.load_snapshot", return_value=self.data), patch("btc_forecast.app.matching_run", return_value=None), patch("btc_forecast.app.run_experiment", return_value=self.result) as runner, patch("btc_forecast.app.save_result"):
            app = AppTest.from_file(str(ROOT / "BTCPredictionModel.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            widget(app.button, "Run benchmark").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertIn("result", app.session_state)
            widget(app.multiselect, "Chart overlays").set_value(["Bollinger bands"]).run()
            self.assertFalse(app.exception)
            self.assertIn("result", app.session_state)
            self.assertEqual(runner.call_count, 1)
            widget(app.multiselect, "Models").set_value(["Random Forest"]).run()
            self.assertFalse(app.exception)
            self.assertNotIn("result", app.session_state)
            self.assertEqual(runner.call_count, 1)

    def test_data_fingerprint_change_invalidates_result(self):
        state = {"result": object(), "result_key": "old"}
        self.assertTrue(sync_result_state(state, "new"))
        self.assertNotIn("result", state)

    def test_refresh_failure_keeps_previous_snapshot_explicitly(self):
        with patch("btc_forecast.app.load_snapshot", return_value=self.data), patch("btc_forecast.app.matching_run", return_value=None), patch("btc_forecast.app.fetch_daily", side_effect=DataError("Download failed")):
            app = AppTest.from_file(str(ROOT / "BTCPredictionModel.py")).run()
            widget(app.button, "Refresh daily data").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("previous snapshot" in x.value for x in app.warning))
            self.assertEqual(app.session_state.snapshot.metadata["fingerprint"], self.data.metadata["fingerprint"])


if __name__ == "__main__":
    unittest.main()
