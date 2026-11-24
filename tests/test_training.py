"""Training callbacks: best-validation weights are always restored; early-stopping direction."""
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import BestEpochTracker, create_callbacks


def test_best_weights_restored_without_early_stopping():
	keras.utils.set_random_seed(3)
	x = np.random.normal(size=(256, 4)).astype("float32")
	y = (x @ np.array([1.0, -2.0, 0.5, 0.0], dtype="float32"))[:, None]
	xv, yv = x[:64] + 0.1, y[:64]
	model = keras.Sequential([keras.layers.Dense(8, activation="relu"), keras.layers.Dense(1)])
	model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")  # high rate: noisy val_loss
	tracker = BestEpochTracker(monitor="val_loss", mode="min")
	hist = model.fit(x, y, validation_data=(xv, yv), epochs=25, batch_size=32, verbose=0, callbacks=[tracker])

	val_losses = hist.history["val_loss"]
	assert tracker.best_epoch == int(np.argmin(val_losses))
	assert model.evaluate(xv, yv, verbose=0) == pytest.approx(min(val_losses), rel=1e-5)


@pytest.mark.parametrize("metric, better", [("val_loss", np.less), ("val_nse", np.greater)])
def test_early_stopping_direction_matches_metric(metric, better):
	callbacks, _, tracker, early_stopping, _ = create_callbacks(
		log_dir=None, min_epochs=1, max_epochs=5, learning_rate=None, patience=3,
		patience_metric=metric, grad_clip_norm=None,
		window=SimpleNamespace(num_train_samples=100, batch_size=10), logger=logging.getLogger("t"))
	# Check actual stopping behavior; Keras 3 creates monitor_op lazily and
	# uses backend operations instead of NumPy function objects.
	model = SimpleNamespace(stop_training=False)
	early_stopping.set_model(model)
	early_stopping.on_train_begin()
	values = [1.0, 0.5, 1.0, 1.0, 1.0] if better is np.less else [1.0, 2.0, 1.0, 1.0, 1.0]
	for epoch, value in enumerate(values, start=1):
		early_stopping.on_epoch_end(epoch, {metric: value})
		assert model.stop_training == (epoch == 5)
	assert tracker.mode == ("min" if better is np.less else "max")


def test_compile_and_fit_reports_restored_epoch_without_early_stopping():
	import pandas as pd
	from data_handling import WindowGenerator
	from model import compile_and_fit, create_lstm_model
	keras.utils.set_random_seed(0)
	rng = np.random.default_rng(0)
	def frame(n, start):
		t = np.arange(start, start + n)
		return pd.DataFrame({"rain": rng.gamma(1.0, size=n), "x_Label": np.sin(t / 5) + rng.normal(0, 0.05, n)})
	w = WindowGenerator("t", 8, 2, 0, frame(200, 0), frame(60, 200), frame(60, 260), batch_size=16,
		label_columns=["x_Label"], train_nan_indices={}, val_nan_indices={}, test_nan_indices={},
		transform_method="minmax")
	model = create_lstm_model(1, 8, 0.0, w.num_features, 8, 1, 2, noise_std=0.0)
	history, best_epoch, last_epoch, _ = compile_and_fit(model, w, patience=-1, max_epochs=3, min_epochs=1,
		grad_clip_norm=None, logger=logging.getLogger("t"))
	assert last_epoch == 2
	assert best_epoch == int(np.argmin(history.history["val_loss"]))


def test_optuna_pruning_reports_val_mse_not_the_trial_loss():
	"""Trials with different losses (MSE, MAE, 1 - NSE, ...) must be pruned on one common
	measure: the pruning report is val_mse (also the trial's final value), not val_loss."""
	from model import TUNING_METRIC
	class FakePruning:
		def __init__(self):
			self.reports = []
		def epoch_end(self, epoch, value):
			self.reports.append((epoch, value))
			return False
	pruning = FakePruning()
	callbacks, *_ = create_callbacks(
		log_dir=None, min_epochs=1, max_epochs=5, learning_rate=None, patience=-1,
		patience_metric="val_loss", grad_clip_norm=None,
		window=SimpleNamespace(num_train_samples=100, batch_size=10), logger=logging.getLogger("t"), pruning=pruning)
	optuna_callbacks = [c for c in callbacks if type(c).__name__ == "OptunaCallback"]
	assert len(optuna_callbacks) == 1
	optuna_callbacks[0].on_epoch_end(3, {"val_loss": 0.16, "val_mse": 0.00033, "val_mae": 0.005})
	assert TUNING_METRIC == "val_mse"
	assert pruning.reports == [(3, 0.00033)]
