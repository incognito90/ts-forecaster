"""Missing target values: labels are NaN (skipped by the loss), change targets are
anchored on the last observed value, and the inputs keep the imputed values."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling import WindowGenerator
from model import LOSS_MAPPING, masked

IN_STEPS, OUT_STEPS = 4, 2
FILL = -5.0                          # stands in for the min_std fill value


def make_window(predict, gaps):
	"""Train/val/test of 30 rows each; `gaps` = test rows whose target was missing."""
	flow = 10.0 + np.arange(90, dtype=float)
	frames, nan_idx = [], []
	for k, start in enumerate((0, 30, 60)):
		f = pd.DataFrame({"rain": np.ones(30), "x_Flow": flow[start:start + 30].copy()})
		rows = gaps if k == 2 else []
		f.loc[rows, "x_Flow"] = FILL                   # imputed value seen by the model
		frames.append(f)
		nan_idx.append({"x_Flow": pd.Index(rows, dtype=int)})
	raw = flow[60:90].copy()
	raw[gaps] = np.nan
	w = WindowGenerator("t", IN_STEPS, OUT_STEPS, 0, *frames, batch_size=4, label_columns=["x_Flow"],
		train_nan_indices=nan_idx[0], val_nan_indices=nan_idx[1], test_nan_indices=nan_idx[2],
		transform_method="none", raw_test_labels=raw[:, None], train_label_means=np.array([30.0]),
		predict=predict)
	return w, raw


def windows(ds):
	xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds.unbatch()])
	return np.stack(xs), np.stack(ys)


def test_missing_labels_are_nan_and_inputs_keep_the_fill():
	gaps = [5, 6]
	w, raw = make_window("level", gaps)
	x, y = windows(w.test_ds)
	# window i has labels at rows i+4, i+5 and inputs at rows i..i+3
	assert np.isnan(y[1, 0, 0]) and np.isnan(y[1, 1, 0]) and np.isnan(y[2, 0, 0])
	assert y[0, 0, 0] == raw[4]
	assert x[3, 2, 1] == FILL                         # input row 5 keeps the imputed value


def test_change_target_is_anchored_on_the_last_observed_value():
	gaps = [5, 6]
	w, raw = make_window("change", gaps)
	assert w.change_scale[0] == pytest.approx(np.std(np.r_[np.ones(29), 2 * np.ones(28)]), rel=1e-6)
	_, y = windows(w.test_ds)
	# window 3: last input row 6 is missing; the anchor is row 4 (value raw[4])
	labels = y[3, :, 0] * w.change_scale[0]
	np.testing.assert_allclose(labels, [raw[7] - raw[4], raw[8] - raw[4]], rtol=1e-6)
	# outputs_to_level adds the same anchor back: a perfect change forecast gives the observations
	level = w.outputs_to_level(y)
	np.testing.assert_allclose(level[3, :, 0], [raw[7], raw[8]], rtol=1e-6)
	# persistence uses the same last observed value
	_, last_observed = w.observed_test_windows()
	assert last_observed[3, 0] == raw[4]


def test_masked_loss_skips_missing_labels():
	y_true = tf.constant([[[1.0], [np.nan]], [[3.0], [5.0]]])
	y_pred = tf.constant([[[2.0], [100.0]], [[3.0], [3.0]]])
	assert float(LOSS_MAPPING["mse"](y_true, y_pred)) == pytest.approx((1 + 0 + 4) / 3)
	assert float(LOSS_MAPPING["mae"](y_true, y_pred)) == pytest.approx((1 + 0 + 2) / 3)
	assert float(masked(keras.losses.mean_squared_error, "x")(tf.fill([2, 2, 1], np.nan), y_pred)) == 0.0
	with tf.GradientTape() as tape:
		p = tf.Variable(y_pred)
		loss = LOSS_MAPPING["huber"](y_true, p)
	grad = tape.gradient(loss, p).numpy()
	assert np.all(np.isfinite(grad)) and grad[0, 1, 0] == 0.0


def test_training_runs_with_missing_labels():
	from model import create_lstm_model
	w, _ = make_window("change", [5, 6])
	m = create_lstm_model(1, 4, 0.0, w.num_features, IN_STEPS, w.num_outputs, OUT_STEPS)
	m.compile(optimizer="adam", loss=LOSS_MAPPING["mse"], metrics=[LOSS_MAPPING["mae"]])
	h = m.fit(w.train, validation_data=w.test, epochs=1, verbose=0)
	assert np.isfinite(h.history["loss"][0]) and np.isfinite(h.history["val_loss"][0])
