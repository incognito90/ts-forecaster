"""WindowGenerator training windows: full shuffle, oversampling by index, float32."""
import collections
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling import WindowGenerator

IN_STEPS, OUT_STEPS, SHIFT = 5, 3, 0
ROWS = 60


def make_window(oversample_indices=None):
	rng = np.random.default_rng(0)
	def frame(n, start):
		return pd.DataFrame({
			"a": np.arange(start, start + n, dtype=float),     # row id: identifies each window
			"b": rng.normal(size=n),
			"x_Flow": rng.gamma(2.0, size=n),
		})
	train, val, test = frame(ROWS, 0), frame(30, 1000), frame(30, 2000)
	return WindowGenerator("t", IN_STEPS, OUT_STEPS, SHIFT, train, val, test, batch_size=8,
		label_columns=["x_Flow"], oversample_indices=oversample_indices,
		train_nan_indices={}, val_nan_indices={}, test_nan_indices={}, transform_method="none")


def window_ids(ds):
	"""Start row of every window in an unbatched or batched dataset (from column 'a')."""
	ids = []
	for x, _ in ds.unbatch() if len(ds.element_spec[0].shape) == 3 else ds:
		ids.append(int(x[0, 0]))
	return ids


def test_all_windows_present_float32_and_reshuffled():
	tf.keras.utils.set_random_seed(0)
	w = make_window()
	n_windows = ROWS - (IN_STEPS + SHIFT + OUT_STEPS) + 1
	epoch1, epoch2 = window_ids(w.train_ds), window_ids(w.train_ds)
	assert sorted(epoch1) == list(range(n_windows))
	assert sorted(epoch2) == list(range(n_windows))
	assert epoch1 != epoch2                     # new order every epoch
	assert epoch1 != list(range(n_windows))     # actually shuffled
	x, y = next(iter(w.train_ds))
	assert x.dtype == tf.float32 and y.dtype == tf.float32
	assert x.shape[1:] == (IN_STEPS, 3) and y.shape[1:] == (OUT_STEPS, 1)


def test_window_contents_match_the_ordered_construction():
	"""Each training window equals the window cut at the same start row by make_dataset."""
	w = make_window()
	ordered = {int(x[0, 0]): (x.numpy(), y.numpy()) for x, y in w.make_dataset(w.train_df, "train")}
	for x, y in w.train_ds.unbatch():
		ref_x, ref_y = ordered[int(x[0, 0])]
		np.testing.assert_array_equal(x.numpy(), ref_x)
		np.testing.assert_array_equal(y.numpy(), ref_y)


def test_oversampling_adds_one_copy_per_valid_index():
	extra = [3, 10, 10_000, -1]                 # last two are out of range and ignored
	w = make_window(oversample_indices=extra)
	counts = collections.Counter(window_ids(w.train_ds))
	assert counts[3] == 2 and counts[10] == 2
	assert all(c == 1 for i, c in counts.items() if i not in (3, 10))
	assert w.num_train_samples == ROWS + 2     # same accounting as before (rows + duplicates)


def test_val_and_test_keep_their_order():
	w = make_window()
	assert window_ids(w.test_ds) == sorted(window_ids(w.test_ds))
	assert window_ids(w.val_ds) == sorted(window_ids(w.val_ds))
