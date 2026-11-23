"""remove_empty_columns: uninformative features are dropped, targets are never dropped silently."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling import remove_empty_columns

N_TRAIN, N_VAL, N_TEST = 70, 15, 15


def frame(**overrides):
	rng = np.random.default_rng(0)
	n = N_TRAIN + N_VAL + N_TEST
	df = pd.DataFrame({"rain": rng.gamma(1.0, size=n), "x_Label": rng.normal(10, 1, size=n)})
	for name, values in overrides.items():
		df[name] = values
	return df


def test_feature_constant_in_validation_is_dropped():
	values = np.r_[np.arange(N_TRAIN, dtype=float), np.full(N_VAL, 5.0), np.arange(N_TEST, dtype=float)]
	df = remove_empty_columns(frame(dam=values), [], ["x_Label"], N_TRAIN, N_VAL, N_TEST)
	assert "dam" not in df.columns and "x_Label" in df.columns


def test_target_constant_in_validation_raises():
	values = np.r_[np.arange(N_TRAIN, dtype=float), np.full(N_VAL, 5.0), np.arange(N_TEST, dtype=float)]
	with pytest.raises(ValueError, match="Label column .x_Label."):
		remove_empty_columns(frame(x_Label=values), [], ["x_Label"], N_TRAIN, N_VAL, N_TEST)


def test_feature_with_zero_variance_in_one_period_is_dropped():
	rng = np.random.default_rng(1)
	values = np.r_[rng.normal(size=N_TRAIN), rng.normal(size=N_VAL), 3.0 + 1e-9 * rng.normal(size=N_TEST)]
	df = remove_empty_columns(frame(sensor=values), [], ["x_Label"], N_TRAIN, N_VAL, N_TEST)
	assert "sensor" not in df.columns
