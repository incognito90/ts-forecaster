"""Scientific-package upgrades must keep each advertised gap-filling path usable."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nan_imputer import NaNImputer


@pytest.mark.parametrize("method", [
	"ffill", "mean", "median", "median_mask", "zero", "interpolate", "min_std", "min_std_2", "min_std_4",
	"interpolate_time", "interpolate_polynomial", "interpolate_spline", "mean_previous_next",
	"rolling_mean_24", "rolling_mean_12", "rolling_mean_6", "knn", "mice", "kalman", "missforest",
	"missforest_singleton_minstd", "kalman_singleton_minstd",
])
def test_imputation_preserves_observations_and_fills_gaps(method):
	t = np.arange(100)
	data = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=100, freq="h"),
		"a": 10 + np.sin(t / 4), "b": 30 + np.cos(t / 7)})
	data.loc[[5, 10, 11, 12, 82, 85, 86, 87], "a"] = np.nan
	data.loc[[6, 15, 16, 17, 83, 88, 89], "b"] = np.nan
	original = data.copy(deep=True)
	imputer = NaNImputer(method=method, time_index_column="Date", mf_rf_verbose=0, mf_verbose=0)
	imputer.fit(data.iloc[:80])
	out = imputer.transform(data.iloc[80:])
	assert np.isfinite(out[["a", "b"]].to_numpy()).all()
	pd.testing.assert_series_equal(out["Date"], data["Date"].iloc[80:])
	for col in ("a", "b"):
		observed = data[col].iloc[80:].dropna()
		np.testing.assert_array_equal(out.loc[observed.index, col], observed)
	if method.endswith("_singleton_minstd"):
		expected_marker = data["a"].iloc[:80].min() - data["a"].iloc[:80].std()
		np.testing.assert_allclose(out.loc[[85, 86, 87], "a"], expected_marker)
	pd.testing.assert_frame_equal(data, original)
