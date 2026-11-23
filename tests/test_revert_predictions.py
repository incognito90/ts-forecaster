"""revertPredictions: vectorised inverse scaling equals the original per-sample loop."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor_main import revertPredictions


def original_revert(window, predictions):
	"""Reference implementation of the inverse scaling."""
	preds = np.empty((0, window.label_steps, window.num_labels), dtype=float)
	for pred in predictions:
		full_array = np.zeros((window.label_steps, window.num_features))
		full_array[:, list(window.label_columns_indices.values())] = pred
		for col in window.columns_to_scale:
			if col in window.label_columns_indices:
				i = window.label_columns_indices[col]
				full_array[:, i] = np.squeeze(window.scalers[col].inverse_transform(full_array[:, i].reshape(-1, 1)))
		pred = full_array[:, list(window.label_columns_indices.values())]
		preds = np.vstack((preds, [pred]))
	return preds


@pytest.mark.parametrize("scaler_cls", [StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer])
def test_matches_original_loop(scaler_cls):
	rng = np.random.default_rng(0)
	labels = ["a_Flow", "b_Flow"]
	scalers = {c: scaler_cls().fit(rng.gamma(2.0, 10.0, size=(500, 1))) for c in labels}
	window = SimpleNamespace(label_steps=4, num_labels=2, num_features=5,
		label_columns_indices={c: i for i, c in enumerate(labels)},
		columns_to_scale=["rain", "temp", "x"] + labels, scalers=scalers)
	preds = rng.normal(size=(50, 4, 2))
	np.testing.assert_allclose(revertPredictions(window, preds), original_revert(window, preds), rtol=0, atol=1e-12)


def test_no_scalers_returns_predictions_unchanged():
	window = SimpleNamespace(label_steps=2, num_labels=1, label_columns_indices={"a_Flow": 0})
	preds = np.arange(6.0).reshape(3, 2, 1)
	np.testing.assert_array_equal(revertPredictions(window, preds), preds)
