"""Runtime migration: saved forecasts, scheduled rates, metric definitions and SHAP."""
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import correlation, nearest_percentile
from model import create_lstm_model, create_callbacks, load_Model, OneCycleScheduler, load_checkpoint_weights, resolve_model_path
from shap_explain import get_custom_objects, make_scalar_output_model, normalize_shap_values, normalize_checkpoint_path


@pytest.mark.parametrize("decoder,mixed", [(False, False), (True, True)])
def test_saved_model_and_checkpoint_preserve_predictions(tmp_path, decoder, mixed):
	policy = keras.mixed_precision.global_policy()
	try:
		keras.mixed_precision.set_global_policy("mixed_float16" if mixed else "float32")
		keras.utils.set_random_seed(7)
		kwargs = dict(num_layers=1, n_units=4, dropout_rate=0.1, num_features=2,
			features_steps=6, num_labels=1, label_steps=3, noise_std=0.01, decoder=decoder)
		model = create_lstm_model(**kwargs)
		x = np.random.default_rng(7).normal(size=(4, 6, 2)).astype("float32")
		y = np.random.default_rng(8).normal(size=(4, 3, 1)).astype("float32")
		model.compile(optimizer="adam", loss="mse")
		model.train_on_batch(x, y)
		expected = model(x, training=False).numpy()
		path = tmp_path / "model.keras"
		model.save(path)
		for loaded in (load_Model(tmp_path), keras.models.load_model(path, compile=False, custom_objects=get_custom_objects())):
			np.testing.assert_allclose(loaded(x, training=False), expected, rtol=1e-5, atol=1e-6)
			assert loaded.get_layer("conditional_noise").stddev == 0.01
		callbacks, *_ = create_callbacks(str(tmp_path / "run"), 0, 1, None, -1, "val_loss", None,
			SimpleNamespace(num_train_samples=4, batch_size=4), logging.getLogger("test"))
		checkpoint = next(c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint))
		checkpoint.set_model(model)
		checkpoint.on_epoch_end(0, {"val_loss": 1.0})
		rebuilt = create_lstm_model(**kwargs)
		load_checkpoint_weights(rebuilt, normalize_checkpoint_path(checkpoint.filepath))
		np.testing.assert_allclose(rebuilt(x, training=False), expected, rtol=1e-5, atol=1e-6)
		wrong = create_lstm_model(**dict(kwargs, n_units=5))
		with pytest.raises(ValueError):
			load_checkpoint_weights(wrong, checkpoint.filepath)
	finally:
		keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("change", [dict(num_layers=1), dict(num_layers=3), dict(n_units=9),
	dict(batch_normalization=True), dict(decoder=True), dict(label_steps=2)])
def test_mismatched_checkpoint_is_rejected(tmp_path, change):
	"""A checkpoint must supply every model weight and nothing else. Keras alone accepts a
	checkpoint with extra layers, e.g. from a 2-layer model loaded into a 1-layer one."""
	kwargs = dict(num_layers=2, n_units=8, dropout_rate=0.0, num_features=3,
		features_steps=6, num_labels=1, label_steps=4, noise_std=0.01)
	path = str(tmp_path / "a.weights.h5")
	create_lstm_model(**kwargs).save_weights(path)
	with pytest.raises(ValueError, match="not in the"):
		load_checkpoint_weights(create_lstm_model(**dict(kwargs, **change)), path)
	load_checkpoint_weights(create_lstm_model(**dict(kwargs, noise_std=0.0)), path)  # the noise layer has no weights


def test_old_savedmodel_directory_gives_a_clear_error(tmp_path):
	(tmp_path / "saved_model.pb").write_bytes(b"")
	with pytest.raises(FileNotFoundError, match="Retrain"):
		resolve_model_path(tmp_path)
	(tmp_path / "saved_model.pb").unlink()
	with pytest.raises(FileNotFoundError, match="No model.keras"):
		resolve_model_path(tmp_path)
	assert resolve_model_path(tmp_path / "model.keras") == str(tmp_path / "model.keras")


@pytest.mark.parametrize("mixed", [False, True])
def test_schedule_updates_actual_optimizer_rate(mixed):
	optimizer = keras.optimizers.Adam()
	if mixed:
		optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
	scheduler = OneCycleScheduler(lr_max=0.005, epochs=1, steps_per_epoch=4)
	scheduler.set_model(SimpleNamespace(optimizer=optimizer))
	for step in range(9):
		scheduler.on_train_batch_begin(step)
		assert float(optimizer.learning_rate.numpy()) == pytest.approx(scheduler.calc_lr(step))


@pytest.mark.parametrize("q", [0, 25, 50, 75, 90, 95, 99, 100])
def test_percentiles_preserve_nearest_interpolation(q):
	# Half-index ties distinguish nearest (round-to-even) from linear interpolation.
	for n in (1, 3, 5, 21):
		x = np.arange(n, dtype="float64") ** 2
		expected = np.percentile(x, q, method="nearest")
		assert float(nearest_percentile(x, q)) == expected
		assert float(tf.function(nearest_percentile)(x, q)) == expected


def test_correlation_values_and_training_gradients():
	x = tf.constant([1., 4., 2., 8.])
	y = tf.Variable([2., 3., 1., 7.])
	with tf.GradientTape() as tape:
		r = correlation(x, y)
	assert float(r) == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-6)
	assert np.isfinite(tape.gradient(r, y).numpy()).all()
	assert np.isnan(correlation(x, tf.ones_like(x)))


def test_evaluation_metadata_from_archive_or_directory(tmp_path):
	import config
	from predictor_main import build_parser, evaluated_model_settings
	parser = build_parser()
	trained = parser.parse_args(["data.csv", "run", "--units-per-layer", "16", "--loss-function", "mae"])
	config.write_config(trained, tmp_path / "run_config.yaml")
	evaluate = parser.parse_args(["data.csv", "run", "--units-per-layer", "32", "--loss-function", "mse"])
	for path in (tmp_path, tmp_path / "model.keras"):
		record = evaluated_model_settings(evaluate, path)
		assert int(record.units_per_layer) == 16
		assert record.loss_function == "mae"


def test_search_resume_requires_matching_runtime(monkeypatch):
	import optuna
	import predictor_main as pm
	args = pm.build_parser().parse_args(["data.csv", "run"])
	settings = pm.tuning_settings(args, [])
	study = optuna.create_study()
	pm.check_study_settings(study, settings)
	study.tell(study.ask(), 0.5)
	pm.check_study_settings(study, settings)
	monkeypatch.setattr(pm.tf, "__version__", "different-runtime")
	with pytest.raises(ValueError, match="runtime"):
		pm.check_study_settings(study, pm.tuning_settings(args, []))
	assert study.user_attrs["settings"] == settings


def test_unversioned_existing_search_is_not_adopted():
	import optuna
	from predictor_main import check_study_settings
	study = optuna.create_study()
	study.tell(study.ask(), 0.5)
	with pytest.raises(ValueError, match="runtime"):
		check_study_settings(study, {"runtime": {"tensorflow": tf.__version__}})
	assert "settings" not in study.user_attrs


def test_gradient_shap_on_reloaded_lstm(tmp_path):
	import shap
	keras.utils.set_random_seed(9)
	model = create_lstm_model(1, 4, 0.0, 2, 6, 1, 3, noise_std=0.001)
	path = tmp_path / "model.keras"
	model.save(path)
	model = keras.models.load_model(path, compile=False, custom_objects=get_custom_objects())
	x = np.random.default_rng(9).normal(size=(4, 6, 2)).astype("float32")
	scalar = make_scalar_output_model(model, 1, 0)
	values = normalize_shap_values(shap.GradientExplainer(scalar, x[:2]).shap_values(x[2:], nsamples=20, rseed=9))
	assert values.shape == (2, 6, 2)
	assert np.isfinite(values).all()
	assert np.any(values != 0)
