"""create_lstm_model: output shapes and size of the direct and encoder-decoder heads."""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MODEL_BUILDERS, create_lstm_model

FEATURES, LABELS, OUT_STEPS, UNITS = 13, 1, 12, 128


def build(input_steps, decoder):
	return create_lstm_model(2, UNITS, 0.25, FEATURES, input_steps, LABELS, OUT_STEPS,
		noise_std=0.0005, decoder=decoder)


@pytest.mark.parametrize("decoder", [False, True])
def test_output_shape(decoder):
	model = build(168, decoder)
	assert model.output_shape == (None, OUT_STEPS, LABELS)
	out = model(np.zeros((3, 168, FEATURES), dtype="float32"))
	assert out.shape == (3, OUT_STEPS, LABELS)


def test_direct_head_is_unchanged():
	"""Default model: same layers and parameter count as the reference architecture."""
	model = build(168, decoder=False)
	assert [l.__class__.__name__ for l in model.layers] == [
		"InputLayer", "ConditionalNoise", "LSTM", "LSTM", "Dropout", "Dense", "Reshape"]
	lstm1 = 4 * ((FEATURES + UNITS) * UNITS + UNITS)
	lstm2 = 4 * ((UNITS + UNITS) * UNITS + UNITS)
	dense = UNITS * OUT_STEPS * LABELS + OUT_STEPS * LABELS
	assert model.count_params() == lstm1 + lstm2 + dense


def test_decoder_size_does_not_depend_on_input_length():
	assert build(168, decoder=True).count_params() == build(2190, decoder=True).count_params()
	added = build(168, True).count_params() - build(168, False).count_params()
	decoder_lstm = 4 * ((UNITS + UNITS) * UNITS + UNITS)
	per_step_dense = UNITS * LABELS + LABELS
	direct_dense = UNITS * OUT_STEPS * LABELS + OUT_STEPS * LABELS
	assert added == decoder_lstm + per_step_dense - direct_dense


def test_registry_contains_lstm():
	assert MODEL_BUILDERS == {"lstm": create_lstm_model}


def test_layers_start_from_different_weights():
	"""Same-shaped LSTM layers must not share initial kernels (Keras reuses values
	when one unseeded initializer instance is used for several layers)."""
	model = create_lstm_model(3, 16, 0.1, 16, 24, 1, 6, noise_std=0.0)
	lstm_kernels = [l.get_weights()[0] for l in model.layers if l.__class__.__name__ == "LSTM"]
	assert lstm_kernels[1].shape == lstm_kernels[2].shape
	assert not np.allclose(lstm_kernels[1], lstm_kernels[2])


@pytest.mark.parametrize("decoder", [False, True])
def test_mixed_precision_regression_loss_stays_float32(decoder):
	policy = tf.keras.mixed_precision.global_policy()
	floatx = tf.keras.backend.floatx()
	try:
		tf.keras.backend.set_floatx("float16")
		tf.keras.mixed_precision.set_global_policy("mixed_float16")
		model = create_lstm_model(1, 4, 0.0, 1, 3, 1, 2, noise_std=0.0, decoder=decoder)
		model.compile(optimizer="adam", loss="mse")
		with tf.GradientTape() as tape:
			pred = model(tf.zeros((2, 3, 1), dtype=tf.float32))
			loss = model.compiled_loss(tf.fill((2, 2, 1), 300.0), pred)
		assert pred.dtype == tf.float32 and loss.dtype == tf.float32
		assert float(loss) == pytest.approx(90000.0)  # exceeds float16's range
		gradients = tape.gradient(loss, model.trainable_weights)
		assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
		assert any(l.compute_dtype == "float16" for l in model.layers if isinstance(l, tf.keras.layers.LSTM))
	finally:
		tf.keras.backend.set_floatx(floatx)
		tf.keras.mixed_precision.set_global_policy(policy)
