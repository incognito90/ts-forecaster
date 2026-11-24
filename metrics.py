"""Training losses and evaluation metrics (TensorFlow).

Every function takes (y_true, y_pred). Non-finite positions are dropped and both arrays are
flattened, so each score pools all remaining samples, horizons and labels.
Call as f(observed, simulated); note HydroErr (he.nse) takes (simulated, observed).
"""
import os
import logging
import tensorflow as tf
import keras


def correlation(x, y):
	"""Pearson correlation of paired 1-D tensors; undefined for constant series."""
	x = x - tf.reduce_mean(x)
	y = y - tf.reduce_mean(y)
	return tf.reduce_mean((x / tf.math.reduce_std(x)) * (y / tf.math.reduce_std(y)))


def nearest_percentile(values, q):
	"""Nearest interpolation, including round-to-even ties, as used by the scores."""
	values = tf.sort(tf.reshape(values, [-1]))
	n = tf.size(values)
	index = tf.cast(tf.round(tf.cast(n - 1, tf.float64) * (tf.cast(q, tf.float64) / 100.0)), tf.int32)
	return tf.cond(n > 0, lambda: tf.gather(values, tf.clip_by_value(index, 0, n - 1)),
		lambda: tf.constant(float("nan"), dtype=values.dtype))



def mask_invalid(y_true, y_pred):
	"""Drop positions where either input is NaN/inf; returns both as 1-D tensors."""
	mask = tf.math.logical_and(tf.math.is_finite(y_true), tf.math.is_finite(y_pred))
	return tf.boolean_mask(y_true, mask, axis=0), tf.boolean_mask(y_pred, mask, axis=0)


def cast_dtype(y_true, y_pred):
	# float32 also under mixed precision
	return tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)


def process_inputs(y_true, y_pred,*, cast=True):
	y_true, y_pred = mask_invalid(y_true, y_pred)
	if cast:
		y_true, y_pred = cast_dtype(y_true, y_pred)
	return y_true, y_pred

def to_scalar(val):
	if isinstance(val, tf.Tensor):
		if val.shape == ():  # scalar tensor
			return float(val.numpy()) if tf.executing_eagerly() else val
		else:
			raise ValueError(f"Expected scalar tensor, got shape {val.shape}")
	return val


def nse(y_true, y_pred,*, cast=True):
	"""NSE = 1 - sum((o-p)^2) / (sum((o-mean(o))^2) + eps)"""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)


	numerator = tf.reduce_sum(tf.square(y_true - y_pred))
	denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
	nse = 1 - numerator / (denominator + keras.config.epsilon())  # Add a small constant to prevent division by zero

	return nse

def nse_loss(y_true, y_pred):
	return 1 - nse(y_true, y_pred)


def kge_metric(y_true, y_pred,*, cast=True):
	"""KGE = 1 - sqrt((r-1)^2 + (std_p/std_o-1)^2 + (mean_p/mean_o-1)^2)"""
	# beta is 1 when mean(y_true) == 0. standard-scaled targets have a batch mean near 0,
	# which makes beta unstable; use minmax with --loss-function kge.
	# inputs are 1-D after masking, so the correlation needs event_axis=None

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)
	obs_mean = tf.reduce_mean(y_true)
	sim_mean = tf.reduce_mean(y_pred)
	obs_std = tf.math.reduce_std(y_true) + keras.config.epsilon()
	sim_std = tf.math.reduce_std(y_pred) + keras.config.epsilon()
	r = correlation(y_true, y_pred)
	beta = tf.math.divide_no_nan(sim_mean, obs_mean) + tf.cast(tf.equal(obs_mean, 0), sim_mean.dtype)
	kge = 1 - tf.sqrt((r - 1) ** 2 + (sim_std / obs_std - 1) ** 2 + (beta - 1) ** 2)

	return kge

def kge_loss(y_true, y_pred):
	return 1 - kge_metric(y_true, y_pred)

def kge_metric_components(y_true, y_pred,*, cast=True):
	"""Returns (kge, r, alpha, beta); alpha and beta are 1.0 when std/mean of y_true is 0."""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)


	obs_mean = tf.reduce_mean(y_true)
	sim_mean = tf.reduce_mean(y_pred)
	obs_sd = tf.math.reduce_std(y_true)
	obs_std = obs_sd + 1e-10
	sim_std = tf.math.reduce_std(y_pred) + 1e-10


	r = tf.cast(
		correlation(y_true, y_pred),
		y_true.dtype)
	beta = tf.cond(tf.not_equal(obs_mean, 0),
		lambda: sim_mean / obs_mean,
		lambda: tf.constant(1.0, dtype=y_true.dtype))
	alpha = tf.cond(tf.not_equal(obs_sd, 0),
		lambda: sim_std / obs_std,
		lambda: tf.constant(1.0, dtype=y_true.dtype))


	kge = 1 - tf.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

	if tf.executing_eagerly():
		# Convert to numpy for eager execution (like in testing or logging)
		return kge.numpy(), r.numpy(), alpha.numpy(), beta.numpy()
	else:
		# Return as is for graph execution (like in training)
		return kge, r, alpha, beta

def rve_metric(y_true, y_pred,*, cast=True):
	"""RVE = (sum(p) - sum(o)) / sum(o), a fraction; opposite sign to pbias."""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	total_observed = tf.reduce_sum(y_true)
	total_predicted = tf.reduce_sum(y_pred)
	rve = (total_predicted - total_observed) / total_observed
	if tf.executing_eagerly():
		return rve.numpy()
	else:
		return rve

def peak_flow_error(y_true, y_pred, threshold=95,*, cast=True):
	"""mean(|p - o|) where o >= percentile(o, threshold); NaN if there are no peaks."""
	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	percentile_value = nearest_percentile(y_true, threshold)
	peaks = y_true >= percentile_value
	errors = tf.boolean_mask(tf.abs(y_pred - y_true), peaks)
	error = tf.cond(tf.size(errors) > 0, lambda: tf.reduce_mean(errors), lambda: tf.constant(float("nan"), errors.dtype))
	if tf.executing_eagerly():
		return error.numpy()
	else:
		return error

def fdc_metric(y_true, y_pred,*, cast=True):
	"""Flow duration curve error: mean(|sort_desc(o) - sort_desc(p)|)"""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)
	# Sorting the values
	sorted_true = tf.sort(y_true, direction='DESCENDING')
	sorted_pred = tf.sort(y_pred, direction='DESCENDING')
	# Calculating FDC error (area between two curves)
	fdc_error = tf.reduce_mean(tf.abs(sorted_true - sorted_pred))
	if tf.executing_eagerly():
		return fdc_error.numpy()
	else:
		return fdc_error

def mae_metric(y_true, y_pred,*, cast=True):

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)
	if tf.executing_eagerly():
		# Convert to numpy for eager execution (like in testing or logging)
		return tf.reduce_mean(tf.abs(y_true - y_pred)).numpy()
	else:
		# Return as is for graph execution (like in training)
		return tf.reduce_mean(tf.abs(y_true - y_pred))

def mape_metric(y_true, y_pred,*, cast=True):
	"""100 * mean(|(o-p)/o|); inf/NaN if y_true contains zeros."""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)
	if tf.executing_eagerly():
		# Convert to numpy for eager execution (like in testing or logging)
		return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)).numpy() * 100
	else:
		# Return as is for graph execution (like in training)
		return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

def pbias_metric(y_true, y_pred,*, cast=True):
	"""PBIAS = 100 * (sum(o) - sum(p)) / sum(o); positive = under-prediction."""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	total_prediction = tf.reduce_sum(y_pred)
	total_observation = tf.reduce_sum(y_true)
	pbias = 100 * (total_observation - total_prediction) / total_observation
	if tf.executing_eagerly():
		# Convert to numpy for eager execution (like in testing or logging)
		return pbias.numpy()
	else:
		# Return as is for graph execution (like in training)
		return pbias

def model_efficiency_metric(y_true, y_pred,*, cast=True):
	"""Same as nse() but without the epsilon."""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	# https://www.sciencedirect.com/science/article/pii/S002216940500206X
	# https://www.researchgate.net/publication/222529929_Evaluating_the_performance_of_hydrological_models_II_Hydrological_model_performance_metrics
	if tf.executing_eagerly():
		# Convert to numpy for eager execution (like in testing or logging)
		return (1 - (tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))))).numpy()
	else:
		# Return as is for graph execution (like in training)
		return 1 - (tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))))

def peak_prediction_accuracy(y_true, y_pred, threshold,*, cast=True):
	"""Returns (recall, fdr) for exceedances of percentile(o, threshold).

	recall is 0 if there are no observed peaks, fdr is NaN if there are no predicted peaks.
	"""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)
	# Determining the threshold value for peaks
	percentile_value = nearest_percentile(y_true, threshold)

	# Identifying peaks
	true_peaks = tf.greater_equal(y_true, percentile_value)
	predicted_peaks = tf.greater_equal(y_pred, percentile_value)

	# Calculating recall (TPR) and false discovery rate (FDR)
	true_peaks_sum = tf.reduce_sum(tf.cast(true_peaks, tf.float32))
	predicted_peaks_sum = tf.reduce_sum(tf.cast(predicted_peaks, tf.float32))

	hits = tf.reduce_sum(tf.cast(tf.logical_and(predicted_peaks, true_peaks), tf.float32))
	false_positives = tf.reduce_sum(tf.cast(tf.logical_and(predicted_peaks, tf.logical_not(true_peaks)), tf.float32))


	recall = tf.where(true_peaks_sum > 0, hits / true_peaks_sum, tf.constant(0.0, dtype=tf.float32))
	fdr = tf.where(predicted_peaks_sum > 0, false_positives / predicted_peaks_sum, tf.constant(float('nan'), dtype=tf.float32))


	if tf.executing_eagerly():
		return recall.numpy(), fdr.numpy()
	else:
		return recall, fdr


def mse_loss(y_true, y_pred,*, cast=True):
	"""MSE for the results CSV; a non-finite result becomes 9999.0."""

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)


	mse = tf.reduce_mean(tf.square(y_true - y_pred))

	return tf.where(tf.math.is_finite(mse), mse, 9999.0)


def msle_loss(y_true, y_pred,*, cast=True):

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	msle = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
	if tf.executing_eagerly():
		# Convert to numpy for eager execution (like in testing or logging)
		return msle.numpy()
	else:
		# Return as is for graph execution (like in training)
		return msle

def rmse_metric(y_true, y_pred,*, cast=True):

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))

def rmse(y_true, y_pred,*, cast=True):

	y_true, y_pred = process_inputs(y_true, y_pred, cast=cast)

	return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


if __name__ == "__main__":
	logging.error(os.path.basename(__file__) + " has no main.")

