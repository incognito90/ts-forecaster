"""Model definition and training utilities for ts-forecaster.

Stacked LSTM with a direct (or encoder-decoder) multi-horizon head,
input (batch, input_steps, features) -> output (batch, output_steps, labels),
plus the losses, callbacks and training loop used by predictor_main.py.
"""

import os
import tempfile
import time
import datetime
import logging
import tensorflow as tf
import keras
import h5py
from typing import Union, Optional # for type hinting in function definition arguments
import numpy as np
import matplotlib.pyplot as plt

from metrics import nse, rmse, nse_loss, kge_loss

# print a progress line every heartbeat_interval training batches
heartbeat_flag = True
heartbeat_interval = 100 

# const mapping used for initializing layers
INITIALIZER_MAPPING = {
	"zeros": keras.initializers.Zeros,
	"ones": keras.initializers.Ones,
	"random_normal": keras.initializers.RandomNormal,
	"random_uniform": keras.initializers.RandomUniform,
	"glorot_normal": keras.initializers.GlorotNormal,
	"glorot_uniform": keras.initializers.GlorotUniform,
	"he_normal": keras.initializers.HeNormal,
	"he_uniform": keras.initializers.HeUniform
}

@keras.utils.register_keras_serializable(package="ts_forecaster")
class ConditionalNoise(keras.layers.Layer):
	"""Gaussian noise on non-zero inputs only, during training (identity at inference)."""
	def __init__(self, stddev, **kwargs):
		name = kwargs.pop('name', 'noise_on_non_zero')
		super(ConditionalNoise, self).__init__(**kwargs, name=name)
		self.stddev = stddev

	def call(self, inputs, training=None):
		if training:
			noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=inputs.dtype)
			# Only add noise to non-zero entries
			return inputs + tf.where(tf.equal(inputs, 0), tf.zeros_like(inputs), noise)
		return inputs

	def get_config(self):
		return dict(super().get_config(), stddev=self.stddev)

def get_lr_metric(optimizer): #for printing learning rate during training
	def lr(y_true, y_pred):
		return optimizer.learning_rate
	return lr


def resolve_model_path(path):
	"""A run's model directory or its model.keras file -> the Keras archive."""
	path = os.fspath(path)
	if not os.path.isdir(path):
		return path
	archive = os.path.join(path, "model.keras")
	if not os.path.exists(archive):
		if os.path.exists(os.path.join(path, "saved_model.pb")):
			raise FileNotFoundError(f"{path} is a TensorFlow SavedModel from an older version; "
				"this version reads only .keras models. Retrain the model.")
		raise FileNotFoundError(f"No model.keras in {path}.")
	return archive


def _weights_layout(path):
	"""(name, shape) of every model array in a .weights.h5 file (not the optimizer state saved in training)."""
	layout = []
	with h5py.File(path, "r") as f:
		f.visititems(lambda name, obj: layout.append((name, obj.shape)) if isinstance(obj, h5py.Dataset) else None)
	return {(name, shape) for name, shape in layout if not name.startswith("optimizer/")}


def load_checkpoint_weights(model, path):
	"""Load a .weights.h5 checkpoint, raising ValueError unless it has exactly the model's weights.

	Keras only checks the weights the model asks for: a checkpoint with extra layers (e.g. from a
	model with more LSTM layers) would otherwise load without an error.
	"""
	with tempfile.TemporaryDirectory() as tmp:
		own = os.path.join(tmp, "model.weights.h5")
		model.save_weights(own)
		expected = _weights_layout(own)
	found = _weights_layout(path)
	if found != expected:
		raise ValueError(f"{len(expected - found)} weight arrays of the model are not in the checkpoint "
			f"and {len(found - expected)} arrays of the checkpoint are not in the model "
			f"(e.g. {sorted(expected ^ found)[:3]}).")
	model.load_weights(path)


def load_Model(filename, learning_rate=None):
	"""Load a saved model with the custom objects it needs.

	It is recompiled with the masked MSE whatever loss it was trained with (only used for prediction).
	"""
	print("Model Loading from: ", filename)
	if learning_rate is None or learning_rate == -1:
		opt = keras.optimizers.Adam()
	else:
		opt = keras.optimizers.Adam(learning_rate=learning_rate)
	lr_metric = get_lr_metric(opt)

	model = keras.models.load_model(
		resolve_model_path(filename),
		custom_objects={"lr": lr_metric, "ConditionalNoise": ConditionalNoise, "rmse": rmse, "nse": nse, "kge_loss": kge_loss, "nse_loss": nse_loss},
		compile=False
	)

	model.compile(loss=LOSS_MAPPING["mse"],
		optimizer=opt,
		metrics=[LOSS_MAPPING["mse"], lr_metric])

	model.summary()
	return model


# Model: LSTM-MIMO (Direct Multi-Horizon / "LSTM-MLP")
# Stacked LSTM with direct multi-horizon dense head. (aka multi-output direct forecasting).
# Summary:
#	Stacked LSTM encoder over inputs X ∈ R^{B×T_in×F}. Final LSTM returns the last hidden state (B,H).
#	A linear head maps (B,H) → (B, L*C) and reshapes to (B, L, C) for direct multi-horizon outputs.
#	No autoregression; predicts all horizons jointly (MIMO).
# Notable choices:
#	• Linear output (activation=None) for regression; use target transforms (e.g., log1p) for non-negativity.
#	• LSTM kernel init: glorot_uniform; recurrent init: orthogonal.
#	• Optional LayerNorm after intermediate LSTMs (prefer over BatchNorm for RNN stability).
# Input/Output:
#	Inputs: (B, features_steps=T_in, num_features=F)
#	Outputs: (B, label_steps=L, num_labels=C)
# Standard names:
#	"LSTM-MIMO", "Direct multi-step LSTM", "Encoder-only LSTM (direct)"
# References:
#	Glorot & Bengio (2010): https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
#	Kratzert et al. (2018): https://hess.copernicus.org/articles/22/6005/2018/
#	Kratzert et al. (2019): https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019WR026065
#	Hyndman (multi-step strategies): https://robjhyndman.com/papers/rectify.pdf

def create_lstm_model(num_layers: int,
					  n_units: int,
					  dropout_rate: float,
					  num_features: int,
					  features_steps: int,
					  num_labels: int,
					  label_steps: int,
					  weight_init: str = "glorot_uniform",
					  batch_normalization: Optional[bool] = False,
					  l1_regularization: Union[float, None] = None,
					  l2_regularization: Union[float, None] = None,
					  recurrent_l1_regularization: Union[float, None] = None,
					  recurrent_l2_regularization: Union[float, None] = None,
					  dense_l1_regularization: Union[float, None] = None,
					  dense_l2_regularization: Union[float, None] = None,
					  noise_std: float = 0.001,
					  decoder: bool = False,
					  input_dropout: float = 0.0,
					  logger=None):
	"""Stacked LSTM, input (B, features_steps, num_features) -> output (B, label_steps, num_labels).

	Args:
		dropout_rate: input dropout of every LSTM layer after the first (no recurrent dropout),
			and of the Dropout layer before the output head.
		input_dropout: input dropout of the first LSTM layer. Zeroes random input values, including
			the latest target value (0 = training minimum under minmax), which biases forecasts
			toward the mean.
		batch_normalization: adds LayerNormalization (not BatchNorm) after each intermediate LSTM.
		noise_std: ConditionalNoise stddev; no noise layer if <= 1e-10.
		decoder: encoder-decoder head instead of the direct Dense head.
	"""

	logger = logger or logging.getLogger()  # Use global logger if none is passed

	logger.info(f"Creating LSTM model with\n\t{num_layers} layers\n\t{n_units} units\n\t{dropout_rate} dropout rate\n\t{num_features} features\n\t{features_steps} feature steps\n\t{num_labels} labels\n\t{label_steps} label steps\n\t{weight_init} weight init\n\t{batch_normalization} batch normalization\n\t{l1_regularization} l1 regularization\n\t{l2_regularization} l2 regularization\n\t{recurrent_l1_regularization} recurrent l1 regularization\n\t{recurrent_l2_regularization} recurrent l2 regularization\n\t{dense_l1_regularization} dense l1 regularization\n\t{dense_l2_regularization} dense l2 regularization\n\t{noise_std} noise std\n\tdecoder {decoder}\n\t{input_dropout} input dropout (first layer)")
	#set up LSTM args
	lstm_args = {
		'units': n_units,
		'return_sequences': True,
		'kernel_initializer': INITIALIZER_MAPPING[weight_init](),
		'recurrent_initializer': 'orthogonal',
		'dropout': dropout_rate,
	}
	
	# Set up kernel_regularizer
	if l1_regularization is not None and l2_regularization is not None: #elastic net
		lstm_args['kernel_regularizer'] = keras.regularizers.l1_l2(l1=l1_regularization, l2=l2_regularization)
	elif l1_regularization is not None: #l1 aka lasso
		lstm_args['kernel_regularizer'] = keras.regularizers.l1(l1_regularization)
	elif l2_regularization is not None: #l2 aka ridge
		lstm_args['kernel_regularizer'] = keras.regularizers.l2(l2_regularization)

	# Set up recurrent_regularizer
	if recurrent_l1_regularization is not None and recurrent_l2_regularization is not None: #elastic net
		lstm_args['recurrent_regularizer'] = keras.regularizers.l1_l2(l1=recurrent_l1_regularization, l2=recurrent_l2_regularization)
	elif recurrent_l1_regularization is not None: #l1 aka lasso
		lstm_args['recurrent_regularizer'] = keras.regularizers.l1(recurrent_l1_regularization)
	elif recurrent_l2_regularization is not None: #l2 aka ridge
		lstm_args['recurrent_regularizer'] = keras.regularizers.l2(recurrent_l2_regularization)


	dense_args = {
		'units': num_labels * label_steps,
		'kernel_initializer': INITIALIZER_MAPPING[weight_init](),
		'activation': None,
		# float32 output so outputs and losses stay float32 under mixed_float16
		'dtype': 'float32',
	}


	if dense_l1_regularization is not None and dense_l2_regularization is not None: #elastic net
		dense_args['kernel_regularizer'] = keras.regularizers.l1_l2(l1=dense_l1_regularization, l2=dense_l2_regularization)
	elif dense_l1_regularization is not None: #l1 aka lasso
		dense_args['kernel_regularizer'] = keras.regularizers.l1(dense_l1_regularization)
	elif dense_l2_regularization is not None: #l2 aka ridge
		dense_args['kernel_regularizer'] = keras.regularizers.l2(dense_l2_regularization)

	# new initializer instance per layer: a reused unseeded instance gives every
	# same-shaped layer the same starting weights
	def fresh_init(layer_args):
		return dict(layer_args, kernel_initializer=INITIALIZER_MAPPING[weight_init]())

	inputs = keras.Input(shape=(features_steps, num_features), name='input_layer')

	if(noise_std > 1e-10):
		x = ConditionalNoise(stddev=noise_std,name="conditional_noise")(inputs)
	else:
		x = inputs

	def layer_args(i):
		# first layer reads the input features
		return dict(fresh_init(lstm_args), dropout=input_dropout if i == 0 else dropout_rate)

	for i in range(num_layers-1):
		x = keras.layers.LSTM(**layer_args(i), name=f"lstm_{i+1}_units_{n_units}")(x)
	
		if batch_normalization: # replace with layer norm?
			x = keras.layers.LayerNormalization(name=f'layer_norm_{i+1}')(x)


	lstm_args['return_sequences'] = False

	if not decoder:
		# direct: final hidden state -> one Dense for all label_steps x num_labels outputs
		x = keras.layers.LSTM(**layer_args(num_layers - 1), name=f"lstm_{num_layers}_units_{n_units}")(x)
		x = keras.layers.Dropout(dropout_rate, name=f'final_dropout')(x)
		x = keras.layers.Dense(**dense_args, name='dense')(x)
		x = keras.layers.Reshape((label_steps, num_labels), name='reshape', dtype='float32')(x)
	else:
		# encoder-decoder: final hidden state repeated per output step, decoder LSTM starts
		# from the encoder states, per-step Dense
		x, state_h, state_c = keras.layers.LSTM(**layer_args(num_layers - 1), return_state=True, name=f"encoder_lstm_{num_layers}_units_{n_units}")(x)
		x = keras.layers.Dropout(dropout_rate, name='final_dropout')(x)
		x = keras.layers.RepeatVector(label_steps, name='repeat_output_steps')(x)
		decoder_args = dict(lstm_args, return_sequences=True, kernel_initializer=INITIALIZER_MAPPING[weight_init]())
		x = keras.layers.LSTM(**decoder_args, name=f"decoder_lstm_units_{n_units}")(x, initial_state=[state_h, state_c])
		step_dense_args = dict(dense_args, units=num_labels, kernel_initializer=INITIALIZER_MAPPING[weight_init]())
		x = keras.layers.TimeDistributed(keras.layers.Dense(**step_dense_args), name='dense_per_step', dtype='float32')(x)

	model = keras.models.Model(inputs=inputs, outputs=x, name='lstm_model')

	return model


# --model-type builders: same arguments as create_lstm_model, must return an uncompiled
# model with output (batch, label_steps, num_labels)
MODEL_BUILDERS = {
	"lstm": create_lstm_model,
}


class VRAMUsageCallback(keras.callbacks.Callback):
	"""Peak GPU memory (bytes) over the training run."""
	def __init__(self):
		super(VRAMUsageCallback, self).__init__()
		self.peak_vram_usage = 0  # Track peak memory usage

	def on_train_begin(self, logs=None):
		# reset, otherwise every Optuna trial reports the peak since the process started
		# (approximate with parallel trials, the counter is shared)
		if tf.config.list_physical_devices('GPU'):
			tf.config.experimental.reset_memory_stats('GPU:0')

	def on_epoch_end(self, epoch, logs=None):
		"""Check GPU memory usage at the end of each epoch."""
		if tf.config.list_physical_devices('GPU'):
			vram_info = tf.config.experimental.get_memory_info('GPU:0')
			current_peak = vram_info['peak']
			self.peak_vram_usage = max(self.peak_vram_usage, current_peak)  # Store highest value


	def get_peak_vram_usage(self):
		"""Return the highest VRAM usage recorded."""
		return self.peak_vram_usage

class OneCycleScheduler(keras.callbacks.Callback):
	"""Triangular cyclical learning rate (Smith 2017), set at every batch.

	Rises linearly from lr_min (default lr_max / 10) to lr_max over half a cycle, falls back over
	the other half, then repeats. Half-cycle = epochs * steps_per_epoch (minus anneal_epochs; there
	is no separate annealing phase).
	"""
	def __init__(self, lr_max, epochs, steps_per_epoch, lr_min=None, anneal_epochs=0):
		super().__init__()
		if lr_min is None:
			lr_min = lr_max / 10
		self.lr_max = lr_max
		self.lr_min = lr_min
		self.anneal_epochs = anneal_epochs
		self.total_steps = epochs * steps_per_epoch
		self.cycle_length = 2 * max(self.total_steps - anneal_epochs * steps_per_epoch, 1)
		self.lrates = []
		self._global_step = 0

	def on_train_begin(self, logs=None):
		self._global_step = 0
		
	def on_train_batch_begin(self, batch, logs=None):
		lr = self.calc_lr(self._global_step)
		self.lrates.append(lr)
		opt = self.model.optimizer
		opt.learning_rate.assign(lr)
		self._global_step += 1
	
	def calc_lr(self, step):
		cycle_position = step % self.cycle_length
		if cycle_position < self.cycle_length // 2:
			# First half of the cycle
			lr = self.lr_min + (self.lr_max - self.lr_min) * (cycle_position / (self.cycle_length // 2))
		else:
			# Second half of the cycle
			lr = self.lr_max - (self.lr_max - self.lr_min) * ((cycle_position - self.cycle_length // 2) / (self.cycle_length // 2))
		return lr
	


class LoggingCallback(keras.callbacks.Callback):
	"""Per-epoch summary for the job logs (printed to stdout)."""
	def __init__(self, logger, vram_callback=None, best_epoch_tracker=None, patience=None, patience_metric="val_loss", early_stopping=None):
		super().__init__()
		self.logger = logger or logging.getLogger()
		self.vram_callback = vram_callback  # Capture VRAM usage
		self.best_epoch_tracker = best_epoch_tracker  # Track best epoch
		self.patience = patience if patience is not None else 0  # Patience for early stopping
		self.early_stopping = early_stopping
		self.patience_metric = patience_metric  # Metric to monitor improvement
		self.epoch_start_time = None

	def on_epoch_begin(self, epoch, logs=None):
		"""Capture start time for each epoch."""
		self.epoch_start_time = time.time()
		print(f"[HB] Starting epoch {epoch+1} at {datetime.datetime.fromtimestamp(self.epoch_start_time)}", flush=True)


	def on_epoch_end(self, epoch, logs=None):
		"""Logs epoch details with standardized format."""
		logs = logs or {}

		# Capture key training and validation metrics
		metrics = {
			"loss": logs.get("loss", "N/A"),
			"mse": logs.get("mse", "N/A"),
			"mae": logs.get("mae", "N/A"),
			"rmse": logs.get("rmse", "N/A"),
			"nse": logs.get("nse", "N/A"),
			"val_loss": logs.get(self.patience_metric, "N/A"),
			"val_mse": logs.get("val_mse", "N/A"),
			"val_mae": logs.get("val_mae", "N/A"),
			"val_rmse": logs.get("val_rmse", "N/A"),
			"val_nse": logs.get("val_nse", "N/A"),
		}

		def fmt(value, spec=".6f"):
			try:
				return format(value, spec)
			except (TypeError, ValueError):
				return str(value)

		# Compute epoch time
		epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else "N/A"

		# Determine if `val_loss` improved
		best_epoch = self.best_epoch_tracker.best_epoch if self.best_epoch_tracker else None
		best_value = self.best_epoch_tracker.best_value if self.best_epoch_tracker else None

		# Mirror EarlyStopping's internal patience counter when available
		patience_used = None
		patience_total = self.patience
		early_stop_prefix = ""

		if self.early_stopping is not None and self.patience and self.patience > 0:
			patience_used = getattr(self.early_stopping, "wait", None)
			patience_total = getattr(self.early_stopping, "patience", self.patience)
			start_from_epoch = getattr(self.early_stopping, "start_from_epoch", 0)

			# Explicitly indicate warm-up period where EarlyStopping is not active yet
			if epoch < start_from_epoch:
				early_stop_prefix = f"(warm-up: EarlyStopping starts at epoch {start_from_epoch+1}) "

		if best_epoch is not None and best_epoch == epoch:
			val_loss_status = f"Improved (New Best: {metrics['val_loss']})"
		else:
			if patience_used is not None:
				val_loss_status = (
					f"Did not improve ({patience_used}/{patience_total}) . "
					f"{metrics['val_loss']} vs Epoch {best_epoch+1 if best_epoch is not None else 'N/A'}: {best_value}"
				)
			else:
				# Fallback to old behaviour if EarlyStopping is not used
				bad_epochs_since_best = epoch - best_epoch if best_epoch is not None else 0
				val_loss_status = (
					f"Did not improve ({bad_epochs_since_best}/{self.patience}) . "
					f"{metrics['val_loss']} vs Epoch {best_epoch+1 if best_epoch is not None else 'N/A'}: {best_value}"
				)

		if early_stop_prefix:
			val_loss_status = early_stop_prefix + val_loss_status


		# Fetch VRAM usage (if available)
		peak_vram = "N/A"
		if self.vram_callback:
			peak_vram = self.vram_callback.get_peak_vram_usage() / (1024 ** 2)  # Convert to MB
			peak_vram = f"{peak_vram:.2f} MB"

		# Log standardized epoch information
		print(
			f"Epoch # {epoch+1} | Time: {fmt(epoch_time, '.2f')}s | "
			f"\n\t\t{val_loss_status} | Peak VRAM Used: {peak_vram} | "
			f"\n\t\tLoss: {fmt(metrics['loss'])} | \t MSE: {fmt(metrics['mse'])} | \t MAE: {fmt(metrics['mae'])} | \t RMSE: {fmt(metrics['rmse'])} | \t  NSE: {fmt(metrics['nse'])} | "
			f"\n\t{self.patience_metric}: {fmt(metrics['val_loss'])} | Val MSE: {fmt(metrics['val_mse'])} | Val MAE: {fmt(metrics['val_mae'])} | Val RMSE: {fmt(metrics['val_rmse'])} | Val NSE: {fmt(metrics['val_nse'])}\n----\n"
		)

	def on_train_begin(self, logs=None):
		print("Training started.")

	def on_train_end(self, logs=None):
		print("Training completed.")

class Heartbeat(keras.callbacks.Callback):
	"""Print a progress line every `every` batches so long SLURM jobs show activity."""
	def __init__(self, every=50, num_batches=None):
		super().__init__()
		self.every = every
		self.num_batches = num_batches
		self._epoch = 0
	def on_epoch_begin(self, epoch, logs=None):
		self._epoch = epoch
		self._epoch_start_time = time.time()

	def on_train_batch_end(self, batch, logs=None):
		if batch != 0 and batch % self.every == 0:
			epoch_time = time.time() - self._epoch_start_time
			print(f"\t[HB] epoch={self._epoch+1} batch={batch}/{self.num_batches} loss={logs.get('loss') if logs else 'NA'} | Time: {epoch_time:.2f}s ( {datetime.datetime.fromtimestamp(time.time())} )", flush=True)

def masked(fn, name):
	"""Wrap an element-wise Keras loss so that missing (NaN) labels are skipped.

	Labels are NaN where the observation was missing before imputation. The mean is over the
	finite labels of the batch only (0 if none). `name` is what Keras logs, e.g. 'mse', 'val_mse'.
	"""
	def wrapped(y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.cast(y_pred, tf.float32)
		valid = tf.math.is_finite(y_true)
		zero = tf.zeros_like(y_true)
		values = fn(tf.where(valid, y_true, zero)[..., None], tf.where(valid, y_pred, zero)[..., None])
		values = tf.where(valid, values, zero)
		return tf.math.divide_no_nan(tf.reduce_sum(values), tf.reduce_sum(tf.cast(valid, tf.float32)))
	wrapped.__name__ = name
	return wrapped


# --loss-function names; unknown names fall back to mse. nse and kge are minimised as
# 1 - NSE and 1 - KGE, pooled over the batch, on the scaled targets.
LOSS_MAPPING = {
	"mse": masked(keras.losses.mean_squared_error, "mse"),
	"mae": masked(keras.losses.mean_absolute_error, "mae"),
	"mape": masked(keras.losses.mean_absolute_percentage_error, "mape"),
	"huber": masked(keras.losses.huber, "huber"),
	"kge": kge_loss,
	"nse": nse_loss
}

# Optuna prunes and ranks trials on val_mse, whatever the loss. val_loss is not comparable
# across trials: its scale depends on the loss function and it includes the regularisation penalties.
TUNING_METRIC = "val_mse"

class BestEpochTracker(keras.callbacks.Callback):
	"""Track the best epoch of `monitor` and restore its weights when training ends.

	Also restores after reaching max_epochs or Optuna pruning, so the evaluated model is always
	the one from best_epoch (0-based).
	"""
	def __init__(self, monitor='val_loss', mode='min'):
		super(BestEpochTracker, self).__init__()
		self.monitor = monitor
		self.mode = mode
		self.best_epoch = None
		self.best_value = float('inf') if mode == 'min' else float('-inf')
		self.best_weights = None

	def on_epoch_end(self, epoch, logs=None):
		current_value = (logs or {}).get(self.monitor)
		if current_value is None:
			logging.getLogger().warning(f"Metric '{self.monitor}' not found in logs.")
			return
		improved = current_value < self.best_value if self.mode == 'min' else current_value > self.best_value
		if improved:
			self.best_value = current_value
			self.best_epoch = epoch
			self.best_weights = self.model.get_weights()

	def on_train_end(self, logs=None):
		if self.best_weights is not None:
			self.model.set_weights(self.best_weights)
			print(f"Restored weights of best epoch {self.best_epoch + 1} ({self.monitor} = {self.best_value})")


def create_callbacks(log_dir, min_epochs,max_epochs,learning_rate, patience, patience_metric, grad_clip_norm, window, logger, pruning=None):
	"""Build the training callbacks and the Adam optimizer for compile_and_fit.

	learning_rate: None -> Adam default (1e-3), -1 -> triangular cyclical schedule peaking at
	0.005, otherwise a constant rate. patience -1 disables early stopping; log_dir None disables
	the CSV log and checkpoint. pruning is a pruning.RepeatPruning of an Optuna trial, or None.

	Returns:
		(callbacks, vram_callback, best_epoch_tracker, early_stopping, base_opt)
	"""
	# Initialize the callbacks list
	callbacks = []
	vram_callback = None
	best_epoch_tracker = None
	best_epoch = 0
	lr_max = 0.005
	early_stopping = None

	# Add a csv logging callback to the list
	if(log_dir is not None):
		csv_logger = keras.callbacks.CSVLogger(log_dir+"_training.log")
		callbacks.append(csv_logger)

	# Add a VRAM usage callback if a GPU is available
	if tf.config.list_physical_devices('GPU'):
		vram_callback = VRAMUsageCallback()
		callbacks.append(vram_callback)


	# Initialize the base optimizer with or without gradient clipping
	steps_per_epoch = int(np.ceil(window.num_train_samples / window.batch_size))
	print(f"Determined steps_per_epoch from window.num_train_samples / window.batch_size: {steps_per_epoch}")
	
	# schedule half-cycle: the earliest epoch early stopping can end training
	if(patience > -1):
		onecycle_epochs = min_epochs + patience
	else:
		onecycle_epochs = max_epochs

	opt_kwargs = {}
	if grad_clip_norm is not None:
		opt_kwargs["clipnorm"] = grad_clip_norm
	if learning_rate is not None and learning_rate != -1:
		opt_kwargs["learning_rate"] = learning_rate
	base_opt = keras.optimizers.Adam(**opt_kwargs)
	if learning_rate == -1:
		callbacks.append(OneCycleScheduler(lr_max, onecycle_epochs, steps_per_epoch))

	# Optuna pruning on TUNING_METRIC, independent of the loss and patience_metric
	if(pruning is not None):
		class OptunaCallback(keras.callbacks.Callback):
			def on_epoch_end(self, epoch, logs=None):
				if pruning.epoch_end(epoch, (logs or {}).get(TUNING_METRIC)):
					self.model.stop_training = True

		callbacks.append(OptunaCallback())

	# Lower is better for losses/errors, higher for skill scores (e.g. val_nse)
	if(patience_metric == "val_loss" or patience_metric == "val_mse" or patience_metric == "val_mae" or patience_metric == "val_mape" or patience_metric == "val_rmse"):
		mode = "min"
	else:
		mode = "max"

	# Add an early stopping callback if patience is set
	if(patience > -1):
		early_stopping = keras.callbacks.EarlyStopping(
			monitor=patience_metric, 
			patience=patience, 
			start_from_epoch=min_epochs, 
			min_delta=0.0,
			mode=mode,
			verbose=1,
			restore_best_weights=False)  # BestEpochTracker always restores the best weights
		callbacks.append(early_stopping)


	# Add a ModelCheckpoint callback to save the best weights
	if log_dir is not None:
		checkpoint_filename = log_dir + "_best.weights.h5"

		#select monitor mode based on patience_metric (val_loss, val_mse, val_mae, val_mape, val_rmse, val_nse)
		cp_callback = keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filename,
			save_weights_only=True,
			save_best_only=True, 
			monitor=patience_metric, 
			mode=mode,
			verbose=1
		)
		callbacks.append(cp_callback)

	best_epoch_tracker = BestEpochTracker(monitor=patience_metric, mode=mode)
	callbacks.append(best_epoch_tracker)


	# Standardized logging callback
	logging_callback = LoggingCallback(logger, vram_callback, best_epoch_tracker, patience, patience_metric, early_stopping)
	callbacks.append(logging_callback)

	if (heartbeat_flag):
		heartbeat = Heartbeat(every=heartbeat_interval, num_batches=steps_per_epoch)
		callbacks.append(heartbeat)

	return callbacks, vram_callback, best_epoch_tracker, early_stopping, base_opt

def compile_and_fit(model, window, loss="mse", patience=2, patience_metric="val_loss", learning_rate=None, log_dir=None, max_epochs=30, min_epochs=10, grad_clip_norm=1.0, pruning=None,logger=None):
	"""Compile the model and train it on window.train, validating on window.val.

	Returns:
		(history, best_epoch, last_epoch, peak_vram_usage): epochs 0-based, peak VRAM in bytes
		(-1 without a GPU).
	"""
	# the logged nse/rmse are Keras means of per-batch values on the scaled targets, not
	# whole-set scores in original units
	if logger is None:
		logger = logging.getLogger()

	logger.info(f"compile_and_fit()\n\tlog_dir: {log_dir}\n\tmax_epochs: {max_epochs}\n\tmin_epochs: {min_epochs}\n\tpatience: {patience}\n\tpatience_metric: {patience_metric}\n\tloss: {loss}\n\tgrad_clip_norm: {grad_clip_norm}\n\tpruning: {pruning is not None}")
	
	callbacks, vram_callback, best_epoch_tracker, early_stopping, base_opt = create_callbacks(log_dir, min_epochs,max_epochs,learning_rate, patience, patience_metric, grad_clip_norm, window, logger, pruning)
	for callback in callbacks:
		logger.info(callback)

	logger.info(f"max_epochs: {max_epochs}")
	
	loss = LOSS_MAPPING.get(loss, LOSS_MAPPING["mse"])

	logger.info(f"loss: {loss}")

	opt = base_opt


	# Initialize training and validation data
	train = window.train
	val = window.val

	# missing labels are NaN; every metric skips them
	metrics=[LOSS_MAPPING["mse"], LOSS_MAPPING["mae"], LOSS_MAPPING["mape"], rmse, nse]

	model.compile(
		loss=loss,
		optimizer=opt,
		metrics=metrics
		)


	history = model.fit(
		train, 
		epochs=max_epochs,  
		shuffle=True, 
		validation_data=val, 
		verbose=0, 
		callbacks=callbacks,
	)

	# BestEpochTracker restored the best epoch, also without early stopping
	best_epoch = best_epoch_tracker.best_epoch if best_epoch_tracker.best_epoch is not None else len(history.history["loss"]) - 1
	last_epoch = len(history.history["loss"]) - 1
	logger.info(f"Best Epoch: {best_epoch + 1}  Last Epoch: {last_epoch + 1}  (1-based)")

	peak_vram_usage = vram_callback.get_peak_vram_usage() if tf.config.list_physical_devices('GPU') else -1
	logger.info(f"Peak VRAM Usage: {peak_vram_usage / (1024**2):.2f} MB")


	return history, best_epoch, last_epoch, peak_vram_usage

def plot_train_history(history, best_epoch, title, filename):
	"""Plot training/validation MAE and MSE per epoch (log y-axis, 1e-5 to 0.05) and mark the best epoch."""
	best_epoch += 1  # Convert to 1-based index for plotting
	logging.info(f"plotting training history {title} {filename} {best_epoch}")
	train_mae = history.history['mae']
	val_mae = history.history['val_mae']
	train_mse = history.history['mse']
	val_mse = history.history['val_mse']
	train_nse = history.history['nse']
	val_nse = history.history['val_nse']
	train_nse = [nse if nse >= 0 else None for nse in train_nse]
	val_nse = [nse if nse >= 0 else None for nse in val_nse]


	val_rmse = history.history['val_rmse']
	train_rmse = history.history['rmse']

	epochs = range(1, len(val_mae) + 1)

	# plot both training and validation values in one plot. Use x line markers for training and o line markers for validation
	fig, ax1 = plt.subplots(figsize=(10, 6))
	y_1_indices = [0.05, 0.01, 0.001, 0.0001, 0.00001]
	ax1.set_ylim(0.00001, 0.05)  # Set the desired y-axis range
	ax1.set_yscale('log')

	# y_1_indices are being cutoff, like 0.000 and 0.000 instead of 0.0001 and 0.00001
	# This is likely due to the way the ticks are being set. We can adjust the ticks to ensure they are displayed correctly.
	ax1.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.05])
	ax1.set_yticklabels([f'{x:.6f}' for x in ax1.get_yticks()])

	ax1.set_yticks(y_1_indices)
	ax1.set_yticklabels([f'{x:.3f}' for x in y_1_indices])

	# Plotting mae and RMSE on the left y-axis
	ax1.plot(epochs, train_mae, 'bx--', label='Training MAE')
	ax1.plot(epochs, val_mae, 'bo--', label='Validation MAE')
	ax1.plot(epochs, train_mse, 'rx-.', label='Training MSE')
	ax1.plot(epochs, val_mse, 'ro-.', label='Validation MSE')
	ax1.set_ylabel('Error')
	ax1.tick_params(axis='y', labelcolor='k')

	# Setting x-axis label and title
	ax1.set_xlabel('Epochs')
	ax1.set_title(title)

	lines, labels = ax1.get_legend_handles_labels()


	# Adding a vertical line at the "best" epoch
	ax1.axvline(x=best_epoch, color='k', linestyle='-')
	lines.append(ax1.axvline(x=best_epoch, color='k', marker='|', linestyle='None'))
	labels.append('Best Epoch')

	ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.06, 1.0))
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


if __name__ == "__main__": 
	logging.error(os.path.basename(__file__) + " has no main.")
