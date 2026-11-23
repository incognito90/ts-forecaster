"""Data loading and preprocessing for the forecasting pipeline.

readData() loads the data file, adds calendar and signed-log change features, splits it chronologically
into train/val/test, imputes NaNs (fit on train) and returns a WindowGenerator,
which scales the columns and builds the tf.data window datasets.
"""

import os
import re
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, MaxAbsScaler
import fastparquet
import copy
from nan_imputer import NaNImputer
from output_paths import append_lines_locked


# --transform_method -> scaler class, one fitted per column on train; "none" skips scaling.
# the NaN-fill methods are in nan_imputer.py
TRANSFORM_MAPPING =	{
	"none": None,
	"standard": StandardScaler,
	"minmax": MinMaxScaler,
	"robust": RobustScaler,
	"quantile": QuantileTransformer,
	"power": PowerTransformer,
	"maxabs": MaxAbsScaler
}

class WindowGenerator():
	"""Sliding windows (input_steps inputs, shift gap, label_steps labels) as tf.data datasets.

	train_ds/val_ds/test_ds yield (inputs [batch, input_steps, num_features],
	labels [batch, label_steps, num_outputs]); only train_ds is shuffled.
	"""
	def __init__(self, testname, input_steps, label_steps, shift,
			   train_df, val_df, test_df,
			   batch_size=32,
			   label_columns=None, 
			   oversample_indices=None, train_nan_indices=None, val_nan_indices=None, test_nan_indices=None, transform_method = "minmax",
			   raw_test_labels=None, train_label_means=None, train_label_naive_mae=None,
			   predict="level"):
		"""Scale the data and build the train/val/test window datasets.

		train/val/test_nan_indices: dict column -> rows that were NaN before imputation.
		raw_test_labels: [test rows, num_labels] test targets in original units, NaN where missing.
		train_label_means, train_label_naive_mae: [num_labels], original units (climatology, MASE scale).
		predict: what the model outputs per target:
			"level": the scaled target values
			"change": (y[t+k] - last observed y) / change_scale
			"both": level / level_scale followed by change (2 * num_labels outputs)
		"""
		self.testname=testname
		self.batch_size=batch_size

		self.input_steps = input_steps 				#number of time steps to look back
		self.label_steps = label_steps 				#number of time steps to predict
		self.shift = shift							#number of time steps to shift the window

		# work out the row numbers
		self.num_train_samples = len(train_df)
		self.num_val_samples = len(val_df)
		self.num_test_samples = len(test_df)
		
		# Work out the label column indices.
		self.label_columns = label_columns 			#list of column names to predict

		if label_columns is not None:
			self.label_columns_indices = {name: i for i, name in
											enumerate(label_columns)}
			
		self.input_columns = train_df.columns 		#list of column names to use as input features	
		self.column_indices = {name: i for i, name in
							enumerate(train_df.columns)}

		self.num_features=len(self.input_columns)	#number of input features
		if not label_columns:
			raise ValueError("label_columns (the target columns) is required")
		self.num_labels=len(self.label_columns)		#number of labels to predict
		if predict not in ("level", "change", "both"):
			raise ValueError(f"predict must be level, change or both, not {predict}")
		self.predict = predict
		# outputs per step, the labels twice with predict == "both"
		self.num_outputs = self.num_labels * (2 if predict == "both" else 1)


		self.raw_test_labels = raw_test_labels
		self.train_label_means = train_label_means
		self.train_label_naive_mae = train_label_naive_mae
		self.test_nan_indices = copy.deepcopy(test_nan_indices)
		self.val_nan_indices = copy.deepcopy(val_nan_indices)
		self.train_nan_indices = copy.deepcopy(train_nan_indices)

		if(transform_method != "none"):
			print(f"Transforming data using {transform_method}")
			with pd.option_context('display.max_columns', None, 'display.width', None):
				print("train_df: \n", train_df.describe().T, "\nval_df:\n ", val_df.describe().T, "\ntest_df:\n ", test_df.describe().T )

			# time features are already in [-1, 1] and never scaled
			# Columns NOT to be scaled
			self.columns_not_to_scale = TIME_FEATURE_COLUMNS

			#print all columns in train_df
			print(f"train_df columns: {train_df.columns}")	
			print(f"columns_not_to_scale: {self.columns_not_to_scale}")
			print(f"columns_to_scale: {train_df.columns.difference(self.columns_not_to_scale)}")

			# Determine columns TO BE scaled (all columns except the ones specified to be excluded)
			self.columns_to_scale = train_df.columns.difference(self.columns_not_to_scale)
			self.indices_to_scale = [self.column_indices[col] for col in self.columns_to_scale]
			print(f"columns_to_scale: {self.columns_to_scale}")

			
			# Initialize a dictionary to hold a scaler for each column
			self.scalers = {}
			self.train_df = train_df
			self.val_df = val_df
			self.test_df = test_df

			# Scale each column individually
			for col in self.columns_to_scale:
				self.scalers[col] = TRANSFORM_MAPPING[transform_method]()
				self.train_df[col] = self.scalers[col].fit_transform(train_df[[col]])

			# Do the same for the validation and test dataframes
			for col in self.columns_to_scale:
				self.val_df[col] = self.scalers[col].transform(val_df[[col]])
				self.test_df[col] = self.scalers[col].transform(test_df[[col]])

			print("transformed data")
			

		else:
			self.train_df = train_df 
			self.val_df = val_df 
			self.test_df = test_df 

		
		# always float32, --half-float only affects the model
		self.train_df = self.train_df.astype(np.float32)
		self.val_df = self.val_df.astype(np.float32)
		self.test_df = self.test_df.astype(np.float32)

		# observed_labels: targets (scaled), NaN where the value was missing, so the loss
		# skips them. label_anchor: last observed value (ffill, past only), the base of
		# the change targets. The inputs keep the imputed values (min_std 0 marker).
		self.observed_labels = {}
		self.label_anchor = {}
		for split, df, nan_idx in (("train", self.train_df, train_nan_indices),
				("val", self.val_df, val_nan_indices), ("test", self.test_df, test_nan_indices)):
			obs = df[self.label_columns].to_numpy(dtype=np.float32, copy=True)
			for k, name in enumerate(self.label_columns):
				rows = np.asarray((nan_idx or {}).get(name, []), dtype=np.int64)
				obs[rows, k] = np.nan
			self.observed_labels[split] = obs
			self.label_anchor[split] = pd.DataFrame(obs).ffill().to_numpy(dtype=np.float32)

		# std of (y[t+k] - anchor[t]) on train, scales the change labels
		self.label_input_indices = [self.column_indices[name] for name in self.label_columns]
		self.change_scale = np.ones(self.num_labels, dtype=np.float32)
		# with predict == both the level labels are also scaled to unit variance,
		# so both heads weigh about the same in the loss
		self.level_scale = np.ones(self.num_labels, dtype=np.float32)
		if self.predict == "both":
			self.level_scale = np.array([max(float(np.nanstd(self.observed_labels["train"][:, k])), 1e-12) for k in range(self.num_labels)], dtype=np.float32)
		if self.predict in ("change", "both"):
			obs = self.observed_labels["train"].astype(np.float64)
			anchor = self.label_anchor["train"].astype(np.float64)
			for k in range(self.num_labels):
				# rows ahead of the last input row: shift + 1 .. shift + label_steps
				diffs = np.concatenate([obs[h:, k] - anchor[:-h, k] for h in range(self.shift + 1, self.shift + 1 + self.label_steps)])
				diffs = diffs[np.isfinite(diffs)]
				self.change_scale[k] = max(float(np.std(diffs)), 1e-12) if len(diffs) else 1.0
			print(f"predict={self.predict}: change scale (scaled units) per target: {dict(zip(self.label_columns, self.change_scale))}")


		self.total_window_size = self.input_steps + self.shift + self.label_steps
		self.input_slice = slice(0, self.input_steps)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]
		self.label_start = self.input_steps + self.shift
		self.labels_slice = slice(self.label_start, self.label_start + self.label_steps)
		self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

		# Generate datasets
		print(f"input_steps: {self.input_steps}, label_steps: {self.label_steps}, shift: {self.shift}, total_window_size: {self.total_window_size}")

		print(f"\nGenerating datasets")
		print(f"\n--------------\nval_df:")
		self.val_ds = self.make_dataset(self.val_df, "val")
		print(f"\n--------------\ntest_df:")
		self.test_ds = self.make_dataset(self.test_df, "test")


		print(f"\n--------------\ntrain_df:")
		self.train_ds = self.make_train_dataset(self.train_df, "train", oversample_indices=oversample_indices)
		for inputs, labels in self.train_ds.take(1):
			print(f"inputs device: {inputs.device}")
			print(f"labels device: {labels.device}")
		# # Validate datasets
		self.validate_datasets(
			self.train_ds, 
			self.val_ds, 
			self.test_ds,
			self.num_train_samples, 
			self.num_val_samples, 
			self.num_test_samples, 
			self.total_window_size,
			self.test_nan_indices,
			self.val_nan_indices,
			self.train_nan_indices
		)
		for inputs, labels in self.train_ds.take(1):
			print(f"inputs device: {inputs.device}")
			print(f"labels device: {labels.device}")


		for inputs, labels in self.train_ds.take(1):
			print(f"inputs device: {inputs.device}")
			print(f"labels device: {labels.device}")

		# batch the train dataset and prefetch
		print(f"Batching with batch_size {self.batch_size} and prefetching datasets")
		self.train_ds = self.train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
		self.val_ds = self.val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
		self.test_ds = self.test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


	def validate_datasets(self, train_ds, val_ds, test_ds, 
						num_train_samples, num_val_samples, num_test_samples, 
						total_window_size,
						test_nan_indices, val_nan_indices, train_nan_indices):
		"""
		Perform all validation checks on train_ds, val_ds, and test_ds.
		Includes dataset cardinality, NaN/Inf checks, and prints relevant shapes and specs.
		"""
		# View and validate the datasets shape, size, and cardinality
		print(f"Train DataFrame shape: {self.train_df.shape}")
		print(f"\ttrain_ds: {self.train_ds}\n\t{self.train_ds.element_spec}")
		assert len(self.train_df) > self.total_window_size, f"Train dataset is too small for the window size ({len(self.train_df)} < {self.total_window_size})"

		print(f"Validation DataFrame shape: {self.val_df.shape}")
		print(f"\tval_ds: {self.val_ds}\n\t{self.val_ds.element_spec}")
		assert len(self.val_df) > self.total_window_size, f"Validation dataset is too small for the window size ({len(self.val_df)} < {self.total_window_size})"
		
		print(f"Test DataFrame shape: {self.test_df.shape}")
		print(f"\ttest_ds: {self.test_ds}\n\t{self.test_ds.element_spec}")
		assert len(self.test_df) > self.total_window_size, f"Test dataset is too small for the window size ({len(self.test_df)} < {self.total_window_size})"
		

		print(f"num_train_samples: {self.num_train_samples}, num_val_samples: {self.num_val_samples}, num_test_samples: {self.num_test_samples}")
		print(f"total_window_size: {self.total_window_size}")
		print(f"input_steps: {self.input_steps}, label_steps: {self.label_steps}, shift: {self.shift}")


		print("Checking NaN indices in datasets")
		#check nan indices are within bounds for each dataset
		def validate_nan_indices(nan_indices, dataset_name, num_samples):
			for column, indices in nan_indices.items():
				# Check if the indices are within bounds
				if len(indices) > 0:
					if np.any(indices < 0) or np.any(indices >= num_samples):
						raise ValueError(f"Invalid {dataset_name} NaN indices for column {column}: {indices}")

		validate_nan_indices(self.train_nan_indices, "train", num_train_samples)
		validate_nan_indices(self.val_nan_indices, "val", num_val_samples)
		validate_nan_indices(self.test_nan_indices, "test", num_test_samples)
					

		# Print and inspect dataset elements
		for ds_name, ds in [("train_ds", self.train_ds), ("val_ds", self.val_ds), ("test_ds", self.test_ds)]:
			print(f"Inspecting {ds_name} samples:")
			for i, (inputs, labels) in enumerate(ds.take(3)):
				print(f"{ds_name} - Sample {i+1}:")
				print(f"\tInputs shape: {inputs.shape}")
				print(f"\tLabels shape: {labels.shape}")

				
				# labels are NaN where missing, so only check them for inf
				tf.debugging.check_numerics(inputs, f"{ds_name} inputs")
				if tf.reduce_any(tf.math.is_inf(labels)):
					raise ValueError(f"Inf detected in {ds_name} at sample {i+1}")

			print(f"{ds_name} inspection complete.")

		print("All dataset validations passed.")


	def split_window(self, features):
		"""Split one window of the _table() array into (inputs, labels).

		inputs: the feature columns (imputed values) of the input rows. labels:
		observed_labels of the label rows (NaN where missing), or for change targets
		(label - anchor) / change_scale with anchor = last observed value at the last input row.
		"""
		F, L = self.num_features, self.num_labels
		inputs = features[self.input_slice, :F]  # [input_steps, num_features]
		labels = features[self.labels_slice, F:F + L]  # [label_steps, num_labels]

		if self.predict in ("change", "both"):
			last = features[self.input_steps - 1, F + L:F + 2 * L]  # [num_labels], last observed value
			change = (labels - last[None, :]) / tf.constant(self.change_scale)
			labels = change if self.predict == "change" else tf.concat([labels / tf.constant(self.level_scale), change], axis=-1)

		return inputs, labels

	def _table(self, data, split):
		"""float32 [rows, num_features + 2 * num_labels]: features, observed_labels, label_anchor."""
		return np.concatenate([data.to_numpy(dtype=np.float32), self.observed_labels[split],
			self.label_anchor[split]], axis=1)

	def make_train_dataset(self, data, split="train", oversample_indices=None):
		"""Unbatched training dataset, fully reshuffled every epoch.

		Windows are sliced from one table tensor by start row, so memory is the size of
		the table. oversample_indices: window start rows to add once more.
		"""
		table = tf.constant(self._table(data, split), dtype=tf.float32)
		num_windows = len(data) - self.total_window_size + 1
		starts = np.arange(num_windows, dtype=np.int64)
		if oversample_indices is not None:
			extra = np.asarray([i for i in oversample_indices if 0 <= i < num_windows], dtype=np.int64)
			print(f"Oversampling: adding {len(extra)} duplicated training windows")
			starts = np.concatenate([starts, extra])
			self.num_train_samples += len(extra)
		print(f"Training windows: {len(starts)} (fully shuffled every epoch)")

		window_size = self.total_window_size
		ds = tf.data.Dataset.from_tensor_slices(starts)
		ds = ds.shuffle(len(starts), reshuffle_each_iteration=True)
		ds = ds.map(lambda i: self.split_window(table[i:i + window_size]), num_parallel_calls=tf.data.AUTOTUNE)
		ds = ds.map(lambda x, y: (tf.ensure_shape(x, (self.input_steps, self.num_features)), tf.ensure_shape(y, (self.label_steps, self.num_outputs))))
		return ds

	def make_dataset(self, data, split):
		"""Unbatched val/test dataset in order, window i starts at row i
		(observed_test_windows and outputs_to_level rely on this order)."""

		print(f"Making dataset from data of shape {data.shape}")
			
		print(f"Label columns: {self.label_columns}")

		print(f"Data columns: {data.columns}")
		print(f"Data shape: {data.shape}")
		print(f"sequence_length: {self.total_window_size}")
		print(f"Batch size: {self.batch_size}")


		ds = tf.keras.utils.timeseries_dataset_from_array(
			data=self._table(data, split),
			targets=None,
			sequence_length=self.total_window_size,
			sequence_stride=1,
			batch_size=None,
			shuffle=False)

		print(f"Dataset cardinality immediately after creation: {ds.cardinality().numpy()}")
		
		ds = ds.map(lambda *args: self.split_window(args[0]))
	
		# Log cardinality immediately
		print(f"Dataset cardinality after split features/labels: {ds.cardinality().numpy()}")


		print("Inspecting 3 dataset elements:")
		for element in ds.take(3):
			if isinstance(element, tuple) and len(element) == 2:
				features, labels = element
				print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
			else:
				print(f"Unexpected element len {len(element)} element structure: {element.shape}")

		ds = ds.map(lambda x, y: (tf.ensure_shape(x, (self.input_steps, self.num_features)), tf.ensure_shape(y, (self.label_steps, self.num_outputs))))


		return ds

	@property
	def train(self):
		return self.train_ds

	@property
	def val(self):
		return self.val_ds

	@property
	def test(self):
		return self.test_ds

	@property
	def example(self):
		"""Get and cache an example batch of `inputs, labels` for plotting."""
		result = getattr(self, '_example', None)
		if result is None:
			# No example batch was found, so get one from the `.train` dataset
			result = next(iter(self.train))
			# And cache it for next time
			self._example = result
		return result

	def seasonal_naive_windows(self, period):
		"""Seasonal-naive forecast for every test window: the value whole seasons
		(`period` steps) before the target that was known at the last input row.

		Returns float64 [n_windows, label_steps, num_labels], NaN where not observed.
		"""
		raw = self.raw_test_labels
		n_windows = raw.shape[0] - self.total_window_size + 1
		starts = np.arange(n_windows)[:, None]
		target = starts + self.label_start + np.arange(self.label_steps)[None, :]
		last_input = starts + self.input_steps - 1
		reference = target - np.ceil((target - last_input) / period).astype(int) * period
		out = np.full((n_windows, self.label_steps, raw.shape[1]), np.nan)
		ok = reference >= 0
		out[ok] = raw[reference[ok]]
		return out

	def outputs_to_level(self, predictions, head=None, starts=None, split="test"):
		"""Turn model outputs (or labels) into scaled target levels.

		predictions: [n, label_steps, num_outputs], e.g. model.predict(window.test).
		head: "level" or "change", which half to use when predict == "both".
		starts: start rows of the n windows in `split` (default 0 .. n-1).
		Returns [n, label_steps, num_labels] in scaled units.
		"""
		p = np.asarray(predictions, dtype=np.float64)
		head = self.predict if self.predict != "both" else (head or "level")
		if self.predict == "both":
			p = p[..., :self.num_labels] * self.level_scale.astype(np.float64) if head == "level" else p[..., self.num_labels:]
		if head == "level":
			return p
		# anchor = last observed value at the last input row, as in the training labels
		starts = np.arange(p.shape[0]) if starts is None else np.asarray(starts, dtype=np.int64)
		last = self.label_anchor[split].astype(np.float64)[starts + self.input_steps - 1]
		return last[:, None, :] + p * self.change_scale[None, None, :].astype(np.float64)

	def observed_test_windows(self):
		"""Observed targets of every test window (original units, NaN where missing).

		Returns observed [n_windows, label_steps, num_labels] and last_observed
		[n_windows, num_labels], the last observed value at or before the last input
		row (persistence forecast, and the anchor of change forecasts).
		"""
		raw = self.raw_test_labels
		starts = np.arange(raw.shape[0] - self.total_window_size + 1)
		label_rows = starts[:, None] + self.label_start + np.arange(self.label_steps)[None, :]
		last = pd.DataFrame(raw).ffill().to_numpy(dtype=np.float64)
		return raw[label_rows], last[starts + self.input_steps - 1]

	def complete_test_window_mask(self):
		"""True for test windows with observed targets at every forecast step (SHAP samples)."""
		observed, _ = self.observed_test_windows()
		return np.isfinite(observed).all(axis=(1, 2))

	def revert_label_predictions(self, predictions):
		"""Inverse-scale predictions [n, label_steps, num_labels] to original units (float64)."""
		preds = np.array(predictions, dtype=np.float64)
		scalers = getattr(self, "scalers", {})
		n, steps = preds.shape[0], preds.shape[1]
		for label, k in self.label_columns_indices.items():
			if label in scalers:
				preds[:, :, k] = scalers[label].inverse_transform(preds[:, :, k].reshape(-1, 1)).reshape(n, steps)
		return preds

	def plot(self, plot_col, model=None, max_subplots=3, filename="slices.png"):
		"""Save a plot of up to max_subplots windows of the example batch to `filename`."""
		inputs, labels = self.example
		plt.figure(figsize=(12, 8))
		plot_col_index = self.column_indices[plot_col]
		max_n = min(max_subplots, len(inputs))
		for n in range(max_n):
			plt.subplot(max_n, 1, n+1)
			plt.ylabel(f'{plot_col} [transformed]')
			plt.plot(self.input_indices, inputs[n, :, plot_col_index],
					label='Inputs', marker='.', zorder=-10)

			if self.label_columns:
				label_col_index = self.label_columns_indices.get(plot_col, None)
			else:
				label_col_index = plot_col_index

			if label_col_index is None:
				continue

			plt.scatter(self.label_indices, labels[n, :, label_col_index],
						edgecolors='k', label='Labels', c='#2ca02c', s=64)
			if model is not None:
				predictions = model(inputs)
				plt.scatter(self.label_indices, predictions[n, :, label_col_index],
						marker='X', edgecolors='k', label='Predictions',
						c='#ff7f0e', s=64)

			if n == 0:
				plt.legend()

		plt.title(plot_col)
		plt.xlabel('Time [h]')
		plt.savefig(filename)
		plt.close()


	def __repr__(self):
		return '\n'.join([
			f'\nWindowGenerator',
			f'\tTotal window size: {self.total_window_size}',
			f'\tInput column name(s): {self.input_columns}',
			f'\tLabel column name(s): {self.label_columns}',

			f'\tinput_steps: {self.input_steps}',
			f'\tnum_features: {self.num_features}',

			f'\tlabel_steps: {self.label_steps}',
			f'\tnum_labels: {self.num_labels}',

			f'\tnum_train_samples: {self.num_train_samples}',
			f'\ttrain_ds: {self.train_ds}',
			f'\t\t{self.train_ds.element_spec}',

			f'\tnum_val_samples: {self.num_val_samples}',
			f'\t\tval_ds: {self.val_ds}',
			f'\t\t{self.val_ds.element_spec}',

			f'\tnum_test_samples: {self.num_test_samples}',
			f'\t\ttest_ds: {self.test_ds}',
			f'\t\t{self.test_ds.element_spec}',

			])


def detect_peaks_in_raw(df, labels, percentile_threshold=0.75):
		"""Train row indices where any label is above its percentile_threshold quantile."""

		print(f"\n----\nDetecting peaks using {percentile_threshold} percentile threshold. ")
		highs = []

		for label in labels:

			# Determine the threshold values from the training dataset
			high_threshold = df[label].quantile(percentile_threshold)

			# Detect high events (peaks)
			high_events = df[df[label] > high_threshold]

			high_indices = high_events.index.values.tolist()

			print(f'{label} high_threshold: {high_threshold:.2f} high_events: {len(high_events)}')

			highs.append(high_indices)
		
		#remove duplicates
		highs = list(set([item for sublist in highs for item in sublist]))

		print(f'{len(highs)} high events.')
		print(f'first 10 high indices: {highs[:10]}')
		print(f'first 10 high values: {df.loc[highs[:10]]}')


		return highs

def detect_valleys_in_raw(df, labels, percentile_threshold=0.25):
		"""Train row indices where any label is below its percentile_threshold quantile."""

		print(f"\n----\nDetecting valleys using {percentile_threshold} percentile threshold. ")
		lows = []

		for label in labels:

			# Determine the threshold values from the training dataset
			low_threshold = df[label].quantile((percentile_threshold))

			# Detect low events (valleys)
			low_events = df[df[label] < low_threshold]
			low_indices = low_events.index.values.tolist()

			print(f'{label} low_threshold: {low_threshold:.2f} low_events: {len(low_events)}')

			lows.append(low_indices)
		
		#remove duplicates
		lows = list(set([item for sublist in lows for item in sublist]))

		print(f'{len(lows)} low events.')
		print(f'first 10 low indices: {lows[:10]}')
		print(f'first 10 low values: {df.loc[lows[:10]]}')


		return lows


#train_df test_df and val_df are dataframes used temporarily before creating a Tensorflow dataset in init
def readData(testname, filename, label_tag="_Label", input_seq_length = 24, output_seq_length = 2, output_seq_offset = 0, train_ratio=0.70, val_ratio=0.15,test_ratio=0.15, scaleBuffer=1.0, batch_size=32, oversample_high_thresholds = None , oversample_low_thresholds = None, nan_imputation_method = "ffill",transform_method="minmax", log_dir=None, trend_period_years=0, date_column="Date", time_features="auto", predict="level", change_lags=(1, 3, 6), change_columns=None):
	"""Load a data file and return a WindowGenerator (used by predictor_main.py and shap_explain.py).

	label_tag: regex, every matching column is a target.
	change_lags, change_columns: signed-log change input features (add_change_features).
	train_ratio, val_ratio: chronological split fractions; test gets the remaining rows.
	oversample_high/low_thresholds: quantiles; train rows beyond them get their window duplicated.
	nan_imputation_method: NaNImputer method, or "none".
	"""
	print(f"Reading data from {filename} with label_tag: {label_tag}, input_seq_length: {input_seq_length}, output_seq_length: {output_seq_length}, output_seq_offset: {output_seq_offset}, train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {test_ratio}, scaleBuffer: {scaleBuffer}, batch_size: {batch_size}, oversample_high_thresholds: {oversample_high_thresholds}, oversample_low_thresholds: {oversample_low_thresholds}, nan_imputation_method: {nan_imputation_method}, transform_method: {transform_method}")

	# ------- Read Data ------- 
	df = load_table(filename, date_column=date_column)
	print("df\n",df.head(100))

	labels_names = list(df.filter(regex = label_tag))
	df = drop_non_numeric_columns(df, labels_names)

	if(labels_names == []):
		sys.exit(f"Error: no target column found. Target columns are the columns whose name contains "
			f"--label-tag '{label_tag}'. Columns in the file: {list(df.columns)}. Rename your target "
			f"column(s) to include '{label_tag}' or pass a different --label-tag (e.g. --label-tag _Flow for the GRWS data).")

	#  ------- Format Time ------- 
	# Format Date-Time into 0-1 values 
	# Each day/year starts at Cos = 1 and Sin = 0
	# Convert 'Date' column to datetime

	seconds_in_day = 24 * 3600

	# parse and sort the dates and infer the time step; windows assume evenly spaced rows
	try:
		df['Date'] = pd.to_datetime(df['Date'])
	except (ValueError, TypeError) as e:
		sys.exit(f"Error: could not read the Date column ({e}). Use ISO dates, e.g. "
			f"'2010-01-31' (daily) or '2010-01-31 13:00:00' (sub-daily).")
	if df['Date'].isna().any():
		sys.exit(f"Error: {int(df['Date'].isna().sum())} rows have an empty or unreadable Date.")
	if not df['Date'].is_monotonic_increasing:
		print("Warning: rows were not in time order; sorting them by Date.")
		df = df.sort_values('Date', kind='stable').reset_index(drop=True)
	if df['Date'].duplicated().any():
		sys.exit(f"Error: the Date column has {int(df['Date'].duplicated().sum())} duplicate time stamps.")
	steps = df['Date'].diff().dropna()
	step = steps.median()
	irregular = float((steps != step).mean())
	if irregular > 0:
		print(f"Warning: {irregular:.1%} of the time steps differ from the typical step ({step}); "
			f"windows are cut by row, so they will not all span the same length of time.")
	timestep_seconds = int(step.total_seconds())
	hourly = step < pd.Timedelta(days=1)   # sub-daily data also get time-of-day features
	print(f"Time step: {step} ({'sub-daily' if hourly else 'daily or longer'})")
	date_time = df['Date']
	features = resolve_time_features(time_features, hourly, trend_period_years)
	print(f"Calendar features: {', '.join(features) if features else 'none'}")
	time_columns = []

	# all sin/cos pairs in [-1, 1], never scaled
	if "time_of_day" in features:
		timestamp_s = date_time.map(pd.Timestamp.timestamp)
		day_time_seconds = timestamp_s % seconds_in_day
		df['Time sin'] = np.sin(day_time_seconds * (2 * np.pi / seconds_in_day))
		df['Time cos'] = np.cos(day_time_seconds * (2 * np.pi / seconds_in_day))
		time_columns += ['Time sin', 'Time cos']

	# 'Day sin/cos': day of year (annual cycle)
	if "day_of_year" in features:
		day_of_year = date_time.dt.dayofyear
		df['Day sin'] = np.sin(2 * np.pi * day_of_year / 365.2425)
		df['Day cos'] = np.cos(2 * np.pi * day_of_year / 365.2425)
		time_columns += ['Day sin', 'Day cos']

	# 'Week sin/cos': position in the week, Monday 00:00 = 0 (with the hour for sub-daily data)
	if "day_of_week" in features:
		week_position = (date_time.dt.dayofweek + (date_time.dt.hour + date_time.dt.minute / 60) / 24) / 7
		df['Week sin'] = np.sin(2 * np.pi * week_position)
		df['Week cos'] = np.cos(2 * np.pi * week_position)
		time_columns += ['Week sin', 'Week cos']

	# 'Year sin/cos': calendar year in a cycle of trend_period_years, from the first year
	if "trend" in features:
		first_year = date_time.dt.year.min()
		df['Year sin'] = np.sin(2 * np.pi * (date_time.dt.year - first_year) / trend_period_years)
		df['Year cos'] = np.cos(2 * np.pi * (date_time.dt.year - first_year) / trend_period_years)
		time_columns += ['Year sin', 'Year cos']

	print("Current DataFrame columns:", df.columns)

	#print first 50 rows of time columns
	print("Time Columns\n",df[time_columns].head(5))

	n = len(df)
	num_train = int(n*train_ratio)
	num_val = int(n*val_ratio)
	num_test = int(n*test_ratio)

	#print all df columns
	print("All DataFrame columns:", df.columns)

	# call the function to remove empty columns
	df = remove_empty_columns(df, time_columns, labels_names, num_train, num_val, num_test)

	#print all df columns
	print("All DataFrame columns:", df.columns)


	#print description of all labels in df
	max_label_length = max(len(label) for label in labels_names)
	format_string = f'{{:<{max_label_length}}}'
	for label in labels_names:
		formatted_label = format_string.format(label)
		stats = df[label].describe().T.to_string()
		stats = ' '.join(stats.split('\n')[1:])  # Remove the first line and join the rest
		print(f'{formatted_label}: {stats}')


	# signed-log change features, from the raw values (before gap filling and scaling)
	df, change_names = add_change_features(df, labels_names, label_tag, parse_change_lags(change_lags),
		parse_change_columns(change_columns), num_train)
	if change_names:
		print(f"Change features: {', '.join(change_names)}")

	#  ------- Split data ------- 
	# Split data into train/test/val
	column_indices = {name: i for i, name in enumerate(df.columns)}
	n = len(df)
	num_train = int(n*train_ratio)
	num_val = int(n*val_ratio)
	num_test = int(n*test_ratio)

	# chronological split, no shuffling; test gets the remaining rows
	train_df = df[0:num_train].reset_index(drop=True)
	val_df = df[num_train:num_train+num_val].reset_index(drop=True)
	test_df = df[num_train+num_val:].reset_index(drop=True)

	print("sizes: ", n, " \ntrain " , train_df.shape, "\n val ", val_df.shape, "\ntest ", test_df.shape)
	print("train_df\n",train_df.head(8))
	print("val_df\n",val_df.head(8))
	print("test_df\n",test_df.head(8))

	print(f'num_train: {num_train} num_val: {num_val} num_test: {num_test}')

	print(f"Oversample thresholds: {oversample_high_thresholds} {oversample_low_thresholds}")

	oversample_indices = None if oversample_high_thresholds is None and oversample_low_thresholds is None else []
	if(oversample_high_thresholds is not None):
		for threshold in oversample_high_thresholds:
			highs = detect_peaks_in_raw(train_df.loc[:, train_df.columns != 'Date'], labels_names, threshold)
			print(f'number of peaks above {threshold} percentile: {len(highs)}')
			oversample_indices.extend(highs)

	if(oversample_low_thresholds is not None):
		for threshold in oversample_low_thresholds:
			lows = detect_valleys_in_raw(train_df.loc[:, train_df.columns != 'Date'], labels_names, threshold)
			print(f'number of valleys below {threshold} percentile: {len(lows)}')
			oversample_indices.extend(lows)

	total_window_size = input_seq_length + output_seq_length + output_seq_offset
	
	# map raw rows to window starts (first label at row r); windows that do not fit
	# in train are dropped and each window is added once
	start_offset = input_seq_length + output_seq_offset
	if oversample_indices is not None:
		mapped = []
		max_start = len(train_df) - total_window_size
		for r in oversample_indices:
			s = r - start_offset						# makes r == first label timestep
			if 0 <= s <= max_start:
				mapped.append(s)
		oversample_indices = sorted(set(mapped))

	print("train_df\n",train_df.head(50))

	# rows that were NaN before imputation; WindowGenerator sets those labels to NaN
	# so the loss skips them
	train_nan_indices = {}
	val_nan_indices = {}
	test_nan_indices = {}

	#print train_nan_indices for each column where  index is less than 100
	for column in df.columns:
		train_nan_indices[column] = train_df[column].index[train_df[column].apply(np.isnan)]
		val_nan_indices[column] = val_df[column].index[val_df[column].apply(np.isnan)]
		test_nan_indices[column] = test_df[column].index[test_df[column].apply(np.isnan)]

		#print the values stored in train_nan_indices that are below 100. Print a max of 10 values.
		if(len(train_nan_indices[column]) > 0):
			print(f'train_nan_indices {column} {train_nan_indices[column][:10]}')
		if(len(val_nan_indices[column]) > 0):
			print(f'val_nan_indices {column} {val_nan_indices[column][:10]}')
		if(len(test_nan_indices[column]) > 0):
			print(f'test_nan_indices {column} {test_nan_indices[column][:10]}')

	# raw targets (not imputed or scaled) for evaluation, and the train mean for climatology
	raw_test_labels = test_df[labels_names].to_numpy(dtype=np.float64, copy=True)
	train_label_means = train_df[labels_names].mean(skipna=True).to_numpy(dtype=np.float64)
	# one-step naive MAE on train (MASE scale)
	train_label_naive_mae = np.nanmean(np.abs(np.diff(train_df[labels_names].to_numpy(dtype=np.float64), axis=0)), axis=0)

	#print the number of nans in each column
	print("Number of NaNs in train_df:")
	for column in train_df.columns:
		num_nans = train_df[column].isna().sum()
		if num_nans > 0:
			print(f"{column}: {num_nans} NaNs")
	print("Number of NaNs in val_df:")	
	for column in val_df.columns:
		num_nans = val_df[column].isna().sum()
		if num_nans > 0:
			print(f"{column}: {num_nans} NaNs")
	print("Number of NaNs in test_df:")
	for column in test_df.columns:
		num_nans = test_df[column].isna().sum()
		if num_nans > 0:
			print(f"{column}: {num_nans} NaNs")

	if nan_imputation_method != "none" and nan_imputation_method != None:
		print(f"Imputing NaNs using method: {nan_imputation_method}")
		import time
		imputer_start_time = time.time()
		imputer = NaNImputer(method=nan_imputation_method,  time_index_column="Date")
		print("Fitting imputer on train_df")
		train_df = imputer.fit_transform(train_df)
		print("Imputing val_df and test_df")
		val_df = imputer.transform(val_df)
		test_df = imputer.transform(test_df)
		imputer_end_time = time.time()
		print(f"Imputer time: {imputer_end_time - imputer_start_time:.2f} seconds")
		#print impute time to csv file, if the file is already being written to, wait until it is free

		imputer_log = os.path.join(log_dir, "imputer_time.csv") if log_dir else "imputer_time.csv"
		append_lines_locked(imputer_log,
			[f"{testname},{nan_imputation_method},{imputer_end_time - imputer_start_time:.2f}\n"],
			header="testname,nan_imputation_method,imputer_time_seconds\n")

		#print the number of nans in each column
	print("Number of NaNs in train_df:")
	for column in train_df.columns:
		num_nans = train_df[column].isna().sum()
		if num_nans > 0:
			print(f"{column}: {num_nans} NaNs")
	print("Number of NaNs in val_df:")	
	for column in val_df.columns:
		num_nans = val_df[column].isna().sum()
		if num_nans > 0:
			print(f"{column}: {num_nans} NaNs")
	print("Number of NaNs in test_df:")
	for column in test_df.columns:
		num_nans = test_df[column].isna().sum()
		if num_nans > 0:
			print(f"{column}: {num_nans} NaNs")	

	#remove 'Date' column from train_df, val_df and test_df
	train_df.drop(columns=['Date'], inplace=True)
	val_df.drop(columns=['Date'], inplace=True)
	test_df.drop(columns=['Date'], inplace=True)

	window = WindowGenerator(testname,input_steps=input_seq_length, label_steps=output_seq_length, shift=output_seq_offset,
	 				train_df=train_df, val_df=val_df, test_df=test_df,
	 				batch_size=batch_size,label_columns=labels_names,
					oversample_indices=oversample_indices, test_nan_indices=test_nan_indices, train_nan_indices=train_nan_indices, val_nan_indices=val_nan_indices,
					transform_method=transform_method,
					raw_test_labels=raw_test_labels, train_label_means=train_label_means,
					train_label_naive_mae=train_label_naive_mae, predict=predict)
	# used for the time unit of the SHAP outputs
	window.timestep_seconds = timestep_seconds
	print("WINDOW\n",window)

	return window

def parse_change_lags(spec):
	"""--change-lags: '1,3,6', [1, 3, 6], 'none', '' or None -> sorted list of distinct positive ints."""
	if spec is None:
		return []
	items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
	items = [str(i).strip() for i in items if str(i).strip() and str(i).strip().lower() != "none"]
	try:
		lags = sorted({int(i) for i in items})
	except ValueError:
		raise ValueError(f"change lags must be positive integers, e.g. 1,3,6 (or none), not {spec!r}")
	if any(k < 1 for k in lags):
		raise ValueError(f"change lags must be positive integers, not {spec!r}")
	return lags


def parse_change_columns(spec):
	"""--change-columns: comma-separated regular expressions, a list, 'none', '' or None -> list."""
	if spec is None:
		return []
	items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
	patterns = [str(i).strip() for i in items if str(i).strip() and str(i).strip().lower() != "none"]
	for p in patterns:
		re.compile(p)
	return patterns


def signed_log_change(series, lag, scale):
	"""sign(d) * log1p(|d| / scale), d = x_t - x_(t-lag); NaN where either value is missing."""
	d = series - series.shift(lag)
	return np.sign(d) * np.log1p(d.abs() / scale)


def change_scale(series, num_train):
	"""Median |1-step change| over the training rows (mean if that is 0, else 1)."""
	d1 = (series - series.shift(1)).iloc[:num_train].abs()
	for s in (d1.median(), d1.mean()):
		if np.isfinite(s) and s > 0:
			return float(s)
	return 1.0


def add_change_features(df, labels_names, label_tag, lags, extra_patterns, num_train):
	"""Add signed-log change features as input columns; returns (df, new column names).

	Every target gets one column per lag k: sign(d) * log1p(|d| / s) with d = x_t - x_(t-k),
	and every other column whose name matches one of extra_patterns (regular expressions)
	gets the 1-step change. s is the median |1-step change| of that column over the
	training rows. Raw minmax-scaled values resolve typical hourly changes poorly
	(e.g. 1e-4 of the range), so the model cannot use the recent trend; this keeps small
	changes resolved and compresses large ones. Changes use only past values, and a
	missing value gives a missing change, filled later like any other input.
	Columns are named '<column without the label tag>_d<k>_change'. No lags = no change
	features at all (extra_patterns are ignored too).
	"""
	if not lags:
		return df, []
	specs = [(c, k) for c in labels_names for k in lags]
	if extra_patterns:
		for c in df.columns:
			if c in labels_names or c == "Date" or c in TIME_FEATURE_COLUMNS:
				continue
			if any(re.search(p, c) for p in extra_patterns):
				specs.append((c, 1))
	added, scales = [], {}
	for c, k in specs:
		if c not in scales:
			scales[c] = change_scale(df[c], num_train)
		base = re.sub(label_tag, "", c) if c in labels_names else c
		name = f"{base}_d{k}_change"
		if re.search(label_tag, name) or name in df.columns:
			raise ValueError(f"change feature name {name!r} matches the label tag or an existing column")
		df[name] = signed_log_change(df[c], k, scales[c])
		added.append(name)
	return df, added


# calendar features (--time-features) and the columns they create
TIME_FEATURES = {
	"time_of_day": ['Time sin', 'Time cos'],
	"day_of_year": ['Day sin', 'Day cos'],
	"day_of_week": ['Week sin', 'Week cos'],
	"trend": ['Year sin', 'Year cos'],
}
TIME_FEATURE_COLUMNS = [c for cols in TIME_FEATURES.values() for c in cols]


def resolve_time_features(spec, sub_daily, trend_period_years):
	"""Turn --time-features into a list of feature names.

	"auto": day_of_year, plus time_of_day for sub-daily data and trend if
	trend_period_years > 0. "none": no features. Otherwise a comma-separated list of
	TIME_FEATURES names; time_of_day is skipped for daily data.
	"""
	if spec is None or (isinstance(spec, str) and spec.strip().lower() == "auto"):
		names = ["day_of_year"] + (["time_of_day"] if sub_daily else []) + (["trend"] if trend_period_years and trend_period_years > 0 else [])
	elif isinstance(spec, str) and spec.strip().lower() == "none":
		names = []
	else:
		names = [n.strip() for n in (spec.split(",") if isinstance(spec, str) else spec) if str(n).strip()]
		unknown = [n for n in names if n not in TIME_FEATURES]
		if unknown:
			raise ValueError(f"unknown calendar feature(s) {unknown}; choose from {', '.join(TIME_FEATURES)}, auto or none")
		if "time_of_day" in names and not sub_daily:
			print("Note: time_of_day skipped (the data are daily or coarser).")
			names.remove("time_of_day")
		if "trend" in names and not (trend_period_years and trend_period_years > 0):
			raise ValueError("the trend feature needs --trend-period-years > 0")
	return [n for n in TIME_FEATURES if n in names]


def load_table(filename, date_column="Date"):
	"""Read a parquet, csv or tsv file; the time column `date_column` is renamed to 'Date'."""
	ext = os.path.splitext(str(filename))[1].lower()
	if ext in (".parquet", ".pq"):
		df = fastparquet.ParquetFile(filename).to_pandas()
	elif ext == ".csv":
		df = pd.read_csv(filename)
	elif ext in (".tsv", ".tab", ".txt"):
		df = pd.read_csv(filename, sep="\t")
	else:
		sys.exit(f"Error: unsupported data file type '{ext}' ({filename}). Use .parquet, .csv or .tsv/.tab/.txt.")
	if date_column not in df.columns:
		sys.exit(f"Error: no time column '{date_column}' in {filename}. Columns: {list(df.columns)}. "
			f"Set --date-column (data.date_column in a config) to the name of the time column.")
	if date_column != "Date":
		if "Date" in df.columns:
			sys.exit(f"Error: {filename} has both '{date_column}' and 'Date' columns; rename one of them.")
		df = df.rename(columns={date_column: "Date"})
	return df


def drop_non_numeric_columns(df, labels_names):
	"""Drop non-numeric feature columns (e.g. station names) with a warning; stop if a target is non-numeric."""
	non_numeric = [c for c in df.columns if c != "Date" and not pd.api.types.is_numeric_dtype(df[c])]
	bad_targets = [c for c in non_numeric if c in labels_names]
	if bad_targets:
		sys.exit(f"Error: target column(s) {bad_targets} are not numeric. Convert them to numbers first.")
	if non_numeric:
		print(f"Warning: dropping non-numeric column(s) {non_numeric}; convert them to numbers to use them as inputs.")
		df = df.drop(columns=non_numeric)
	return df


def remove_empty_columns(df, time_columns, labels_names, num_train, num_val, num_test):
	"""Drop zero-std, constant and all-NaN feature columns, over all rows and within each split.

	Time columns are kept; raises ValueError if a label column would be dropped.
	"""
	#remove columns with zero standard deviation

	#get std in all columns except time_columns. So do not include time_columns in std calculation
	std = df.drop(time_columns, axis=1).std()
	#drop 'Date' column from std
	std = std.drop('Date', axis=0, errors='ignore')

	
	train_std = df.drop(time_columns, axis=1).iloc[:num_train].std()
	train_std = train_std.drop('Date', axis=0, errors='ignore')
	val_std = df.drop(time_columns, axis=1).iloc[num_train:num_train+num_val].std()
	val_std = val_std.drop('Date', axis=0, errors='ignore')
	test_std = df.drop(time_columns, axis=1).iloc[num_train+num_val:].std()
	test_std = test_std.drop('Date', axis=0, errors='ignore')

	# Find columns with zero standard deviation
	zero_std_columns = std.index[abs(std) < 1e-6]
	zero_std_columns = zero_std_columns.append(train_std.index[abs(train_std) < 1e-6])
	zero_std_columns = zero_std_columns.append(val_std.index[abs(val_std) < 1e-6])
	zero_std_columns = zero_std_columns.append(test_std.index[abs(test_std) < 1e-6])
	zero_std_columns = zero_std_columns.unique()

	#remove time_columns from zero_std_columns
	zero_std_columns = [x for x in zero_std_columns if x not in time_columns]

	# check if labels_names are in zero_std_columns and throw an error if they are
	for label in labels_names:
		if label in zero_std_columns:
			raise ValueError(f"Label column '{label}' has zero standard deviation. This is likely an error in the data processing pipeline. Please check the data for issues.")

	if(len(zero_std_columns) > 0):
		print(f'Dropping columns with zero standard deviation: {zero_std_columns}')
		df.drop(zero_std_columns, axis=1, inplace=True)
		print(f'Updated DataFrame shape: {df.shape}')

	print("Current DataFrame columns:", df.columns)

	constant_columns = df.columns[df.nunique() <= 1]
	#remove time_columns from constant_columns
	constant_columns = [x for x in constant_columns if x not in time_columns]
	print(f'df shape: {df.shape} constant_columns: {constant_columns}')


	# check if labels_names are in constant_columns and throw an error if they are
	for label in labels_names:
		if label in constant_columns:
			raise ValueError(f"Label column '{label}' has constant values. This is likely an error in the data processing pipeline. Please check the data for issues.")

	if(len(constant_columns) > 0):
		print(f'dropping constant columns: {constant_columns}')
		df.drop(constant_columns, axis=1, inplace=True)
		print(f'df shape: {df.shape}')

	#check the above conditions again, but for the train, val and test subsets of df
	train_constant_columns = df.iloc[:num_train].columns[df.iloc[:num_train].nunique() <= 1]
	train_constant_columns = [x for x in train_constant_columns if x not in time_columns]
	print(f'train_constant_columns: {train_constant_columns}')

	# check if labels_names are in train_constant_columns and throw an error if they are
	for label in labels_names:
		if label in train_constant_columns:
			raise ValueError(f"Label column '{label}' has constant values in the training set. This is likely an error in the data processing pipeline. Please check the data for issues.")

	if(len(train_constant_columns) > 0):
		print(f'dropping train_constant_columns: {train_constant_columns}')
		df.drop(train_constant_columns, axis=1, inplace=True)
		print(f'df shape: {df.shape}')

	val_constant_columns = df.iloc[num_train:num_train+num_val].columns[df.iloc[num_train:num_train+num_val].nunique() <= 1]
	val_constant_columns = [x for x in val_constant_columns if x not in time_columns]
	print(f'val_constant_columns: {val_constant_columns}')
	for label in labels_names:
		if label in val_constant_columns:
			raise ValueError(f"Label column '{label}' has constant values in the validation set. This is likely an error in the data processing pipeline. Please check the data for issues.")
	if(len(val_constant_columns) > 0):
		print(f'dropping val_constant_columns: {val_constant_columns}')
		df.drop(val_constant_columns, axis=1, inplace=True)
		print(f'df shape: {df.shape}')

	test_constant_columns = df.iloc[num_train+num_val:].columns[df.iloc[num_train+num_val:].nunique() <= 1]
	test_constant_columns = [x for x in test_constant_columns if x not in time_columns]

	# check if labels_names are in test_constant_columns and throw an error if they are
	for label in labels_names:
		if label in test_constant_columns:
			raise ValueError(f"Label column '{label}' has constant values in the test set. This is likely an error in the data processing pipeline. Please check the data for issues.")

	print(f'test_constant_columns: {test_constant_columns}')
	if(len(test_constant_columns) > 0):
		print(f'dropping test_constant_columns: {test_constant_columns}')
		df.drop(test_constant_columns, axis=1, inplace=True)
		print(f'df shape: {df.shape}')

	#check for all nan columns in df, train, val and test
	nan_columns = df.columns[df.isna().all()]
	#add to nan_columns if the train, val or test subsets have all nan columns
	train_nan_columns = df.iloc[:num_train].columns[df.iloc[:num_train].isna().all()]
	train_nan_columns = [x for x in train_nan_columns if x not in time_columns]
	nan_columns = nan_columns.append(train_nan_columns)

	val_nan_columns = df.iloc[num_train:num_train+num_val].columns[df.iloc[num_train:num_train+num_val].isna().all()]
	val_nan_columns = [x for x in val_nan_columns if x not in time_columns]
	nan_columns = nan_columns.append(val_nan_columns)

	test_nan_columns = df.iloc[num_train+num_val:].columns[df.iloc[num_train+num_val:].isna().all()]
	test_nan_columns = [x for x in test_nan_columns if x not in time_columns]
	nan_columns = nan_columns.append(test_nan_columns)

	# check if labels_names are in nan_columns and throw an error if they are
	for label in labels_names:
		if label in nan_columns:
			raise ValueError(f"Label column '{label}' has all NaN values. This is likely an error in the data processing pipeline. Please check the data for issues.")

	if(len(nan_columns) > 0):
		print(f'dropping nan columns: {nan_columns}')
		df.drop(nan_columns, axis=1, inplace=True)
		print(f'df shape: {df.shape}')
	
	print("Current DataFrame columns:", df.columns)
	return df


	

if __name__ == "__main__": 
	logging.error(os.path.basename(__file__) + " has no main.")
