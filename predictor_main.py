"""Train, tune (Optuna) and evaluate an LSTM forecaster on one data file.

Usually run through run.py (`python run.py --config my.yaml`), or directly:

    python predictor_main.py --config configs/template.yaml [--option value ...]
"""

#https://www.tensorflow.org/tutorials/structured_data/time_series
#time how long it takes to load os package then print the time


import warnings
import logging
import time
start = time.time()
import os
import uuid
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs
import sys
import json
import re
import math
import argparse
import psutil # for determining ram usage
import datetime

from pathlib import Path #extract filename from path
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib as mpl

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=sys.maxsize)
import seaborn as sns
import tensorflow as tf
import optuna
import search_space
from optuna.pruners import MedianPruner
from pruning import RepeatPruning

# optuna pruning (median rule), warm-up is --min-epochs in every repeat
PRUNER_STARTUP_TRIALS = 5
PRUNER_INTERVAL_STEPS = 5
from output_paths import OutputPaths, append_lines_locked, check_csv_header
import config as run_config
import evaluation
import csv
import io

sns.set_theme()

import data_handling as dh
import keras
from model import (
	MODEL_BUILDERS,
	TUNING_METRIC,
	compile_and_fit,
	load_Model,
	plot_train_history,
)
from metrics import to_scalar


end = time.time()
print("import time: ", end - start)

mpl.rcParams['figure.figsize'] = (8, 6)

# if True, test() also saves the predictions and observations to <output-dir>/predictions/
save_obs_vs_pred = False


def revertPredictions(window, predictions):
	"""Inverse-scale predictions (n_samples, label_steps, num_labels) back to label units."""
	preds = np.array(predictions, dtype=np.float64)
	scalers = getattr(window, "scalers", {})
	n_samples, n_steps = preds.shape[0], preds.shape[1]
	for label, k in window.label_columns_indices.items():
		if label in scalers:
			preds[:, :, k] = scalers[label].inverse_transform(preds[:, :, k].reshape(-1, 1)).reshape(n_samples, n_steps)
	return preds


def level_predictions(window, predictions):
	"""Model outputs -> list of (name suffix, predictions in label units).

	One entry for --predict level/change, two for both ("_levelhead", "_changehead").
	"""
	predict = getattr(window, "predict", "level")
	if predict == "level":
		return [("", revertPredictions(window, predictions))]
	if predict == "change":
		return [("", revertPredictions(window, window.outputs_to_level(predictions)))]
	return [(f"_{head}head", revertPredictions(window, window.outputs_to_level(predictions, head)))
		for head in ("level", "change")]


# --nan-fill options
NAN_FILL_METHODS = ['ffill', 'zero', 'mean', 'median', 'median_mask', 'min_std', 'min_std_2', 'min_std_4', 'interpolate', 'interpolate_time', 'interpolate_polynomial', 'interpolate_spline', 'mean_previous_next', 'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24', 'knn', 'mice', 'kalman', 'missforest', 'missforest_singleton_minstd', 'kalman_singleton_minstd', 'none']

def csv_line(fields):
	"""One quoted CSV line, so lists stay in one column."""
	buf = io.StringIO()
	csv.writer(buf, lineterminator="\n").writerow(["" if f is None else f for f in fields])
	return buf.getvalue()


def evaluated_model_settings(args, model_dir):
	"""Settings for the results.csv rows of a saved model in evaluate mode.

	The model and training settings come from the model's run_config.yaml (blank if missing).
	"""
	record = argparse.Namespace(**vars(args))
	keys = run_config.SECTIONS["model"] + run_config.SECTIONS["training"]
	model_dir = os.fspath(model_dir)
	if model_dir.endswith(".keras"):
		model_dir = os.path.dirname(model_dir)
	saved = os.path.join(model_dir, "run_config.yaml")
	if not os.path.exists(saved):
		logging.warning(f"{saved} not found: the model and training columns of these results.csv rows are left blank.")
		for k in keys:
			setattr(record, k, None)
		return record
	trained = run_config.read_config(saved)
	for k in keys:
		setattr(record, k, trained.get(k))
	data_keys = ["filename", "label_tag", "input_length", "output_length", "output_offset", "time_features",
		"trend_period_years", "change_lags", "change_columns", "predict", "nan_fill", "transform_method", "train_ratio", "val_ratio"]
	differ = [k for k in data_keys if k in trained and str(trained[k]) != str(getattr(args, k, None))]
	if differ:
		logging.warning(f"The evaluate run's data settings differ from the training run ({saved}): "
			+ ", ".join(f"{k}={getattr(args, k, None)} (trained with {trained[k]})" for k in differ))
	return record


def results_header(args):
	"""(metric names, reference names, header line) of results.csv for these settings."""
	metric_names = evaluation.resolve_metric_groups(args.metrics)
	references = ["Persistence", "Climatology"] + (["Seasonal naive"] if args.seasonal_period > 0 else [])
	header_fields = (["Test Name", "Station", "Timestep", "Trial Number", "Trial Repeat", "Trial Seed"] + list(vars(args).keys())
		+ ["Num Features", "Num Labels", "data_variance", "obs_std"] + metric_names
		+ ["Train Time (s)", "RAM (GB)", "VRAM (GB)", "Patience", "Epochs Used", "Best Epoch", "Peak VRAM (bytes)"]
		+ [f"{ref} {m}" for ref in references for m in metric_names])
	return metric_names, references, csv_line(header_fields)


def test(window, y_test, pred, paths, test_name,nan_fill, transform_method,trainTime,peak_vram,patience,epochs_used, best_epoch, args, trial=None,logger=None,trial_repeat=None,trial_seed=None):
	"""Score the test forecasts and append one row per label and lead time to results.csv.

	Args:
		y_test, pred: arrays (n_samples, label_steps, num_labels) in label units.
		best_epoch: 0-based; written 1-based, blank if there was no training.
		trial_repeat: 1-based repeat number (tuning only).
	"""


	logger = logger or logging.getLogger()
	best_epoch_1based = best_epoch + 1 if epochs_used and best_epoch is not None else None
	
	assert y_test.shape[0] == pred.shape[0], "y_test and pred must have the same number of rows"
	assert y_test.shape[1] == pred.shape[1], "y_test and pred must have the same number of time steps"
	assert y_test.shape[2] == pred.shape[2], "y_test and pred must have the same number of columns"

	assert len(window.label_columns) == y_test.shape[2], "y_test must have the same number of columns as the window label columns"
	assert len(window.label_columns) == pred.shape[2], "pred must have the same number of columns as the window label columns"

	
	logger.info(f'\n---------\nTest: {test_name}')

	logger.info("First 5 predictions: %s", pred[:5])
	logger.info("First 5 y_test: %s", y_test[:5])

	if(trial is not None):
		trial_text = f"_Trial-{trial.number}"
		if trial_repeat is not None:
			trial_text += f"_Repeat-{trial_repeat}"
	else:
		trial_text = ""

	metric_names, references, rowHeader = results_header(args)
	trial_fields = [trial.number if trial is not None else None, trial_repeat, trial_seed]
	arg_values = [value for value in vars(args).values()]


	pid = os.getpid()
	python_process = psutil.Process(pid)
	memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
	logger.info("RAM during test (GB): %s", memoryUse)


	if tf.config.list_physical_devices('GPU'):
		vram_peak_gb = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1e9
	else:
		vram_peak_gb = float("nan")
	logger.info(f"Peak VRAM (GB): {vram_peak_gb}")

	
	num_features = window.num_features
	num_labels = window.num_labels

	lines_to_write = []


	print(f"Saving test results to: {paths.results_csv}")
	# persistence = last observed value at or before the last input row,
	# climatology = training mean of the target
	_, last_observed = window.observed_test_windows()
	seasonal = window.seasonal_naive_windows(args.seasonal_period) if args.seasonal_period > 0 else None
	if(np.isnan(pred).all()):
		logger.info(test_name+ trial_text + "," + "NaN")
		lines_to_write.append(test_name + trial_text + "," + "NaN\n")
	else:
		naive_mae = getattr(window, "train_label_naive_mae", None)
		for index, label in enumerate(window.label_columns):
			logger.info(label)
			scale = naive_mae[index] if naive_mae is not None else np.nan
			for j in range(1, y_test.shape[1] + 1):
				# model and references are scored on the same windows (observed target,
				# model forecast and persistence forecast all present)
				model_forecast = pred[:, j-1, index]
				common = np.isfinite(y_test[:, j-1, index]) & np.isfinite(model_forecast) & np.isfinite(last_observed[:, index])
				observed = np.where(common, y_test[:, j-1, index], np.nan)
				scores, data_variance, obs_std = evaluation.compute_metrics(observed, model_forecast, metric_names, scale)
				if j == 1 and "KGE" in scores and (np.isnan(scores["KGE"]) or scores["KGE"] < 0):
					logger.warning(f"KGE for {label} at lead 1 is {scores['KGE']}. This may indicate an issue with the model or data.")
				reference_forecasts = {
					"Persistence": last_observed[:, index],
					"Climatology": np.full_like(observed, window.train_label_means[index]),
				}
				if seasonal is not None:
					reference_forecasts["Seasonal naive"] = seasonal[:, j-1, index]
				reference_scores = []
				for ref in references:
					ref_scores, _, _ = evaluation.compute_metrics(observed, reference_forecasts[ref], metric_names, scale)
					reference_scores += [ref_scores[m] for m in metric_names]
				values = ([test_name + trial_text, label, str(j)] + trial_fields + arg_values
					+ [num_features, num_labels, data_variance, obs_std]
					+ [scores[m] for m in metric_names]
					+ [trainTime, memoryUse, vram_peak_gb, patience, epochs_used, best_epoch_1based, peak_vram]
					+ reference_scores)
				line = csv_line(to_scalar(v) for v in values)
				print(line)
				lines_to_write.append(line)
	# Write all lines to the results file at once
	logger.info("Saving test results to: " + paths.results_csv)
	append_lines_locked(paths.results_csv, lines_to_write, header=rowHeader)

	
	if save_obs_vs_pred:
		#make a single df with both the predictions and the observed values for each label add _pred_timestep# or _obs_timestep# to the column name ex flow_pred_1, flow_obs_1, flow_pred_2, flow_obs_2
		pred_and_obs = pd.DataFrame()
		for index,i in enumerate(window.label_columns):
			for j in range(1,y_test.shape[1]+1):
				pred_and_obs[i+"_pred_"+str(j)] = pred[:,j-1,index]
				pred_and_obs[i+"_obs_"+str(j)] = y_test[:,j-1,index]

		#save the pred_and_obs df to a csv file
		logger.info(f"Saving observed vs predicted values to {paths.predictions}")
		pred_and_obs.to_csv(paths.predictions+"/"+test_name+"_"+nan_fill+"_"+transform_method+trial_text+".csv")
	
		#save pred to <output-dir>/predictions/<test_name>_pred.parquet; this is a 3d array with shape (num_test, label_steps, num_labels)
		pred_df = pd.DataFrame(pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2]))
		logger.info("pred_df\n%s", pred_df.head(8))
		# columns are label1_t1, label2_t1, ..., label1_t2, ...
		step_label_names = [f"{label}_t{s + 1}" for s in range(window.label_steps) for label in window.label_columns]
		pred_df.columns = step_label_names

		pred_df.to_parquet(paths.predictions+"/"+test_name+"_pred.parquet")

		#save y_test to <output-dir>/predictions/<test_name>_y_test.parquet; this is a 3d array with shape (num_test, label_steps, num_labels)
		y_test_df = pd.DataFrame(y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
		logger.info("y_test_df\n%s", y_test_df.head(8))
		y_test_df.columns = step_label_names
		y_test_df.to_parquet(paths.predictions+"/"+test_name+"_y_test.parquet")
	
		#logger.info pred_df and y_test_df to csv files
		pred_df.to_csv(paths.predictions+"/"+test_name+"_pred.csv")
		y_test_df.to_csv(paths.predictions+"/"+test_name+"_y_test.csv")

def run_seed(test_num):
	"""Random seed of a run: 42 + test_num, or 42 when no --test-num is given."""
	return 42 + int(test_num) if test_num is not None else 42


def loaded_cuda_libraries():
	"""Paths of the CUDA/cuDNN libraries loaded in this process (Linux only)."""
	names = ("libcudart.so", "libcublas.so", "libcudnn.so", "libcuda.so")
	try:
		with open("/proc/self/maps") as maps:
			paths = {line.split()[-1] for line in maps if "/" in line}
	except OSError:
		return []
	return sorted(p for p in paths if os.path.basename(p).startswith(names))


def log_gpu_setup():
	"""Print the CUDA/cuDNN build and loaded versions and the GPUs found.

	TF's own messages are hidden (TF_CPP_MIN_LOG_LEVEL=2), so a broken CUDA setup
	would otherwise silently fall back to the CPU.
	"""
	build = tf.sysconfig.get_build_info()
	if not build.get("is_cuda_build"):
		print("TensorFlow build: CPU only")
		return
	print(f"TensorFlow built for CUDA {build.get('cuda_version')} / cuDNN {build.get('cudnn_version')}")
	gpus = tf.config.list_physical_devices('GPU')  # also loads the CUDA libraries
	if not gpus:
		logging.warning("No usable GPU: this GPU build of TensorFlow will run on the CPU. Check the driver and "
			"the CUDA/cuDNN libraries (unset TF_CPP_MIN_LOG_LEVEL to see TensorFlow's messages).")
		return
	for gpu in gpus:
		details = tf.config.experimental.get_device_details(gpu)
		print(f"GPU {gpu.name}: {details.get('device_name')} (compute capability {details.get('compute_capability')})")
	for path in loaded_cuda_libraries():
		version = ""
		if os.path.basename(path).startswith("libcudnn.so"):
			try:
				import ctypes
				v = ctypes.CDLL(path).cudnnGetVersion()  # e.g. 8600 = 8.6.0
				version = f" (cuDNN {v // 1000}.{v % 1000 // 100}.{v % 100})"
			except (OSError, AttributeError):
				pass
		print(f"Loaded: {path}{version}")

def track_vram():
	""" Returns peak VRAM usage in MB """
	if tf.config.list_physical_devices('GPU'):
		vram_info = tf.config.experimental.get_memory_info('GPU:0')
		return vram_info['peak'] / (1024**2)  # Convert bytes to MB
	return -1

def setup_logger(trial_number, log_dir):
	"""Creates a per-trial logger with separate handlers for stdout and stderr."""
	os.makedirs(log_dir, exist_ok=True)
	log_filename = f"{log_dir}/trial_{trial_number}.log"

	# Create a trial-specific logger
	logger = logging.getLogger(f"trial_{trial_number}")
	logger.setLevel(logging.DEBUG)  # Capture all levels of logs

	# Ensure handlers are not duplicated
	if logger.hasHandlers():
		logger.handlers.clear()

	# File handler for logging to a file
	file_handler = logging.FileHandler(log_filename, mode="w")
	file_handler.setLevel(logging.DEBUG)

	# Stream handler for logging INFO and DEBUG to stdout
	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setLevel(logging.DEBUG)
	stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

	# Stream handler for logging WARNING and above to stderr
	stderr_handler = logging.StreamHandler(sys.stderr)
	stderr_handler.setLevel(logging.WARNING)

	# Formatter for consistent log formatting
	formatter = logging.Formatter('%(name)s - %(message)s')
	file_handler.setFormatter(formatter)
	stdout_handler.setFormatter(formatter)
	stderr_handler.setFormatter(formatter)

	# Add handlers to the logger
	logger.addHandler(file_handler)
	logger.addHandler(stdout_handler)
	logger.addHandler(stderr_handler)

	logger.propagate = False  # Ensures logs stay within this logger and do not bubble up

	return logger

MAX_RETRIES = 3  # attempts at building/training/testing a trial before giving up

def tuning_settings(args, search_entries):
	"""Settings stored with an Optuna study; a resumed study must have the same ones."""
	keys = ["filename", "label_tag", "date_column", "input_length", "output_length", "output_offset",
		"train_ratio", "val_ratio", "time_features", "trend_period_years", "change_lags", "change_columns",
		"predict", "nan_fill", "transform_method", "oversample_high_thresholds", "oversample_low_thresholds", "model_type",
		"max_epochs", "min_epochs", "hyperparameter_optimization_repeat", "hyperparameter_optimization_repeat_method"]
	settings = {k: getattr(args, k, None) for k in keys}
	settings["search_space"] = search_entries
	settings["runtime"] = {"python": sys.version.split()[0], "tensorflow": tf.__version__,
		"keras": keras.__version__, "numpy": np.__version__, "pandas": pd.__version__, "optuna": optuna.__version__}
	return json.loads(json.dumps(settings, default=str))


def check_study_settings(study, settings):
	"""Never mix trial scores across settings or runtimes, including unversioned studies."""
	stored = study.user_attrs.get("settings")
	if stored is None and not study.trials:
		study.set_user_attr("settings", settings)
	elif stored != settings:
		differ = sorted(k for k in set(stored or {}) | set(settings) if (stored or {}).get(k) != settings.get(k))
		raise ValueError(f"different or missing settings ({', '.join(differ)}). "
			"Use another run name or output folder for a new search, or the original settings and runtime to resume it.")


#create objective function for hyperparameter optimization with optuna
def objective(strategy, trial, name, nan_fill, transform_method, num_features, input_seq_length, num_labels, output_seq_length, window, max_epochs,min_epochs, num_repeat=1, repeat_method='worst',split=None, paths=None, search_entries=None, args=None):
	"""Train one trial num_repeat times and return its validation MSE (combined with repeat_method).

	Trials are pruned and ranked on validation MSE (model.TUNING_METRIC), whatever
	the loss function. Returns None if a trial fails MAX_RETRIES times.
	"""
	if args is None:
		raise ValueError("objective() needs the parsed settings (args)")
	print(f"Trial {trial.number} {name} {nan_fill} {transform_method}")
	logging.info(f"Trial {trial.number} {name} {nan_fill} {transform_method}")
	
	log_dir = os.path.join(paths.optuna, "trial_logs")
	print(f"Log directory: {log_dir}")
	os.makedirs(log_dir, exist_ok=True)
	trial_logger = setup_logger(trial.number, log_dir)
	trial_logger.info(f"Trial {trial.number} {name}  - Started - {nan_fill} {transform_method}")

	# settings not in the search space keep their command-line value
	sampled = search_space.sample(trial, search_entries)
	# copy of args with the sampled values, mapped the same way as best_config.yaml
	trial_args = argparse.Namespace(**vars(args))
	vars(trial_args).update(search_space.best_settings(sampled, search_entries))
	def pick(key, cli_value):
		return sampled[key] if key in sampled else cli_value
	num_layers = pick('num_layers', args.num_layers)
	if 'n_units_exponent' in sampled:
		n_units = 2 ** sampled['n_units_exponent']
	else:
		n_units = pick('units_per_layer', args.units_per_layer)
	dropout_rate = pick('dropout_rate', args.dropout_rate)
	input_dropout = pick('input_dropout', args.input_dropout) or 0.0  # switched off -> 0
	loss = pick('loss', args.loss_function)
	l1_regularization = pick('l1_regularization', args.l1_regularization)
	l2_regularization = pick('l2_regularization', args.l2_regularization)
	recurrent_l1_regularization = pick('recurrent_l1_regularization', args.l1_recurrent_regularization)
	recurrent_l2_regularization = pick('recurrent_l2_regularization', args.l2_recurrent_regularization)
	dense_l1_regularization = pick('dense_l1_regularization', args.l1_dense_regularization)
	dense_l2_regularization = pick('dense_l2_regularization', args.l2_dense_regularization)
	weight_init = pick('weight_init', args.weight_init)
	batch_norm = pick('batch_norm', args.batch_normalization)
	decoder = pick('decoder', args.decoder)
	noise = pick('noise', args.noise) or 0.0  # switched off -> no noise layer
	patience = pick('patience', args.patience)
	learning_rate = pick('learning_rate', args.learning_rate)  # None = Adam's default rate
	gradient_clip = pick('gradient_clip', args.gradient_clip)

	trial_logger.info(f"num_layers {num_layers=}\nn_units {n_units=}\ndropout_rate {dropout_rate=}\ninput_dropout {input_dropout=}\nl1_regularization {l1_regularization=}\nl2_regularization {l2_regularization=}\nrecurrent_l1_regularization {recurrent_l1_regularization=}\nrecurrent_l2_regularization {recurrent_l2_regularization=}\ndense_l1_regularization {dense_l1_regularization=}\ndense_l2_regularization {dense_l2_regularization=}\nweight_init {weight_init=}\nbatch_norm {batch_norm=}\nnoise {noise=}\npatience {patience=}\nlearning_rate {learning_rate=}\ngradient_clip {gradient_clip=}\nloss {loss=}")
	
	mse_scores = []
	timers = []
	# with 'worst' every repeat can be pruned, otherwise only the first (see pruning.py)
	pruning = RepeatPruning(trial, n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=min_epochs,
		interval_steps=PRUNER_INTERVAL_STEPS, prune_all_repeats=(repeat_method == "worst" or num_repeat == 1))

	for i in range(num_repeat):
		trial_text = f"{trial.number} repeat {i + 1}"

		trial_logger.info(f"Trial {trial_text} - Repeat {i+1}/{num_repeat} - Started")
		pruning.start_repeat()


		keras.backend.clear_session()  # Clear previous models, freeing memory and preventing naming issues
		trainTime = None  # Before training
		start_vram = track_vram()
		start = time.time()
		seed = run_seed(args.test_num) + i
		keras.utils.set_random_seed(seed)

		with strategy.scope():

			# Create the model
			for attempt in range(MAX_RETRIES):
				try:
					model = MODEL_BUILDERS[args.model_type](num_layers, n_units, dropout_rate, num_features, input_seq_length, num_labels, output_seq_length, weight_init=weight_init, batch_normalization=batch_norm, l1_regularization=l1_regularization, l2_regularization=l2_regularization, recurrent_l1_regularization=recurrent_l1_regularization, recurrent_l2_regularization=recurrent_l2_regularization, dense_l1_regularization=dense_l1_regularization, dense_l2_regularization=dense_l2_regularization, noise_std=noise, decoder=decoder, input_dropout=input_dropout, logger=trial_logger)
					trial_logger.info(f"Trial {trial_text} model created")
					break
				except Exception as e:
					trial_logger.warning(f"Trial {trial_text} attempt {attempt+1} to create model failed due to: {e}")
					if attempt == MAX_RETRIES - 1:
						trial_logger.error(f"Trial {trial_text} failed after {MAX_RETRIES} attempts.")
						return None  # Fail gracefully without affecting parameter selection. Or use raise optuna.TrialPruned() to prune the trial


			model.summary(expand_nested=True, print_fn=trial_logger.info)		


			timer_start = time.time()
			# Compile the model using the compile_and_fit function
			for attempt in range(MAX_RETRIES):
				try:
					if attempt:
						pruning.reset_repeat()
					history, best_epoch, last_epoch, peak_vram_usage = compile_and_fit(model, window, patience=patience, patience_metric=args.patience_metric, loss=loss, learning_rate=learning_rate, log_dir=None, max_epochs=max_epochs, min_epochs=min_epochs, grad_clip_norm=gradient_clip, pruning=pruning, logger=trial_logger)
					break
				except Exception as e:
					trial_logger.warning(f"Trial {trial_text} attempt {attempt+1} failed to train due to: {e}")
					if attempt == MAX_RETRIES - 1:
						trial_logger.error(f"Trial {trial_text} failed after {MAX_RETRIES} attempts.")
						return None # Fail gracefully without affecting parameter selection. Or use raise optuna.TrialPruned() to prune the trial

			timers.append(time.time() - timer_start)
		# Calculate the time taken to train the model
		trainTime = time.time() - start
		trial_logger.info(f"Training Time: {trainTime}")
		peak_vram = peak_vram_usage  # bytes


		if pruning.pruned:
			trial_logger.info(f"Trial {trial_text} Pruned")
			pruning.prune()  # raises optuna.TrialPruned
		
		#print all trial keys. This will print all the hyperparameters and metrics of the model
		for key in history.history.keys():
			trial_logger.info(f"{key}: {history.history[key]}")

		
		keys = list(history.history.keys())
		#store the keys in a string with commas between them
		keys = ','.join(keys)

		filename= os.path.join(paths.optuna, "param_search.csv")
		header = f'Trial Number,Trial Repeat,Trial Seed,num_layers,n_units,dropout_rate,input_dropout,loss,l1_regularization,l2_regularization,recurrent_l1_regularization,recurrent_l2_regularization,dense_l1_regularization,dense_l2_regularization,weight_init,batch_norm,noise,patience,learning_rate,gradient_clip,train_time,best_epoch,last_epoch,{keys}\n'
		row = f'{trial.number},{i + 1},{seed},{num_layers},{n_units},{dropout_rate},{input_dropout},{loss},{l1_regularization},{l2_regularization},{recurrent_l1_regularization},{recurrent_l2_regularization},{dense_l1_regularization},{dense_l2_regularization},{weight_init},{batch_norm},{noise},{patience},{learning_rate},{gradient_clip},{trainTime},{best_epoch + 1},{last_epoch + 1}'
		for key in history.history.keys():
			row += f',{history.history[key][best_epoch] if best_epoch is not None and history.history[key][best_epoch] is not None else "None"}'
		append_lines_locked(filename, [row + "\n"], header=header)

		val_loss = history.history['val_loss'][best_epoch]
		val_mse = history.history[TUNING_METRIC][best_epoch]  # same measure the pruning uses

		mse_scores.append(val_mse)
		
		print(f"Trial {trial_text} Training Complete: Best Epoch: {best_epoch + 1}, Last Epoch: {last_epoch + 1}. Validation Loss ({loss}): {val_loss}. Validation MSE: {val_mse}")
		# Return the validation loss of the best epoch

		for attempt in range(MAX_RETRIES):
			try:
				predictions = model.predict(window.test)


				y_test, _ = window.observed_test_windows()

				#get the number of epochs the model was trained for
				epochs_used=len(history.history['loss'])
				trial_logger.info(f"epochs_used {epochs_used}")
				trial_logger.info(f"last_epoch {last_epoch}")
				for suffix, pred in level_predictions(window, predictions):
					test(window, y_test, pred, paths, name + suffix, nan_fill, transform_method,trainTime,peak_vram,patience,epochs_used=epochs_used, best_epoch=best_epoch, args=trial_args, trial=trial, trial_repeat=i + 1, trial_seed=seed)
				trial_logger.info(f"Trial {trial_text} pred/test - Completed")
				
				break
			except Exception as e:
				trial_logger.warning(f"Trial {trial_text} attempt {attempt+1} failed to predict/test due to: {e}")
				if attempt == MAX_RETRIES - 1:
					trial_logger.error(f"Trial {trial_text} failed after {MAX_RETRIES} attempts.")
					return None # Fail gracefully without affecting parameter selection. Or use raise optuna.TrialPruned() to prune the trial

		
		trial_logger.info(f"Trial {trial_text} {name} - Completed")
			# june 20, attempting to reduce memory growth
		del model
		del history
		del predictions
		gc.collect()
		keras.backend.clear_session()

	sorted_mse_scores = sorted(mse_scores)

	average_train_time = np.mean(timers)

	

	if(num_repeat > 1):
		mid_index = len(sorted_mse_scores) // 2
		better_mid_index = (mid_index + 0) // 2  # Best index is always 0 in sorted list
		best_index = 0
		worst_index = len(sorted_mse_scores) - 1
		worst_mid_index = (mid_index + worst_index) // 2
		val_mse_median = np.median(sorted_mse_scores)
		val_mse_mean = np.mean(mse_scores)
		val_mse_worst = np.max(mse_scores)
		val_mse_best = np.min(mse_scores)
		val_mse_better_midrank = sorted_mse_scores[better_mid_index]
		val_mse_worst_midrank = sorted_mse_scores[worst_mid_index]
		val_mse_variance = np.var(sorted_mse_scores)
		val_mse_std = np.std(sorted_mse_scores)

		print(f"Trial {trial_text} - MSE Scores: {sorted_mse_scores} - Mean Train Time {average_train_time}")
		print(f"\tTrial {trial_text} - Median MSE: {val_mse_median}")
		print(f"\tTrial {trial_text} - Mean MSE: {val_mse_mean}")
		print(f"\tTrial {trial_text} - Worst MSE: {val_mse_worst}")
		print(f"\tTrial {trial_text} - Best MSE: {val_mse_best}")
		print(f"\tTrial {trial_text} - Better Midrank Index: {better_mid_index}. MSE: {val_mse_better_midrank}")
		print(f"\tTrial {trial_text} - Worst Midrank Index: {worst_mid_index}. MSE: {val_mse_worst_midrank}")
		print(f"\tTrial {trial_text} - Variance of MSE Scores: {val_mse_variance}")
		print(f"\tTrial {trial_text} - Standard Deviation of MSE Scores: {val_mse_std}")
		print(f"\tTrial {trial_text} - Repeat Method: {repeat_method}")

		filename= os.path.join(paths.optuna, "mse_scores.csv")
		header_mse_labels = ",".join(f"mse_{i+1}" for i in range(num_repeat))
		mse_labels = ",".join(str(v) for v in mse_scores[:num_repeat])
		header = f'Trial Number,num_repeat,repeat_method,{header_mse_labels},num_layers,n_units,dropout_rate,l1_regularization,l2_regularization,recurrent_l1_regularization,recurrent_l2_regularization,dense_l1_regularization,dense_l2_regularization,weight_init,batch_norm,noise,patience,learning_rate,gradient_clip\n'
		row = f'{trial.number},{num_repeat},{repeat_method},{mse_labels},{num_layers},{n_units},{dropout_rate},{l1_regularization},{l2_regularization},{recurrent_l1_regularization},{recurrent_l2_regularization},{dense_l1_regularization},{dense_l2_regularization},{weight_init},{batch_norm},{noise},{patience},{learning_rate},{gradient_clip}\n'
		append_lines_locked(filename, [row], header=header)


		value = {'median': val_mse_median, 'mean': val_mse_mean, 'worst': val_mse_worst, 'best': val_mse_best,
			'better_midrank': val_mse_better_midrank, 'worse_midrank': val_mse_worst_midrank,
			'variance': val_mse_variance, 'std': val_mse_std}.get(repeat_method)
		if value is not None:
			# report the epoch curve of the worst repeat ('worst') or the repeat closest
			# to the score, which later trials are pruned against
			reported = pruning.finish(mse_scores, value, repeat_method)
			trial_logger.info(f"Trial {trial.number}: reported the epoch curve of repeat {reported + 1} for pruning")
			return value


	logging.shutdown() 
	# june 20, attempting to reduce memory growth

	pruning.finish(mse_scores, val_mse, repeat_method)
	return val_mse 

def __main(args):
	"""Run one experiment: load the data, then tune, or train/load a model and test it."""

	pd.set_option("display.precision", 2)
	pd.set_option('display.float_format',  '{:,.2f}'.format)
	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=3)

	
	print(tf.version.VERSION)
	print('Using TensorFlow version: ' + tf.version.VERSION)
	print("Available GPUs:", tf.config.list_physical_devices('GPU'))
	log_gpu_setup()
	print("Thread limits inside container:")
	print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
	print("TF_NUM_INTRAOP_THREADS =", os.environ.get("TF_NUM_INTRAOP_THREADS"))


	filename = args.filename
	print("filename:", args.filename)
	file = Path(filename).stem
	print("file: " + str(file))

	if not os.path.exists(args.filename):
		raise FileNotFoundError(f"Input file not found: {args.filename}")

	label_tag = args.label_tag
	print("label_tag: " + str(label_tag))

	input_seq_length = args.input_length
	print("input_seq_length: " + str(input_seq_length))

	output_seq_length = args.output_length
	print("output_seq_length: " + str(output_seq_length))

	output_seq_offset = args.output_offset
	print("output_seq_offset: " + str(output_seq_offset))

	half_float = args.half_float.lower() == "true"
	print("half_float: " + str(half_float))

	oversample_high_thresholds = args.oversample_high_thresholds
	print("oversample_high_thresholds: " + str(oversample_high_thresholds))

	oversample_low_thresholds = args.oversample_low_thresholds
	print("oversample_low_thresholds: " + str(oversample_low_thresholds))

	nan_fill = args.nan_fill
	print("nan_fill: " + str(nan_fill))

	transform_method = args.transform_method
	print("transform_method: " + str(transform_method))

	model_type = args.model_type
	print("model_type: " + str(model_type))

	num_layers=args.num_layers
	print("num_layers: " + str(num_layers))

	n_units=args.units_per_layer
	print("n_units: " + str(n_units))

	dropout_rate=args.dropout_rate
	print("dropout_rate: " + str(dropout_rate))

	loss_fuction = args.loss_function
	print("loss_fuction: " + str(loss_fuction))

	noise = args.noise
	print("noise: " + str(noise))

	l1_reg=args.l1_regularization
	print("l1_reg: " + str(l1_reg))

	l2_reg=args.l2_regularization
	print("l2_reg: " + str(l2_reg))

	recurrent_l1_reg=args.l1_recurrent_regularization
	print("recurrent_l1_reg: " + str(recurrent_l1_reg))

	recurrent_l2_reg=args.l2_recurrent_regularization
	print("recurrent_l2_reg: " + str(recurrent_l2_reg))

	dense_l1_reg=args.l1_dense_regularization
	print("dense_l1_reg: " + str(dense_l1_reg))

	dense_l2_reg=args.l2_dense_regularization
	print("dense_l2_reg: " + str(dense_l2_reg))

	batch_norm=args.batch_normalization
	print("batch_norm: " + str(batch_norm))


	patience=args.patience
	print("patience: " + str(patience))


	patience_metric=args.patience_metric
	print("patience_metric: " + str(patience_metric))

	max_epochs = args.max_epochs
	print("max_epochs: " + str(max_epochs))

	min_epochs = args.min_epochs
	print("min_epochs: " + str(min_epochs))

	learning_rate=args.learning_rate
	print("learning_rate: " + str(learning_rate))

	weight_init = args.weight_init
	print("weight_init: " + str(weight_init))

	gradient_clip = args.gradient_clip
	print("gradient_clip: " + str(gradient_clip))

	mirroredStrategy = args.mirrored.lower() == "true"
	print("mirroredStrategy: " + str(mirroredStrategy))

	batch_size = args.batch_size
	print("batch_size: " + str(batch_size))

	testNum = args.test_num
	print("testNum: " + str(testNum))
	# also seeds the optuna sampler
	seed = run_seed(testNum)
	keras.utils.set_random_seed(seed)
	print("Random seed set to: " + str(seed))
	# process-wide, so it covers every optuna trial too
	if args.deterministic.lower() == "true":
		tf.config.experimental.enable_op_determinism()
		print("Deterministic TensorFlow ops: enabled")

	name = args.testname
	print("name: " + str(name))

	trainFlag = args.train
	print("trainFlag: " + str(trainFlag))

	if(not trainFlag):
		trainedModelFile = args.trainedModelFile
		print("trainedModelFile: " + str(trainedModelFile))

	save_model = args.save_model.lower() == "true"
	print("save_model: " + str(save_model))

	hyperparameter_optimization = args.hyperparameter_optimization
	print("hyperparameter_optimization: " + str(hyperparameter_optimization))

	hyperparameter_optimization_repeat = args.hyperparameter_optimization_repeat
	print("hyperparameter_optimization_repeat: " + str(hyperparameter_optimization_repeat))

	hyperparameter_optimization_repeat_method = args.hyperparameter_optimization_repeat_method
	print("hyperparameter_optimization_repeat_method: " + str(hyperparameter_optimization_repeat_method))


	paths = OutputPaths(args.output_dir or os.path.join("outputs", name)).create()
	print("Output folder: " + paths.root)
	# fail now, not after training, if results.csv has other columns
	try:
		check_csv_header(paths.results_csv, results_header(args)[2])
	except ValueError as e:
		sys.exit(f"Error: {e}")

	#set half float precision
	# mixed precision only: don't set floatx to float16, that makes the inputs
	# float16 and rounds small regularization factors to 0
	if(half_float):
		policy = keras.mixed_precision.Policy('mixed_float16')
		keras.mixed_precision.set_global_policy(policy)

		print('Compute dtype: %s' % policy.compute_dtype)
		print('Variable dtype: %s' % policy.variable_dtype)


	testName = name +"_"+ file  + "_" + str(input_seq_length) +"_"+ str(output_seq_length) +"_"+ str(output_seq_offset) + "_" + str(model_type) +"_"+ str(half_float) +"_"+ str(testNum)+"_"+ str(patience) +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	used_config = run_config.write_config(args, os.path.join(paths.root, "configs", testName + ".yaml"),
		header="Settings used for run " + testName + ". Repeat it with: python predictor_main.py --config <this file>")
	print("Settings written to: " + used_config)
	print(testName)

	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
	gpus = tf.config.list_physical_devices('GPU')
	num_gpus = len(gpus)

	if gpus and args.num_gpu_splits!=None and args.num_gpu_splits > 1:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	if mirroredStrategy and num_gpus> 1:
		strategy = tf.distribute.MirroredStrategy()
		print("REPLICAS: ", strategy.num_replicas_in_sync)
		batch_size = batch_size * strategy.num_replicas_in_sync
	else:
		device = "/GPU:0" if num_gpus >= 1 else "/CPU:0"
		strategy = tf.distribute.OneDeviceStrategy(device=device)

	window= dh.readData(testName,filename, log_dir=paths.logs, date_column=args.date_column, time_features=args.time_features, trend_period_years=args.trend_period_years, change_lags=args.change_lags, change_columns=args.change_columns, label_tag=label_tag, input_seq_length=input_seq_length, output_seq_length=output_seq_length, output_seq_offset=output_seq_offset, batch_size=batch_size, train_ratio=args.train_ratio, val_ratio=args.val_ratio, oversample_high_thresholds=oversample_high_thresholds,  oversample_low_thresholds=oversample_low_thresholds, nan_imputation_method=nan_fill, transform_method=transform_method, predict=args.predict)

	val_performance = {}
	performance = {}


	if(hyperparameter_optimization != None):
		print("Hyperparameter Optimization")
		print("Number of trials: ", hyperparameter_optimization)

		n_jobs = args.hyperparameter_optimization_n_jobs # Number of parallel jobs
		if n_jobs > 1:

			# Set GPU memory growth at the start (before Optuna parallelization)
			gpus = tf.config.list_physical_devices('GPU')
			if gpus:
				try:
					# Enable memory growth instead of setting per-trial limits
					for gpu in gpus:
						tf.config.experimental.set_memory_growth(gpu, True)
				except RuntimeError as e:
					print("TensorFlow GPU Initialization Error:", e)


		from sqlalchemy import create_engine
		from optuna.storages._rdb.models import BaseModel
		from optuna.trial import TrialState

		# a shared db dir must exist and gets a <testname>/ subfolder
		if args.hyperparameter_optimization_db_dir is None:
			optuna_db_dir = paths.optuna
		elif os.path.exists(args.hyperparameter_optimization_db_dir):
			optuna_db_dir = os.path.join(args.hyperparameter_optimization_db_dir, name)
		else:
			raise FileNotFoundError(f"Hyperparameter optimization database directory does not exist: {args.hyperparameter_optimization_db_dir}")
		os.makedirs(optuna_db_dir, exist_ok=True)

		# If running in SLURM array, use per-array DB file
		split = args.cur_gpu_split
		slurm_array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
		if slurm_array_id is not None:
			print(f"Running in SLURM array job (task {slurm_array_id})")
			db_suffix = f"{slurm_array_id}" + (f"_split{split}" if split else "")
		else:
			db_suffix = f"{split}"
		optuna_db_path = os.path.join(optuna_db_dir, f"{name}_array{db_suffix}.db")


		filename= os.path.join(paths.optuna, "param_search.csv")
		print("optuna_db_path: ", optuna_db_path)
		print("optuna filename: ", filename)


		import sqlite3

		with sqlite3.connect(optuna_db_path) as con:
			con.execute("PRAGMA journal_mode=WAL;")
			con.execute("PRAGMA synchronous=NORMAL;")  # Optional, safe trade-off

		print("connected to optuna db: ", optuna_db_path)


		storage_name = f"sqlite:///{optuna_db_path}?timeout=30"
		print("Using storage: ", storage_name)

		print("Trial name: ", name)

		engine = create_engine(storage_name)
		BaseModel.metadata.create_all(engine, checkfirst=True)

		search_entries = search_space.load_search_space(args.search_space)
		print(f"Search space: {args.search_space} ({len(search_entries)} entries)")

		# not consulted: pruning is done by RepeatPruning in objective()
		pruner = MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=min_epochs, interval_steps=PRUNER_INTERVAL_STEPS)
		sampler = optuna.samplers.TPESampler(seed=seed)
		study = optuna.create_study(direction="minimize" , pruner=pruner, sampler=sampler, study_name=name, storage=storage_name, load_if_exists=True)
		# Only resume scores from matching data, search settings and runtime.
		study_settings = tuning_settings(args, search_entries)
		try:
			check_study_settings(study, study_settings)
		except ValueError as e:
			sys.exit(f"Error: the Optuna study '{name}' in {optuna_db_path}: {e}")
		
		existing_trials = len([t for t in study.trials
                       if t.state in (TrialState.COMPLETE,
                                      TrialState.PRUNED,
                                      TrialState.RUNNING)])
		remaining_trials = max(0, hyperparameter_optimization - existing_trials)
		print(f"Existing trials: {existing_trials} - remaining: {remaining_trials}")

		if remaining_trials:
			study.optimize(lambda trial: objective(strategy, trial, name,nan_fill, transform_method, window.num_features, input_seq_length, window.num_outputs, output_seq_length, window, max_epochs,min_epochs, num_repeat=hyperparameter_optimization_repeat, repeat_method=hyperparameter_optimization_repeat_method, split=split, paths=paths, search_entries=search_entries, args=args), n_trials=remaining_trials, n_jobs=n_jobs, gc_after_trial=True, timeout=args.hyperparameter_optimization_timeout)
		else:
			print("Target number of trials already reached - skipping optimisation.")	
		print("Number of finished trials: ", len(study.trials))
		if not any(t.state == TrialState.COMPLETE for t in study.trials):
			sys.exit(f"Error: no trial of the Optuna study '{name}' completed (see the trial logs in "
				f"{os.path.join(paths.optuna, 'trial_logs')} for the errors), so there is no best configuration.")
		print("Best trial:")
		trial = study.best_trial
		print("Best trial number: ", trial.number)
		print("Value: ", trial.value)
		print("Params: ")
		for key, value in trial.params.items():
			print(f"    {key}: {value}")

		filename= os.path.join(paths.optuna, "param_search_best.csv")
		# fixed columns: every entry of the search space (blank if the best trial did not sample it)
		param_names = [e["name"] for e in search_entries]
		header = "Trial Number,value," + ",".join(param_names) + "\n"
		row = f"{trial.number},{trial.value}," + ",".join(str(trial.params.get(k, "")) for k in param_names) + "\n"
		append_lines_locked(filename, [row], header=header)

		# config for training the final model with the best trial's values
		best_overrides = search_space.best_settings(trial.params, search_entries)
		best_overrides.update({"mode": "train", "hyperparameter_optimization": None, "output_dir": os.path.join(paths.root, "final"), "save_model": True})
		best_config = run_config.write_config(args, os.path.join(paths.root, "best_config.yaml"), overrides=best_overrides,
			header=(f"Best settings found by the Optuna search '{name}' (trial {trial.number}, score {trial.value}).\n"
				"Train the final model with: python run.py --config <this file>  (run.py --mode tune does this)"))
		print("Best configuration written to: " + best_config)
		exit()

	test_uuid = str(uuid.uuid4())[:8]
	if(not trainFlag): # load model from file
		print("Model Loading from: ", trainedModelFile)
		model = load_Model(trainedModelFile)
		trainTime = 0
		last_epoch = 0
		best_epoch = 0
		peak_vram_usage = 0
		print("Model loaded.")
	else: # create new model
		with strategy.scope():
			build_model = MODEL_BUILDERS[model_type]
			model = build_model(
				num_layers,
				n_units,
				dropout_rate,
				window.num_features, window.input_steps, window.num_outputs, window.label_steps,
				weight_init,
				batch_normalization = batch_norm,
				l1_regularization = l1_reg,
				l2_regularization = l2_reg,
				recurrent_l1_regularization = recurrent_l1_reg,
				recurrent_l2_regularization = recurrent_l2_reg,
				dense_l1_regularization = dense_l1_reg,
				dense_l2_regularization = dense_l2_reg,
				noise_std = noise,
				decoder = args.decoder,
				input_dropout = args.input_dropout,
			)
			model_name = model_type
			print(model_name)
			model.summary(expand_nested=True)

			startTime = time.time()
			history, best_epoch, last_epoch, peak_vram_usage = compile_and_fit(model, window, 
				patience=patience,  
				patience_metric=patience_metric,
				loss=loss_fuction,
				max_epochs=max_epochs, 
				min_epochs=min_epochs, 
				learning_rate=learning_rate, 
				grad_clip_norm=gradient_clip, 
				log_dir=os.path.join(paths.checkpoints, testName+test_uuid)
				)
			trainTime = time.time() - startTime
			print("last line in strategy scope")
		print("first line outside strategy scope")	

		# convert the history.history dict to a pandas DataFrame, and save to csv 
		hist_df = pd.DataFrame(history.history) 
		hist_csv_file = os.path.join(paths.history, testName+test_uuid+"_history.csv")
		print("Saving history to: ", hist_csv_file)
		with open(hist_csv_file, mode='w') as f:
			hist_df.to_csv(f)

		print("Model compiled and fitted. Time taken: ", trainTime)
		val_performance[model_name] = model.evaluate(window.val)
		print("Model validation eval complete.")

		print(model_name, " train performance: ", model.evaluate(window.train))

		if(save_model):
			model_path = os.path.join(paths.models, testName+test_uuid)
			print("Saving model to: ", model_path)
			os.makedirs(model_path, exist_ok=True)
			model.save(os.path.join(model_path, "model.keras"))
			# used by evaluate mode
			run_config.write_config(args, os.path.join(model_path, "run_config.yaml"))

	print(f"\n-----\nPredicting")
	predictions = model.predict(window.test)


	print("raw predictions: ")
	print("predictions shape: ", predictions.shape)

	#print the first 8 lines of the numpy.ndarray predictions
	print(predictions[:8])
	
	pred_sets = level_predictions(window, predictions)
	pred = pred_sets[0][1]

	print("reverted predictions: ")
	print("pred shape: ", pred.shape)
	print(pred[:8])


	y_test, _ = window.observed_test_windows()
	print("reverted y_test: ")
	print("y_test shape: ", y_test.shape)
	print(y_test[:10])

	print(f"Nans in y_test: {np.isnan(y_test).sum()}")


	#get the number of epochs the model was trained for
	if 'history' in locals():
		epochs_used=len(history.history['loss'])
	else:
		epochs_used=0

	print("epochs_used ", epochs_used)
	print("last_epoch ", last_epoch)

	#if pred contains only nan then exit
	if(np.isnan(pred).all()):
		print("All predictions are nan")
		#throw an error
		raise Exception("All predictions are nan")

	record_args = args if trainFlag else evaluated_model_settings(args, trainedModelFile)
	for suffix, pred in pred_sets:
		test(window, y_test, pred, paths, testName + suffix,nan_fill, transform_method,trainTime,peak_vram_usage,patience,epochs_used=epochs_used, best_epoch=best_epoch, args=record_args)

	if 'history' in locals():
		plot_train_history(history, best_epoch,
			"Training Progress. " + file + " ISL " + str(input_seq_length) + " OSL " + str(output_seq_length), 
			os.path.join(paths.images, "training_"+ testName +".png"))

	if trainFlag:
		if save_model:
			print("Saved model to: ", model_path)
		print(f"Saved history to: ", hist_csv_file)

	print("\n END")

def clean_args(parser, args):
	"""Check and convert the parsed arguments in place (parser.error() on bad values)."""
	if args.filename is None:
		parser.error("no input file: give it as the first argument or as data.file in --config")
	if args.testname is None:
		parser.error("no experiment name: give it as the second argument or as run.name in --config")

		# Convert the string 'None' to the actual None object
	if args.l1_regularization == 'None':
		args.l1_regularization = None
	elif args.l1_regularization is not None:
		args.l1_regularization = float(args.l1_regularization)  # Convert to float if it's not None

	if args.l2_regularization == 'None':
		args.l2_regularization = None
	elif args.l2_regularization is not None:
		args.l2_regularization = float(args.l2_regularization)

	if args.l1_recurrent_regularization == 'None':
		args.l1_recurrent_regularization = None
	elif args.l1_recurrent_regularization is not None:
		args.l1_recurrent_regularization = float(args.l1_recurrent_regularization)

	if args.l2_recurrent_regularization == 'None':
		args.l2_recurrent_regularization = None
	elif args.l2_recurrent_regularization is not None:
		args.l2_recurrent_regularization = float(args.l2_recurrent_regularization)

	if args.l1_dense_regularization == 'None':
		args.l1_dense_regularization = None
	elif args.l1_dense_regularization is not None:
		args.l1_dense_regularization = float(args.l1_dense_regularization)

	if args.l2_dense_regularization == 'None':
		args.l2_dense_regularization = None
	elif args.l2_dense_regularization is not None:
		args.l2_dense_regularization = float(args.l2_dense_regularization)

	if args.gradient_clip == 'None':
		args.gradient_clip = None
	elif args.gradient_clip is not None:
		args.gradient_clip = float(args.gradient_clip)

		

	# If output_offset is not specified, set it to 0 by default
	if args.output_offset is None:
		args.output_offset = 0

	if args.mode == "evaluate":
		args.train = "False"
	if args.train.lower() not in ["true", "false"]:
		parser.error("--train must be True or False")
	args.train = args.train.lower() == "true"
	if not args.train and args.trainedModelFile is None:
		parser.error("--trainedModelFile must be specified when --train is False")
	if not args.train:
		# no search when evaluating a saved model, even if the config has tuning settings
		args.hyperparameter_optimization = None


	if args.hyperparameter_optimization is not None and (not isinstance(args.hyperparameter_optimization, int) or args.hyperparameter_optimization <= 0):
		parser.error("--hyperparameter-optimization must be either 'None' or an integer greater than 0 representing the number of trials to run")


	if args.hyperparameter_optimization_repeat < 1:
		parser.error("--hyperparameter-optimization-repeat must be greater than 0")

	if args.hyperparameter_optimization_repeat_method not in ["median", "mean", "worst", "best", "better_midrank", "worse_midrank"]:
		parser.error("--hyperparameter-optimization-repeat-method must be one of the following: median, mean, worst, best, better_midrank, worse_midrank")

	if args.weight_init not in ["zeros", "ones", "random_normal", "random_uniform", "glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]:
		parser.error("--weight-init must be one of the following: zeros, ones, random_normal, random_uniform, glorot_normal, glorot_uniform, he_normal, he_uniform")

	if args.learning_rate is not None and args.learning_rate != -1 and args.learning_rate <= 0:
		parser.error("--learning-rate must be -1 (cyclical schedule) or a positive rate; omit it for Adam's default rate.")

	if args.loss_function not in ["mse", "mae", "mape", "huber", "nse", "kge"]:
		parser.error("--loss-function must be one of the following: mse, mae, mape, huber, nse, kge")

	if args.gradient_clip is not None and args.gradient_clip <= 0:
		parser.error("--gradient-clip must be greater than 0.")

	if args.hyperparameter_optimization is not None:
		try:
			search_space.load_search_space(args.search_space)
		except (OSError, ValueError) as e:
			parser.error(f"--search-space: {e}")

	if args.model_type not in MODEL_BUILDERS:
		parser.error(f"--model-type must be one of: {', '.join(MODEL_BUILDERS)}")

	if args.l1_regularization is not None and args.l1_regularization <= 0:
		parser.error("--l1-regularization must be greater than 0.")

	if args.l2_regularization is not None and args.l2_regularization <= 0:
		parser.error("--l2-regularization must be greater than 0.")

	if args.l1_recurrent_regularization is not None and args.l1_recurrent_regularization <= 0:
		parser.error("--l1-recurrent-regularization must be greater than 0.")
	
	if args.l2_recurrent_regularization is not None and args.l2_recurrent_regularization <= 0:
		parser.error("--l2-recurrent-regularization must be greater than 0.")

	if args.half_float.lower() not in ["auto", "true", "false"]:
		parser.error("--half-float must be auto, True or False")
	if args.half_float.lower() == "auto":
		args.half_float = "True" if tf.config.list_physical_devices("GPU") else "False"
		print(f"--half-float auto -> {args.half_float} (GPU {'found' if args.half_float == 'True' else 'not found'})")

	if not 0 <= args.input_dropout < 1:
		parser.error("--input-dropout must be at least 0 and less than 1")

	if args.suppress_warnings.lower() not in ["true", "false"]:
		parser.error("--suppress-warnings must be True or False")
	if args.deterministic.lower() not in ["true", "false"]:
		parser.error("--deterministic must be True or False")

	if args.decoder.lower() not in ["true", "false"]:
		parser.error("--decoder must be True or False")
	args.decoder = args.decoder.lower() == "true"

	if args.batch_normalization.lower() not in ["true", "false"]:
		parser.error("--batch-normalization must be one of the following: True, False")
	elif args.batch_normalization.lower() == "true":
		args.batch_normalization = True
	else:
		args.batch_normalization = False
	
	if args.patience != -1 and args.patience < 0:
		parser.error("--patience must be >= than 0 OR -1 do disable.")

	if args.patience_metric not in ["val_loss", "val_mse", "val_mae", "val_mape", "val_rmse", "val_nse"]:
		parser.error(f"--patience-metric must be one of the following: val_loss, val_mse, val_mae, val_mape, val_rmse, val_nse\nNot {args.patience_metric}")

	if args.max_epochs <= 0:
		parser.error("--max-epochs must be greater than 0")

	if args.num_layers < 1:
		parser.error("--num-layers must be greater than 0")

	if args.units_per_layer <= 0:
		parser.error("--units-per-layer must be greater than 0")

	if args.dropout_rate < 0 or args.dropout_rate > 1:	
		parser.error("--dropout-rate must be between 0 and 1")

	if args.noise < 0:
		parser.error("--noise must be greater than 0")

	#clean up the oversample-thresholds argument
	if args.oversample_high_thresholds is not None:
		#convert the string to a list of floats
		args.oversample_high_thresholds = list(map(float, args.oversample_high_thresholds))
		#check if the values are between 0 and 1
		for value in args.oversample_high_thresholds:
			if value <= 0.5 or value >= 1:
				parser.error("--oversample-high-thresholds must be a list of floats between 0.5 and 1")

	#clean up the oversample-thresholds argument
	if args.oversample_low_thresholds is not None:
		#convert the string to a list of floats
		args.oversample_low_thresholds = list(map(float, args.oversample_low_thresholds))
		#check if the values are between 0 and 1
		for value in args.oversample_low_thresholds:
			if value <= 0 or value >= 0.5:
				parser.error("--oversample-low-thresholds must be a list of floats between 0 and 0.5")

	
	
	if args.input_length <= 0:
		parser.error("--input-length must be greater than 0")

	if args.output_length <= 0:
		parser.error("--output-length must be greater than 0")

	if args.output_offset < 0:
		parser.error("--output-offset must be greater than or equal to 0")

	if args.batch_size <= 0:
		parser.error("--batch-size must be greater than 0")

	if args.test_num is not None and args.test_num < 0:
		parser.error("--test-num must be greater than or equal to 0")

	if args.nan_fill not in NAN_FILL_METHODS:
		parser.error(f"--nan-fill must be one of: {', '.join(NAN_FILL_METHODS)}")

	try:
		evaluation.resolve_metric_groups(args.metrics)
	except ValueError as e:
		parser.error(f"--metrics: {e}")
	args.metrics = [g.strip() for g in str(args.metrics).split(",") if g.strip()]

	try:
		dh.resolve_time_features(args.time_features, True, args.trend_period_years)
	except ValueError as e:
		parser.error(f"--time-features: {e}")
	if args.seasonal_period < 0:
		parser.error("--seasonal-period must be >= 0 (0 disables the seasonal-naive baseline)")

	try:
		dh.parse_change_lags(args.change_lags)
	except ValueError as e:
		parser.error(f"--change-lags: {e}")
	try:
		dh.parse_change_columns(args.change_columns)
	except re.error as e:
		parser.error(f"--change-columns: invalid regular expression ({e})")

	if args.trend_period_years < 0:
		parser.error("--trend-period-years must be >= 0 (0 disables the trend features)")

	if args.transform_method not in ['none', 'standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power']:
		parser.error("--transform-method must be one of: none, standard, minmax, robust, maxabs, quantile, power")

	if args.train_ratio < 0 or args.train_ratio > 1:
		parser.error("--train-ratio must be between 0 and 1")
	
	if args.test_ratio < 0 or args.test_ratio > 1:
		parser.error("--test-ratio must be between 0 and 1")
	
	if args.val_ratio < 0 or args.val_ratio > 1:
		parser.error("--val-ratio must be between 0 and 1")
	
	ratio_sum = args.train_ratio + args.test_ratio + args.val_ratio
	if not math.isclose(ratio_sum, 1.0, abs_tol=1e-9):
		parser.error(f"--train-ratio + --test-ratio + --val-ratio must equal 1 (they sum to {ratio_sum:g})")

	return args

def build_parser():
	# Create an ArgumentParser object
	parser = argparse.ArgumentParser(description="Train (or load) an LSTM forecaster, test it and append the "
		"metrics to <output-dir>/results.csv. See README.md.")

	# Add required arguments to the parser
	parser.add_argument("filename", nargs="?", default=None, type=str, help="Input data file (parquet, CSV or TSV), or data.file in --config.")
	parser.add_argument("testname", nargs="?", default=None, type=str, help="Name of the experiment, or run.name in --config. Used for the default output folder.") 
	parser.add_argument("--config", type=str, default=None, help="YAML file with the settings (see configs/template.yaml); command-line options override it.")
	parser.add_argument("--mode", type=str, default="train", choices=["train", "tune", "evaluate", "explain"], help="What run.py does: train, tune (search, then train the final model), evaluate (a saved model) or explain (SHAP). Default: train.")
	parser.add_argument("--final-repeats", type=int, default=1, help="run.py --mode tune: number of final models trained with the best settings. Default: 1.")

	# Add optional arguments to the parser

	# Setup parameters
	parser.add_argument("--train", 							action="store", default="True", help="True: train a new model. False: evaluate the model in --trainedModelFile. Default: True.")
	parser.add_argument("--hyperparameter-optimization", type=lambda x: None if x.lower() == 'none' else int(x), default=None, help="Run an Optuna search with this many trials instead of one training run. Default: None (off).")
	parser.add_argument("--hyperparameter-optimization-timeout", type=lambda x: None if str(x).lower() == 'none' else float(x), default=None, help="Stop starting new Optuna trials after this many seconds. Default: None (no limit).")
	parser.add_argument("--hyperparameter-optimization-db-dir", type=str, default=None, help="Directory for the Optuna database; a shared directory must already exist. Default: <output-dir>/optuna.")
	parser.add_argument("--search-space", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "optuna_search_space.json"), help="JSON file with the Optuna search space (see search_space.py). Default: configs/optuna_search_space.json.")
	parser.add_argument("--output-dir", type=str, default=None, help="Folder for everything the run writes; several runs can share it. Default: outputs/<testname>.")
	parser.add_argument("--hyperparameter-optimization-repeat", type=int, default=1, help="Number of times each Optuna trial is trained, with different seeds. Default: 1.")
	parser.add_argument("--hyperparameter-optimization-repeat-method", type=str, default="worst", help="How the repeats of a trial are combined into its score: worst, median, mean, best, better_midrank, worse_midrank. Default: worst.")
	parser.add_argument("--hyperparameter-optimization-n-jobs", type=int, default=1, help="Number of Optuna trials run in parallel in this process. Default: 1.")

	parser.add_argument("--trainedModelFile", 				type=str, help="Saved model folder to evaluate when --train False.")
	parser.add_argument("--suppress-warnings", 			action="store", default="False", help="Hide Python warnings (True or False). Default: False.")
	parser.add_argument("--save-model", 					action="store", default="True", help="Save the trained model to <output-dir>/models/ (True or False). Default: True.")
	parser.add_argument("--test-num", 						type=int, default=None, help="Repeat number of this run; the random seed is 42 + test-num. Default: None (seed 42).")
	# Data parameters		
	parser.add_argument("--label-tag", 						type=str, default="_Label", help="Every column whose name contains this text is a target. Default: _Label.")
	parser.add_argument("--date-column", 					type=str, default="Date", help="Name of the time column. Default: Date.")
	parser.add_argument("--input-length", 					type=int, default=24, help="Input sequence length: how many timesteps (Hours or Days) of data to use for prediction")
	parser.add_argument("--output-length", 					type=int, default=2, help="Output sequence length: how many timesteps (Hours or Days) of data to predict")
	parser.add_argument("--output-offset", 					type=int, default=0, help="Output sequence offset (default: 0): how many timesteps (Hours or Days) to offset the output sequence from the input sequence. E.g. for one prediction 12 timesteps ahead set output-offset to 12 and output-length to 1.")
	parser.add_argument("--half-float", 					action="store", default="auto",help="Train in mixed precision (float16 compute, float32 weights): auto, True or False. Default: auto (on when a GPU is found).")
	parser.add_argument("--oversample-high-thresholds", 	type=lambda x: None if x.lower() == 'none' else list(map(float, x.split(','))), default=None, help="Oversample the data by duplicating rows where the label is greater than the threshold. Default is None, values must be between 0.5 and 1. If set, provide a list of floats separated by commas. For example: 0.75,0.9,0.99")
	parser.add_argument("--oversample-low-thresholds", 		type=lambda x: None if x.lower() == 'none' else list(map(float, x.split(','))), default=None, help="Oversample the data by duplicating rows where the label is lower than the threshold. Default is None, values must be between 0 and 0.5. If set, provide a list of floats separated by commas. For example: 0.25,0.1,0.01")
	parser.add_argument("--nan-fill", 						type=str, default="min_std", help="How gaps (NaN) are filled, fitted on the training period. Options: ffill, zero, mean, median, median_mask, min_std, min_std_2, min_std_4, interpolate, interpolate_time, interpolate_polynomial, interpolate_spline, mean_previous_next, rolling_mean_6, rolling_mean_12, rolling_mean_24, knn, mice, kalman, missforest, missforest_singleton_minstd, kalman_singleton_minstd, none. Default: min_std (training minimum - training std, so gaps stand out from observed values; see README).")
	parser.add_argument("--transform-method", 				type=str, default="minmax", help="How each column is scaled, fitted on the training period. Options: none, standard, minmax, robust, maxabs, quantile, power. Default: minmax.")
	parser.add_argument("--trend-period-years", 			type=int, default=0, help="Period in years of the 'Year sin/cos' trend features; 0 leaves them out. Default: 0.")
	parser.add_argument("--time-features", 				type=str, default="auto", help="Calendar input features, comma-separated: time_of_day, day_of_year, day_of_week, trend; or auto or none. Default: auto (day_of_year, plus time_of_day for sub-daily data and trend when --trend-period-years > 0).")
	parser.add_argument("--change-lags", 					type=str, default="1,3,6", help="Signed-log change input features: for every target and lag k, sign(d)*log1p(|d|/s) with d = the change over the last k steps and s = the median absolute 1-step change on the training period. Computed from the raw values before gap filling and scaling. Comma-separated lags, or none to turn all change features off (including --change-columns). Default: 1,3,6.")
	parser.add_argument("--change-columns", 				type=str, default=None, help="Regular expressions (comma-separated) of other input columns that also get a 1-step signed-log change feature, e.g. '-HG-' for the river stage of the GRWS data. Default: None.")
	parser.add_argument("--metrics", 					type=str, default="core", help="Metric groups written to results.csv, comma-separated: core, hydrology, peaks, relative, or all. Default: core.")
	parser.add_argument("--seasonal-period", 				type=int, default=0, help="Season length in time steps for a seasonal-naive reference forecast (e.g. 24 for hourly data); 0 disables it. Default: 0.")
	parser.add_argument("--train-ratio", 					type=float, default=0.7, help="Fraction of the rows, from the start, used for training. Default: 0.7.")
	parser.add_argument("--val-ratio", 						type=float, default=0.15, help="Fraction of the rows after the training period used for validation. Default: 0.15.")
	parser.add_argument("--test-ratio", 					type=float, default=0.15, help="Fraction of the rows at the end used for testing; the three ratios must sum to 1. Default: 0.15.")

	# Model parameters			
	parser.add_argument("--model-type", 					type=str, default="lstm", help="Model architecture, a name in model.MODEL_BUILDERS. Default: lstm.")
	parser.add_argument("--decoder", 						action="store", default="False", help="False: one Dense layer maps the last LSTM state to all output steps. True: encoder-decoder LSTM. Default: False.")
	parser.add_argument("--num-layers", 					type=lambda x: max(int(x), 1), default=2, help="Number of stacked LSTM layers (minimum 1). Default: 2.")
	parser.add_argument("--units-per-layer",				type=int, default=512, help="Units per LSTM layer. Default: 512.")
	parser.add_argument("--dropout-rate", 					type=float, default=0.4, help="Dropout between LSTM layers and before the output layer. Default: 0.4.")
	parser.add_argument("--input-dropout", 					type=float, default=0.0, help="Dropout on the input features. Default: 0.")
	parser.add_argument("--loss-function", 					type=str, default="mse", help="Loss function to use. Default is mse. Options are: mse, mae, mape, huber, nse, kge. Note that KGE and NSE are actually (1 - KGE) and (1 - NSE) respectively (i.e. lower is better).")

	parser.add_argument("--noise", 							type=float, default=0.0, help="Std of Gaussian noise added to non-zero scaled inputs during training; 0 disables it. Default: 0.0.")
	parser.add_argument("--weight-init", 					type=str, default="glorot_uniform", help="Kernel initializer of the LSTM and output layers: zeros, ones, random_normal, random_uniform, glorot_normal, glorot_uniform, he_normal, he_uniform. Default: glorot_uniform.")
	parser.add_argument("--gradient-clip", 					type=str, default=None, help="Clip gradients to this global norm (e.g. 1.0). Default: None (no clipping).")
	parser.add_argument("--l1-regularization", 				type=str, default=None, help="L1 penalty on the LSTM input kernels (e.g. 1e-6). Default: None (off).")
	parser.add_argument("--l2-regularization", 				type=str, default=None, help="L2 penalty on the LSTM input kernels (e.g. 1e-6). Default: None (off).")
	parser.add_argument("--l1-recurrent-regularization", 	type=str, default=None, help="L1 penalty on the LSTM recurrent kernels (e.g. 1e-6). Default: None (off).")
	parser.add_argument("--l2-recurrent-regularization", 	type=str, default=None, help="L2 penalty on the LSTM recurrent kernels (e.g. 1e-6). Default: None (off).")
	parser.add_argument("--l1-dense-regularization", 		type=str, default=None, help="L1 penalty on the output layer kernel (e.g. 1e-6). Default: None (off).")
	parser.add_argument("--l2-dense-regularization", 		type=str, default=None, help="L2 penalty on the output layer kernel (e.g. 1e-6). Default: None (off).")
	parser.add_argument("--batch-normalization", 			action="store", default="False", help="Add LayerNormalization (not BatchNormalization) after every LSTM layer except the last (True or False). Default: False.")

	# Training parameters
	parser.add_argument("--mirrored", 						action="store", default="False", help="Use MirroredStrategy when more than one GPU is visible; the batch size is multiplied by the number of GPUs (True or False). Default: False.")
	parser.add_argument("--predict", 						type=str, default="change", choices=["level", "change", "both"], help="What the model outputs for each target: change (from the last observed value), level, or both (two heads, scored separately). Default: change.")
	parser.add_argument("--deterministic", 				action="store", default="False", help="Use deterministic TensorFlow ops, so runs with the same --test-num give identical results (True or False). Default: False.")
	parser.add_argument("--num-gpu-splits", 				type=int, default=1, help="Number of runs sharing one GPU; > 1 enables GPU memory growth. Default: 1.")
	parser.add_argument("--cur-gpu-split",					type=int, default=0, help="Index of this run among the runs sharing a GPU, also used in the Optuna database name. Default: 0.")
	parser.add_argument("--patience", 						type=int, default=-1, help="Stop training when --patience-metric has not improved for this many epochs after --min-epochs; -1 disables early stopping. Default: -1.")

	parser.add_argument("--patience-metric", 				type=str, default="val_loss", help="Validation metric for early stopping and the best epoch: val_loss, val_mse, val_mae, val_mape, val_rmse or val_nse. Default: val_loss.")
	parser.add_argument("--max-epochs", 					type=int, default=60, help="Maximum number of training epochs. Default: 60.")
	parser.add_argument("--min-epochs", 					type=int, default=10, help="No early stopping before this epoch. Default: 10.")
	parser.add_argument("--learning-rate", 					type=float, default=None, help="Adam learning rate; -1 uses a triangular cyclical schedule (0.0005 to 0.005). Default: None (Adam's default, 0.001).")
	parser.add_argument("--batch-size", 					type=int, default=32, help="Training batch size (multiplied by the number of GPUs with --mirrored True). Default: 32.")
	return parser


if __name__ == "__main__":
	print("ts-forecaster: LSTM time-series forecasting")
	print(f"The date/time is: {datetime.datetime.now()}")
	print("Command line arguments: ", sys.argv)

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[logging.StreamHandler(sys.stdout)],  # Send logs to stdout
		force=True  # replace handlers added by libraries on import
	)
	print(f"Root logger has handlers: {logging.getLogger().handlers}")


	logging.info("start")
	start = time.time()
	parser = build_parser()
	run_config.apply_config(parser)

	# Parse the command-line arguments
	args = parser.parse_args()
	print(args)
	clean_args(parser, args)
	if args.suppress_warnings.lower() == "true":
		warnings.filterwarnings("ignore")
	

	end = time.time()
	print("time to parse args: ", end - start)

	# Call the main function with the parsed arguments
	__main(args)
	logging.info("finished")
