"""NaN imputation for the feature/label DataFrames.

fit() on the training split and transform() val/test with the same statistics or
models, so nothing from val/test leaks into the fill values. The method is chosen
with --nan-fill. The default, min_std, fills with (training min - std), a value
below the observed range (Kemp et al. 2026, TMLR, https://openreview.net/forum?id=DMmCMIrrez).
"""

import pandas as pd
import numpy as np

class NaNImputer:
	"""
	A utility class for handling NaN imputation using various strategies.
	"""

	def __init__(self, method="ffill", **kwargs):
		"""exclude_columns are never imputed; time_index_column is excluded too and used
		by interpolate_time. Other kwargs are read by the individual methods.
		"""
		self.method = method

		self.kwargs = kwargs
		self.exclude_columns = list(kwargs.pop("exclude_columns", []))
		self.time_index_column = kwargs.pop("time_index_column", None)

		if self.time_index_column:
			if self.time_index_column in self.exclude_columns:
				raise ValueError(f"'{self.time_index_column}' cannot be in exclude_columns.")
			self.exclude_columns.append(self.time_index_column)


		if method not in [
			"ffill", "mean", "median", "median_mask", "zero", "interpolate",
			"min_std","min_std_4","min_std_2", "interpolate_time", "interpolate_polynomial", "interpolate_spline",
			"mean_previous_next", "rolling_mean_24", "rolling_mean_12", "rolling_mean_6",
			"knn", "mice", "kalman", "missforest","missforest_singleton_minstd","kalman_singleton_minstd"
		]:
			raise ValueError(f"Unsupported imputation method: {method}")

	def fit(self, df, **kwargs):
		"""Fit the statistics / model of the chosen method on df (the training split)."""
		self.kwargs.update(kwargs)
		features = df.drop(columns=self.exclude_columns, errors='ignore')
		self.impute_columns_ = features.columns.tolist()

		if self.method == "mean":
			self.statistics_ = features.mean(axis=0)
		elif self.method == "median" or self.method == "median_mask":
			self.statistics_ = features.median(axis=0)
		elif self.method == "min_std" or self.method == "min_std_4" or self.method == "min_std_2":
			self.statistics_ = {col: (features[col].min(), features[col].std()) for col in features.columns}
		elif self.method == "knn":
			from sklearn.impute import KNNImputer
			self.model_ = KNNImputer(n_neighbors=self.kwargs.get("n_neighbors", 5))
			self.model_.fit(features)
		elif self.method == "mice":
			from sklearn.experimental import enable_iterative_imputer
			from sklearn.impute import IterativeImputer
			self.model_ = IterativeImputer()
			self.model_.fit(features)
		elif self.method == "kalman":
			self._kalman_fit(features)
		elif self.method == "missforest":
			self._missforest_fit(features)
		elif self.method == "missforest_singleton_minstd":
			# store min/std needed for the min_std part
			self.hybrid_minstd_stats_ = {col: (features[col].min(), features[col].std()) for col in features.columns}
			# fit MissForest model
			self._missforest_fit(features)
		elif self.method == "kalman_singleton_minstd":
			# store min/std needed for the min_std part
			self.hybrid_minstd_stats_ = {col: (features[col].min(), features[col].std()) for col in features.columns}
			# fit Kalman models
			self._kalman_fit(features)


		return self

	def transform(self, df, **kwargs):
		"""Impute NaNs in df with the fitted method and return a new DataFrame."""
		self.kwargs.update(kwargs)

		skip_columns = kwargs.get("skip_columns", [])
		current_exclude = list(set(self.exclude_columns + skip_columns))


		print(f"Starting imputation with method: {self.method}. keyword arguments: {self.kwargs}")

		if self.method in ["mean","median","median_mask","min_std","min_std_2","min_std_4"] and not hasattr(self, "statistics_"):
			raise ValueError("The imputer has not been fitted yet. Call fit() before transform().")
		if self.method in ["knn", "mice", "missforest", "kalman", "missforest_singleton_minstd", "kalman_singleton_minstd"] and not hasattr(self, "model_"):
			raise ValueError("The imputer has not been fitted yet. Call fit() before transform().")
		if self.method.endswith("_singleton_minstd") and not hasattr(self, "hybrid_minstd_stats_"):
			raise ValueError("The imputer has not been fitted yet. Call fit() before transform().")

		df_excluded = df[current_exclude] if current_exclude else pd.DataFrame(index=df.index)
		features = df[self.impute_columns_]  # still use only fit-time impute columns
		features = features.drop(columns=skip_columns, errors='ignore')

		has_mask = False
		#empty mask
		mask = pd.DataFrame(index=features.index)

		if self.method == "ffill":
			features = self._forward_fill(features)
		elif self.method == "mean":
			features = self._mean_impute(features)
		elif self.method == "median":
			features = self._median_impute(features)
		elif self.method == "median_mask":
			# Create mask columns
			for col in features.columns:
				mask_col = f"{col}_was_nan"
				mask = pd.concat([mask, features[col].isna().rename(mask_col)], axis=1)
			has_mask = True
			#Impute
			features = self._median_impute(features)
		elif self.method == "zero":
			features = self._zero_impute(features)
		elif self.method == "interpolate":
			features = self._interpolate(features)
		elif self.method == "min_std":
			features = self._min_std_impute(features)
		elif self.method == "min_std_4":
			features = self._min_std_impute(features, denominator=4)
		elif self.method == "min_std_2":
			features = self._min_std_impute(features, denominator=2)
		elif self.method == "interpolate_time":
			# needs the time column, so pass the full df
			interpolated = self._interpolate_time(df)[list(features.columns)]
			interpolated.index = features.index
			features = interpolated
		elif self.method == "interpolate_polynomial":
			features = self._interpolate_polynomial(features)
		elif self.method == "interpolate_spline":
			features = self._interpolate_spline(features)
		elif self.method == "mean_previous_next":
			features = self._mean_previous_next(features)
		elif self.method.startswith("rolling_mean_"):
			window = int(self.method.split("_")[-1])
			features = self._rolling_mean_impute(features, window)
		elif self.method == "knn":
			features = self._knn_impute(features)
		elif self.method == "mice":
			features = self._mice_impute(features)
		elif self.method == "kalman":
			features = self._kalman_impute(features)
		elif self.method == "missforest":
			features = self._missforest_impute(features)
		elif self.method == "missforest_singleton_minstd":
			features = self._missforest_singleton_minstd_impute(features, denominator=int(self.kwargs.get("hybrid_minstd_denominator", 1)))
		elif self.method == "kalman_singleton_minstd":
			features = self._kalman_singleton_minstd_impute(features, denominator=int(self.kwargs.get("hybrid_minstd_denominator", 1)))

		else:
			raise ValueError(f"Unsupported imputation method: {self.method}")


		nans_count = features.isna().sum().sum()
		if( nans_count > 0):
			print(f"WARNING: {nans_count} NaNs found in features after imputation")

		# Recombine
		if not has_mask:
			result = pd.concat([df_excluded, features], axis=1)
			result = result[df.columns]  # reorder
		else:
			# recombine and place each mask column immediately after its original feature
			result = pd.concat([df_excluded, features, mask], axis=1)
			cols = []
			for col in features.columns:
				cols.append(col)
				mask_col = f"{col}_was_nan"
				if mask_col in result.columns:
					cols.append(mask_col)
			# append any remaining excluded columns preserving their order
			for col in df_excluded.columns:
				if col not in cols:
					cols.append(col)
			result = result[cols]	# reorder


		print(f"Imputation completed using method: {self.method}")


		return result

	def fit_transform(self, df, **kwargs):
		self.fit(df, **kwargs)
		return self.transform(df, **kwargs)

	def _forward_fill(self, df):
		df.ffill(inplace=True)
		df.bfill(inplace=True)
		return df

	def _mean_impute(self, df):
		return df.fillna(self.statistics_)

	def _median_impute(self, df):
		return df.fillna(self.statistics_)

	def _zero_impute(self, df):
		return df.fillna(0)

	def _interpolate(self, df):
		df = df.select_dtypes(include=[np.number])	# ensure no datetime slips in
		df = df.interpolate(method="linear")
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df

	def _min_std_impute(self, df,denominator=1):
		"""Fill NaNs with training min - std / denominator (below the observed range)."""
		for col in df.columns:
			min_val, std_val = self.statistics_[col]
			# assign, inplace fillna on df[col] does nothing under copy-on-write (pandas 3)
			df[col] = df[col].fillna(min_val - (std_val/denominator))
		return df

	def _interpolate_time(self, df):
		if not self.time_index_column:
			raise ValueError("time_index_column must be set for interpolate_time method.")
		if self.time_index_column not in df.columns:
			raise ValueError(f"time_index_column '{self.time_index_column}' not found in DataFrame.")

		time_values = df[self.time_index_column]
		features = df.drop(columns=[self.time_index_column], errors='ignore')
		features = features.set_index(time_values)
		features = features.interpolate(method="time")
		features.reset_index(drop=True, inplace=True)
		features = features.fillna(method="ffill").fillna(method="bfill")
		features[self.time_index_column] = time_values.reset_index(drop=True)
		cols = [self.time_index_column] + [col for col in features.columns if col != self.time_index_column]

		return features[cols]

	def _interpolate_polynomial(self, df):
		order = self.kwargs.get("order", 2)
		for column in df.columns:
			if df[column].notna().sum() < order + 1:
				df[column] = df[column].interpolate(method="linear")
			else:
				df[column] = df[column].interpolate(method="polynomial", order=order)
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df

	def _interpolate_spline(self, df):
		df = df.interpolate(method="spline", order=self.kwargs.get("order", 2))
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df

	def _mean_previous_next(self, df):
		df = df.apply(lambda x: x.fillna((x.shift() + x.shift(-1)) / 2))
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df

	def _rolling_mean_impute(self, df, window):
		df = df.fillna(df.rolling(window=window, min_periods=1).mean())
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df

	def _knn_impute(self, df):
		df = pd.DataFrame(self.model_.transform(df), columns=df.columns, index=df.index)
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df

	def _mice_impute(self, df):
		df = pd.DataFrame(self.model_.transform(df), columns=df.columns, index=df.index)
		df = df.fillna(method="ffill").fillna(method="bfill")
		return df
		
	def _kalman_fit(self, features):
		"""Fit a local-level Kalman filter per column with EM (needs pykalman)."""
		try:
			from pykalman import KalmanFilter
		except ImportError:
			raise ImportError("Install 'pykalman' (pip install pykalman) to use kalman.")
		self.model_ = {}
		em_iter = int(self.kwargs.get("kalman_em_iter", 5)) # was 20
		col_count = features.shape[1]
		current = 0
		for col in self.impute_columns_:
			print(f"Fitting Kalman filter for column: {col}: {current+1}/{col_count}")
			current += 1
			y = features[col].astype(float).values
			y_ma = np.ma.masked_invalid(y)
			kf = KalmanFilter(
				transition_matrices=[1.0],
				observation_matrices=[1.0],
				transition_covariance=np.array([[1e-3]]),
				observation_covariance=np.array([[1.0]]),
				initial_state_mean=np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 0.0,
				initial_state_covariance=1.0
			).em(y_ma, n_iter=em_iter)
			self.model_[col] = {
				"transition_matrices": kf.transition_matrices,
				"observation_matrices": kf.observation_matrices,
				"transition_covariance": kf.transition_covariance,
				"observation_covariance": kf.observation_covariance,
				"initial_state_mean": kf.initial_state_mean,
				"initial_state_covariance": kf.initial_state_covariance,
			}
	
	def _kalman_impute(self, df):
		"""Replace NaNs with the Kalman-smoothed state; observed values are kept."""
		from pykalman import KalmanFilter
		out = df.copy()
		col_count = out.shape[1]
		current = 0
		for col in out.columns:
			print(f"Fitting Kalman filter for column: {col}: {current+1}/{col_count}")
			current += 1
			y = out[col].astype(float).values
			y_ma = np.ma.masked_invalid(y)
			params = self.model_.get(col)
			if params is None:
				s = pd.Series(y, index=out.index).ffill()
				if pd.isna(s.iloc[0]):
					s = s.bfill()
				if s.isna().all():
					s = pd.Series(0.0, index=out.index)
				out[col] = s
				continue
			kf = KalmanFilter(**params)
			mean, _ = kf.smooth(y_ma)
			imputed = np.where(np.isnan(y), mean[:, 0], y)
			s = pd.Series(imputed, index=out.index).ffill().bfill()
			if s.isna().all():
				s = pd.Series(0.0, index=out.index)
			out[col] = s
		return out

	def _missforest_fit(self, features):
		"""MissForest: IterativeImputer with a RandomForestRegressor (mf_* kwargs)."""
		from threadpoolctl import threadpool_limits
		from sklearn.experimental import enable_iterative_imputer	
		from sklearn.impute import IterativeImputer
		from sklearn.ensemble import RandomForestRegressor
		print("Fitting MissForest imputer")
		rf = RandomForestRegressor(
			n_estimators=self.kwargs.get("mf_n_estimators", 30),
			max_features=self.kwargs.get("mf_max_features", "sqrt"),
			min_samples_leaf=self.kwargs.get("mf_min_samples_leaf", 5),	
			random_state=self.kwargs.get("mf_random_state", 0),
			n_jobs=self.kwargs.get("mf_n_jobs", 1),
			max_depth=self.kwargs.get("mf_max_depth", 4),
			verbose=self.kwargs.get("mf_rf_verbose", 1)
		)
		print("RandomForestRegressor created")
		self.model_ = IterativeImputer(
			estimator=rf,
			initial_strategy=self.kwargs.get("mf_initial_strategy", "median"),
			max_iter=self.kwargs.get("mf_max_iter", 2),
			tol=self.kwargs.get("mf_tol", 1e-3),	
			skip_complete=True,
			sample_posterior=False,
			random_state=self.kwargs.get("mf_random_state", 0),
			verbose=self.kwargs.get("mf_verbose", 1)
		)
		print("Fitting MissForest model...")
		thread_cap = int(self.kwargs.get("mf_thread_limit", 1))
		with threadpool_limits(limits=thread_cap):
			self.model_.fit(features)

	def _missforest_impute(self, df):
		from threadpoolctl import threadpool_limits
		thread_cap = int(self.kwargs.get("mf_thread_limit", 1))
		with threadpool_limits(limits=thread_cap):
			imputed = self.model_.transform(df)
		return pd.DataFrame(imputed, columns=df.columns, index=df.index)

	def _split_nan_masks(self, df, min_run_len_for_block=2):
		# Returns two DataFrames of booleans: singleton_mask (len==1) and block_mask (len>=min_run_len_for_block)
		singleton_masks = {}
		block_masks = {}
		for col in df.columns:
            # boolean NaN vector
			nanb = df[col].isna().values
			if not nanb.any():
				singleton_masks[col] = pd.Series(False, index=df.index)
				block_masks[col] = pd.Series(False, index=df.index)
				continue
			# run-length encode over NaNs
			# identify starts of segments where NaN status changes
			switch = np.empty_like(nanb, dtype=bool)
			switch[0] = True
			switch[1:] = nanb[1:] != nanb[:-1]
			seg_ids = switch.cumsum()
			# lengths per segment
			_, seg_counts = np.unique(seg_ids, return_counts=True)
			# map seg length back to positions
			seg_len = seg_counts[seg_ids - 1]
			# singleton NaNs: NaN True & seg_len==1
			singleton = nanb & (seg_len == 1)
			# block NaNs: NaN True & seg_len>=min_run_len_for_block
			block = nanb & (seg_len >= min_run_len_for_block)
			singleton_masks[col] = pd.Series(singleton, index=df.index)
			block_masks[col] = pd.Series(block, index=df.index)
		return pd.DataFrame(singleton_masks), pd.DataFrame(block_masks)


	def _missforest_singleton_minstd_impute(self, df, denominator=1):
		"""Single NaNs get the MissForest value, runs of 2+ NaNs get min_std."""
		# 1) compute masks
		singleton_mask, block_mask = self._split_nan_masks(df, min_run_len_for_block=2)

		# 2) MissForest pass (for singletons)
		mf_imputed = pd.DataFrame(self.model_.transform(df), columns=df.columns, index=df.index)

		# 3) min_std values (for blocks)
		minstd_vals = {}
		for col in df.columns:
			min_val, std_val = self.hybrid_minstd_stats_[col]
			minstd_vals[col] = min_val - (std_val / max(1, denominator))
		minstd_fill = pd.Series(minstd_vals)

		# 4) assemble result column-wise
		out = df.copy()
		for col in df.columns:
			col_vals = out[col].values
			# where singleton NaN → take MissForest value
			sel_single = singleton_mask[col].values
			col_vals[sel_single] = mf_imputed[col].values[sel_single]
			# where block NaN → take min_std value
			sel_block = block_mask[col].values
			col_vals[sel_block] = minstd_fill[col]
			out[col] = col_vals

		# 5) sanity: if any NaNs remain (e.g., columns entirely NaN), back/forward fill as last resort
		if out.isna().values.any():
			out = out.fillna(method="ffill").fillna(method="bfill")
		return out

	def _kalman_singleton_minstd_impute(self, df, denominator=1):
		"""Single NaNs get the Kalman value, runs of 2+ NaNs get min_std."""
		# 1) compute masks
		singleton_mask, block_mask = self._split_nan_masks(df, min_run_len_for_block=2)

		# 2) Kalman pass (for singletons)
		kalman_imputed = self._kalman_impute(df)

		# 3) min_std values (for blocks)
		minstd_vals = {}
		for col in df.columns:
			min_val, std_val = self.hybrid_minstd_stats_[col]
			minstd_vals[col] = min_val - (std_val / max(1, denominator))
		minstd_fill = pd.Series(minstd_vals)

		# 4) assemble result column-wise
		out = df.copy()
		for col in df.columns:
			col_vals = out[col].values
			# where singleton NaN → take Kalman value
			sel_single = singleton_mask[col].values
			col_vals[sel_single] = kalman_imputed[col].values[sel_single]
			# where block NaN → take min_std value
			sel_block = block_mask[col].values
			col_vals[sel_block] = minstd_fill[col]
			out[col] = col_vals
		# 5) sanity: if any NaNs remain (e.g., columns entirely NaN), back/forward fill as last resort
		if out.isna().values.any():
			out = out.fillna(method="ffill").fillna(method="bfill")
		return out