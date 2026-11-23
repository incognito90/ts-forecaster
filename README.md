# ts-forecaster

LSTM forecasting for multivariate time series. You give it a table with a time
column, the column(s) you want to forecast and any other input columns. It
forecasts the next *N* steps of the targets from the previous *M* steps of all
columns and scores the result against simple reference forecasts
(persistence and climatology). It can also tune its own settings with Optuna
and explain its forecasts with SHAP.

It was built for hourly streamflow forecasting, but nothing in it is specific
to streamflow. Example river-flow and air-quality data are included.

## Requirements

An NVIDIA GPU and [Apptainer](https://apptainer.org) (or Singularity). The
container provides TensorFlow 2.13 with CUDA 11.8 and the pinned Python
packages in `requirements.txt`.

## Install

```bash
git clone <this repository> ts-forecaster && cd ts-forecaster
apptainer build ts-forecaster.sif Singularity.def
```

## Quick start

```bash
apptainer exec --nv ts-forecaster.sif python3 run.py --config configs/examples/grws_leggatt.yaml
```

This forecasts the next 12 hours of river flow at the Leggatt gauge (Grand
River, Ontario) from the previous 72 hours of flow, stage and weather. When it
finishes, open `outputs/grws_leggatt/results.csv`: one row per lead time
(1-12 h) with the model's scores (`mae`, `rmse`, `NSE`, `KGE`, ...) and the
same scores for the reference forecasts (`Persistence mae`, `Climatology mae`,
...). A useful model beats persistence, especially at the longer lead times.

The second example forecasts PM2.5 in Beijing 24 hours ahead:

```bash
apptainer exec --nv ts-forecaster.sif python3 run.py --config configs/examples/air_quality.yaml
```

The commands below are shown as `python3 run.py ...`; run them the same way,
inside the container (`apptainer exec --nv ts-forecaster.sif python3 run.py ...`).

## Your own data

1. Make a table (Parquet, CSV or tab-separated) with
   - a `Date` column (any format pandas can read; the time step is detected),
   - the target column(s), whose names end in a common tag such as `_Label`
     (e.g. `sales_Label`),
   - any numeric input columns. Past values of the targets are always used as
     inputs too. Leave missing values empty.
2. Copy one of the example configs and change the `data` section:
   ```yaml
   run:
     name: my_experiment
   data:
     file: data/my_series.csv
     label_tag: _Label        # marks the target column(s)
     input_length: 48         # time steps of history used as input
     output_length: 12        # time steps to forecast
   ```
3. Run `python3 run.py --config my_config.yaml`. Results are written to
   `outputs/my_experiment/`.

## Settings

Every setting, with its default and a short description, is in
[`configs/template.yaml`](configs/template.yaml). A config only needs the
settings you want to change. Settings can also be given on the command line,
which overrides the config:

```bash
python3 run.py --config my_config.yaml --max-epochs 5 --input-length 168
```

Each run saves the settings it used in `<output folder>/configs/`, so any run
can be repeated from that file. `test_num` sets the random seed
(42 + test_num); use `--deterministic True` for identical repeats on a GPU.

## What a run does

1. Reads the table, picks the targets and adds calendar features (time of day
   and day of year by default) and change features (below).
2. Splits the rows by time: 70% training, 15% validation, 15% test.
3. Fills gaps and scales every column, using statistics from the training
   period only.
4. Cuts the series into windows of `input_length` inputs and `output_length`
   targets.
5. Trains the LSTM to predict the change of each target from its last
   observed value (`predict: change`; `level` predicts the values directly).
   Training stops early when the validation score stops improving, and the
   best epoch is kept.
6. Forecasts the test period and scores each lead time against the observed
   values.

## Change features

Scaled to 0–1, a typical hourly change of a river's flow is about 0.0001, too
small for the model to use the recent trend. So, from the raw values, every
target gets input columns with its recent change over `change_lags` steps
(default `[1, 3, 6]`):

    sign(d) · log(1 + |d| / s),   d = x(t) − x(t−k)

where `s` is the median absolute 1-step change in the training period. Small
changes stay distinguishable and large ones are compressed. These columns
(`<target>_d<k>_change`) are inputs only. A missing value gives a missing
change, which is filled like any other gap. `change_columns` adds the 1-step
change of other columns, given as regular expressions (the Leggatt example
uses the river stage, `"-HG-"`). `change_lags: none` turns all change
features off.

## Missing values

By default (`nan_fill: min_std`, `transform_method: minmax`) a gap is filled
with the training minimum minus one standard deviation, which becomes exactly 0
after scaling, while every observed value is above 0. The model can therefore
tell a missing value from an observed one. Missing target values are left out
of the training loss and of the scores, and forecasts start from the last
observed value. This method is described in Kemp et al. (2026), cited below.

Other fill methods include `ffill`, `mean`, `median`, `interpolate`, `knn`,
`mice`, `kalman` and `missforest` (full list in `configs/template.yaml`).

## Results

`results.csv` has one row per target and lead time. `evaluation.metrics`
chooses which scores are written:

| Group | Scores |
|---|---|
| `core` (default) | `mae`, `rmse`, `mse`, `r2`, `smape`, `mase`, `bias`, `Pearson Correlation` |
| `hydrology` | `NSE`, `KGE`, `alpha`, `beta`, `pbias`, `rve`, `fdc`, `nmse`, `model efficiency` |
| `peaks` | peak recall, peak false-discovery rate and peak-flow error at the 75th/90th/95th/99th percentiles |
| `relative` | `mape`, `msle` (positive values only) |

The reference forecasts are scored on the same windows as the model:

- **Persistence:** the last observed value.
- **Climatology:** the training-period mean.
- **Seasonal naive:** with `seasonal_period: N`, the value N steps earlier.

The skill against persistence is `1 - mse / Persistence mse`.

Several runs can write to the same output folder; their rows are appended to
the same `results.csv`.

## Tuning

```bash
python3 run.py --config my_config.yaml --mode tune --hyperparameter-optimization 50
```

This runs an Optuna search with 50 trials. The search space defaults to
[`configs/optuna_search_space.json`](configs/optuna_search_space.json) and can
also be written in the config (see [`search_space.py`](search_space.py)). Trials
are compared on validation MSE.

Each trial can be trained several times with different seeds
(`tuning.repeats`). Its score is then its worst repeat (`tuning.repeat_method`,
default `worst`), and the trial is pruned as soon as any repeat falls behind
the median of the finished trials at the same epoch.

The best settings are written to `<output folder>/best_config.yaml`, and the
final model is trained from it into `<output folder>/final/`. Running the same
search again resumes it.

## Explaining forecasts (SHAP)

Train with the model saved (the default), then:

```bash
python3 run.py --config my_config.yaml --mode explain --saved-model outputs/my_experiment/models/<model> --run-shap True
```

This computes SHAP values for random, high-target or low-target test windows
(`shap_sample_mode`) at chosen lead times (`shap_horizon`). Summaries per
feature, per feature group and per input lag go to `<output folder>/shap/`.
The `explain` section of `configs/template.yaml` lists the options.

## Tests

```bash
python3 -m pytest -q
```

## Data and citation

The example data and their licences are described in
[`data/README.md`](data/README.md). The streamflow data come from the Grand
River Watershed Streamflow datasets (https://doi.org/10.5683/SP3/CH3T6Z) and
contain information made available under Grand River Conservation Authority's
Open Data Licence v2.0.

If you use the `min_std` gap filling, please cite:

> Kemp, J., Kremer, S. C., Ali, G. and Gillis, D. (2026). Learning from Missing
> Values: Encoding Missingness in Representation-Space for LSTM Time Series
> Forecasting. *Transactions on Machine Learning Research*.
> https://openreview.net/forum?id=DMmCMIrrez

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data: see [`data/README.md`](data/README.md).
