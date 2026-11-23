"""Optuna search space, read from a JSON file (default configs/optuna_search_space.json).

    {"parameters": [
      {"name": "num_layers", "type": "int", "low": 1, "high": 3},
      {"name": "use_l1", "type": "categorical", "choices": [true, false]},
      {"name": "l1_regularization", "type": "float", "low": 1e-10, "high": 1e-4, "log": true, "only_if": "use_l1"}]}

Entries are sampled in file order. A name is a key of SEARCHABLE or an on/off switch
(use_* / set_*); low/high are inclusive (optional "step"); only_if names an earlier
switch, and while it is false the setting is off (OPTIONAL). Settings not listed
keep their command-line value.
"""
import json

# searchable settings and the command-line option each one falls back to
SEARCHABLE = {
	"num_layers": "--num-layers",
	"units_per_layer": "--units-per-layer",
	"n_units_exponent": "--units-per-layer (as 2**value)",
	"dropout_rate": "--dropout-rate",
	"input_dropout": "--input-dropout",
	"loss": "--loss-function",
	"l1_regularization": "--l1-regularization",
	"l2_regularization": "--l2-regularization",
	"recurrent_l1_regularization": "--l1-recurrent-regularization",
	"recurrent_l2_regularization": "--l2-recurrent-regularization",
	"dense_l1_regularization": "--l1-dense-regularization",
	"dense_l2_regularization": "--l2-dense-regularization",
	"weight_init": "--weight-init",
	"batch_norm": "--batch-normalization",
	"decoder": "--decoder",
	"noise": "--noise",
	"patience": "--patience",
	"learning_rate": "--learning-rate",
	"gradient_clip": "--gradient-clip",
}


# settings that can be switched off with only_if: no regularization, no clipping,
# default learning rate, no input dropout, no noise
OPTIONAL = ["l1_regularization", "l2_regularization", "recurrent_l1_regularization",
	"recurrent_l2_regularization", "dense_l1_regularization", "dense_l2_regularization",
	"learning_rate", "gradient_clip", "input_dropout", "noise"]


def _is_switch(name):
	return name.startswith(("use_", "set_"))


def load_search_space(path):
	"""Read and check a search space (JSON/YAML file, or the inline list/dict from a config)."""
	if isinstance(path, (list, dict)):
		spec, path = path, "tuning.search_space"
	elif str(path).endswith((".yaml", ".yml")):
		import yaml
		with open(path) as f:
			spec = yaml.safe_load(f)
	else:
		with open(path) as f:
			spec = json.load(f)
	entries = spec.get("parameters") if isinstance(spec, dict) else spec
	if not isinstance(entries, list) or not entries:
		raise ValueError(f"{path}: expected a non-empty list of parameters (or an object with a 'parameters' list)")

	seen = set()
	for i, e in enumerate(entries):
		where = f"{path}: parameter #{i + 1} ({e.get('name', '?')})"
		name, kind = e.get("name"), e.get("type")
		if name in seen:
			raise ValueError(f"{where}: defined twice")
		if name not in SEARCHABLE and not _is_switch(name):
			raise ValueError(f"{where}: unknown name; searchable settings are {', '.join(SEARCHABLE)} "
				"(plus on/off switches named use_* or set_*)")
		if "units_per_layer" in seen and name == "n_units_exponent" or "n_units_exponent" in seen and name == "units_per_layer":
			raise ValueError(f"{where}: give either units_per_layer or n_units_exponent, not both")
		if kind in ("int", "float"):
			if "low" not in e or "high" not in e:
				raise ValueError(f"{where}: {kind} needs 'low' and 'high'")
			# yaml reads 1e-6 (no decimal point) as a string
			cast = int if kind == "int" else float
			try:
				e["low"], e["high"] = cast(e["low"]), cast(e["high"])
				if "step" in e and e["step"] is not None:
					e["step"] = cast(e["step"])
			except (TypeError, ValueError):
				raise ValueError(f"{where}: 'low', 'high' and 'step' must be numbers")
			if e["low"] > e["high"]:
				raise ValueError(f"{where}: {kind} needs 'low' <= 'high'")
		elif kind == "categorical":
			if not isinstance(e.get("choices"), list) or not e["choices"]:
				raise ValueError(f"{where}: categorical needs a non-empty 'choices' list")
		else:
			raise ValueError(f"{where}: type must be int, float or categorical")
		gate = e.get("only_if")
		if gate is not None and (gate not in seen or not _is_switch(gate)):
			raise ValueError(f"{where}: only_if '{gate}' must name an on/off switch defined earlier")
		if gate is not None and name not in OPTIONAL:
			raise ValueError(f"{where}: only_if can only switch off {', '.join(OPTIONAL)}")
		seen.add(name)
	return entries


def sample(trial, entries):
	"""Sample every entry in file order; switched-off settings are None."""
	values = {}
	for e in entries:
		name, gate = e["name"], e.get("only_if")
		if gate is not None and not values[gate]:
			values[name] = None
			continue
		if e["type"] == "int":
			values[name] = trial.suggest_int(name, e["low"], e["high"], step=e.get("step", 1), log=e.get("log", False))
		elif e["type"] == "float":
			values[name] = trial.suggest_float(name, e["low"], e["high"], step=e.get("step"), log=e.get("log", False))
		else:
			values[name] = trial.suggest_categorical(name, e["choices"])
	return values


def best_settings(params, entries):
	"""Trial params -> {argparse dest: value}, e.g. n_units_exponent=7 -> units_per_layer=128."""
	settings = {}
	for e in entries:
		name = e["name"]
		if _is_switch(name):
			continue
		gate = e.get("only_if")
		if gate is not None and not params.get(gate, False):
			value = None
		else:
			value = params.get(name)
		if name == "n_units_exponent":
			settings["units_per_layer"] = None if value is None else 2 ** int(value)
			continue
		dest = SEARCHABLE[name].split()[0].lstrip("-").replace("-", "_")
		if value is None and dest in ("input_dropout", "noise"):
			value = 0.0  # switched off: no input dropout / no noise layer
		settings[dest] = value
	return settings
