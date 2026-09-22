#!/usr/bin/env python3
# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate that `jax_privacy.saliency.topk_vote_probe` produces useful masks.

Runs two DP-SGD fine-tunes of Gemma3 under matched (eps, delta):

  (1) baseline: the default LoRA layers selected by keras_hub's
      `enable_lora()` policy (typically query+value projections)
  (2) DP-SAPF: LoRA on the top-`--top_k_percent`% attention layers selected
      by the DP probe

Reports ROUGE-1/2/L for both and the delta. This script provides an end-to-end
validation path for iterating on the probe implementation.

Typical run (single GPU):
  python examples/dpsapf_validate.py \\
      --dataset cnn_dailymail --test_run \\
      --top_k_percent 5 --total_epsilon 4.0

`--test_run` swaps to Gemma3 1B. Drop it for the full 4B model.
Keras defines the model and loss; `jax_privacy.training.DPTrainer` performs
Poisson sampling, clipping, noise addition, and Optax updates. This requires
the current `DPTrainer(config=..., performance_flags=...)` API. Each reported
budget covers the probe and one fine-tune, not the joint release of both runs.
"""

import argparse
import os

# Deferred imports below: the heavy JAX/Keras/TF stack is imported inside
# functions so `os.environ['KERAS_BACKEND']` and cache-dir env vars can be
# set first (they must be in place before `import keras`).
# pylint: disable=import-outside-toplevel


DEFAULT_CACHE_ROOT = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "jax_privacy",
)


# ---------------------------------------------------------------------------
# Dataset registry + Keras/LoRA helpers (example-side glue; the DP mechanism
# itself lives in `jax_privacy.saliency`).
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "samsum": {
        "loader": "tfds",
        "tfds_name": "samsum",
        "input_field": "dialogue",
        "output_field": "summary",
        "prompt_prefix": "Summarize the following dialogue:\n",
        "prompt_suffix": "\nSummary:\n",
        "note": (
            "~14.7K dialogue/summary pairs. Requires a manual `corpus.7z` "
            "download to $TFDS_DATA_DIR/downloads/manual/."
        ),
    },
    "cnn_dailymail": {
        "loader": "tfds",
        "tfds_name": "cnn_dailymail",
        "input_field": "article",
        "output_field": "highlights",
        "prompt_prefix": "Summarize the following news article:\n",
        "prompt_suffix": "\nHighlights:\n",
        "note": (
            "~287K news articles -> multi-sentence highlights. "
            "Long inputs. Requires `pip install beautifulsoup4 lxml`."
        ),
    },
    "xsum_hf": {
        "loader": "hf",
        "hf_name": "EdinburghNLP/xsum",
        "input_field": "document",
        "output_field": "summary",
        "prompt_prefix": "Summarize the following article in one sentence:\n",
        "prompt_suffix": "\nSummary:\n",
        "note": "~204K BBC articles -> 1-sentence summaries via HuggingFace.",
    },
}


def _make_source_to_gemma3_format(cfg):
  """tf.data.map fn emitting {prompts, responses} string dicts."""
  import tensorflow as tf

  in_field, out_field = cfg["input_field"], cfg["output_field"]
  prefix, suffix = cfg["prompt_prefix"], cfg["prompt_suffix"]

  def fn(d):
    return {
        "prompts": tf.strings.join([prefix, d[in_field], suffix]),
        "responses": d[out_field],
    }

  return fn


def _load_dataset_split(cfg, split_spec):
  """TFDS vs HuggingFace dispatcher.

  Returns a tf.data.Dataset with known cardinality (needed for DP accounting
  and reasonable batching downstream).
  """
  import tensorflow as tf
  import tensorflow_datasets as tfds

  if cfg.get("loader") == "hf":
    try:
      from datasets import load_dataset  # pytype: disable=import-error
    except ImportError as e:
      raise ImportError(
          "The HuggingFace `datasets` package is required for the xsum_hf "
          "dataset; install it with `pip install datasets`."
      ) from e
    in_field, out_field = cfg["input_field"], cfg["output_field"]
    hf_ds = load_dataset(cfg["hf_name"], split=split_spec)

    def gen():
      for ex in hf_ds:
        yield {in_field: ex[in_field], out_field: ex[out_field]}

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            in_field: tf.TensorSpec(shape=(), dtype=tf.string),
            out_field: tf.TensorSpec(shape=(), dtype=tf.string),
        },
    ).apply(tf.data.experimental.assert_cardinality(len(hf_ds)))

  return tfds.load(cfg["tfds_name"], split=split_spec)


def _get_lora_candidate_layers(backbone, attn_only=False):
  """Dense / EinsumDense sublayers of `backbone` eligible for a LoRA adapter."""
  import keras

  out, seen = [], set()
  # pylint: disable-next=protected-access
  for layer in backbone._flatten_layers(recursive=True, include_self=False):
    if id(layer) in seen:
      continue
    if not isinstance(layer, (keras.layers.Dense, keras.layers.EinsumDense)):
      continue
    if not (hasattr(layer, "kernel") and hasattr(layer, "enable_lora")):
      continue
    if attn_only and "attention" not in layer.path:
      continue
    seen.add(id(layer))
    out.append(layer)
  return out


def _enable_lora_on_paths(backbone, paths, rank):
  """Enable LoRA on layers whose `.path` is in `paths`; freeze the rest."""
  p2l = {
      l.path: l for l in _get_lora_candidate_layers(backbone, attn_only=False)
  }
  ids = {id(p2l[p]) for p in paths if p in p2l}
  backbone.trainable = True
  backbone._lora_rank = rank  # pylint: disable=protected-access
  # pylint: disable-next=protected-access
  for layer in backbone._flatten_layers(include_self=False):
    if id(layer) in ids:
      layer.trainable = True
      layer.enable_lora(rank=rank)
      bias = getattr(layer, "bias", None)
      if bias is not None:
        bias.trainable = False
    else:
      layer.trainable = False


def _set_env_defaults(cache_root):
  os.makedirs(cache_root, exist_ok=True)
  for k, sub in [
      ("KERAS_HOME", "keras"),
      ("KAGGLEHUB_CACHE", "kagglehub"),
      ("TFDS_DATA_DIR", "tfds"),
      ("HF_HOME", "huggingface"),
      ("JAX_COMPILATION_CACHE_DIR", "jax_compilation_cache"),
  ]:
    os.environ.setdefault(k, os.path.join(cache_root, sub))
    os.makedirs(os.environ[k], exist_ok=True)
  os.environ["KERAS_BACKEND"] = "jax"
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--cache_root",
      default=DEFAULT_CACHE_ROOT,
      help=(
          "Root for default caches (default: %(default)s). Existing cache "
          "environment variables take precedence."
      ),
  )
  p.add_argument(
      "--dataset",
      default="cnn_dailymail",
      choices=["samsum", "cnn_dailymail", "xsum_hf"],
  )
  p.add_argument("--model", default="gemma3_instruct_4b_text")
  p.add_argument("--sequence_length", type=int, default=1024)
  p.add_argument("--test_ds_sequence_length", type=int, default=1024)
  p.add_argument("--epochs", type=int, default=1)
  p.add_argument(
      "--batch_size",
      type=int,
      default=4,
      help="Physical microbatch size inside DPTrainer.",
  )
  p.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=128,
      help=(
          "Effective batch multiplier: the expected Poisson batch size is "
          "batch_size * gradient_accumulation_steps. DPTrainer aggregates "
          "microbatches within each update; Optax does not accumulate steps."
      ),
  )
  p.add_argument("--train_preprocessing_batch_size", type=int, default=128)
  p.add_argument("--test_batch_size", type=int, default=4)
  p.add_argument("--lora_rank", type=int, default=64)
  p.add_argument("--learning_rate", type=float, default=3e-3)
  p.add_argument(
      "--seed",
      type=int,
      default=None,
      help=(
          "Optional reproducibility seed. Leave unset for private randomness; "
          "a public or predictable seed invalidates the DP guarantee."
      ),
  )
  # Probe
  p.add_argument(
      "--probe_samples",
      type=int,
      default=50000,
      help="Expected Poisson probe size (capped at the training-set size).",
  )
  p.add_argument("--probe_topk", type=int, default=8)
  p.add_argument("--probe_noise_multiplier", type=float, default=20.0)
  p.add_argument("--probe_microbatch_size", type=int, default=4)
  p.add_argument("--probe_preprocessing_batch_size", type=int, default=128)
  p.add_argument("--top_k_percent", type=float, default=5.0)
  # DP
  p.add_argument("--total_epsilon", type=float, default=4.0)
  p.add_argument("--delta", type=float, default=2e-5)
  p.add_argument("--clipping_norm", type=float, default=1e-3)
  # Smoke
  p.add_argument("--test_run", action="store_true")
  p.add_argument(
      "--skip_baseline",
      action="store_true",
      help="Skip the baseline run and only evaluate the probe-selected mask.",
  )
  p.add_argument(
      "--skip_dpsapf",
      action="store_true",
      help="Skip the DP-SAPF run; useful to isolate the baseline number.",
  )
  return p.parse_args()


def apply_test_run(args):
  if args.test_run:
    args.model = "gemma3_instruct_1b"
    args.probe_samples = min(args.probe_samples, 10000)
  return args


def main():
  args = apply_test_run(parse_args())
  _set_env_defaults(args.cache_root)

  # Heavy imports go here, after env vars are set.
  import tensorflow as tf

  # TensorFlow only handles input data/tokenization. Keep its allocator off
  # the accelerator used by JAX for the probe and DP training.
  tf.config.set_visible_devices([], "GPU")

  import gc
  import jax
  import keras
  import keras_hub  # pytype: disable=import-error
  import numpy as np

  from jax_privacy import accounting
  from jax_privacy import batch_selection
  from jax_privacy import saliency

  print(f"Dataset: {args.dataset}   Model: {args.model}")
  print(
      f"top_k_percent={args.top_k_percent}   total_epsilon={args.total_epsilon}"
  )

  # ---------- Data + model ----------
  dataset_cfg = DATASET_REGISTRY[args.dataset]
  fmt = _make_source_to_gemma3_format(dataset_cfg)
  train_ds = _load_dataset_split(dataset_cfg, "train").map(fmt)
  test_ds = _load_dataset_split(dataset_cfg, "test").map(fmt)

  train_size = int(train_ds.cardinality().numpy())
  print(f"train_size={train_size}")
  if train_size <= 0:
    raise ValueError("The training split must contain at least one record.")
  if args.probe_samples <= 0:
    raise ValueError("--probe_samples must be positive.")
  if args.probe_preprocessing_batch_size <= 0:
    raise ValueError("--probe_preprocessing_batch_size must be positive.")

  expected_probe_size = min(args.probe_samples, train_size)
  probe_sampling_probability = expected_probe_size / train_size
  probe_sampling = batch_selection.CyclicPoissonSampling(
      sampling_prob=probe_sampling_probability,
      iterations=1,
      partition_type=batch_selection.PartitionType.INDEPENDENT,
  )
  probe_dp_event = accounting.dpsgd_event(
      noise_multiplier=args.probe_noise_multiplier,
      iterations=1,
      sampling_prob=probe_sampling.sampling_prob,
  )
  sampling_seed, probe_noise_seed_sequence, training_seed_sequence = (
      np.random.SeedSequence(args.seed).spawn(3)
  )
  sampling_rng = np.random.default_rng(sampling_seed)
  probe_indices = next(
      probe_sampling.batch_iterator(train_size, rng=sampling_rng)
  )
  print(
      "probe Poisson sampling: "
      f"q={probe_sampling_probability:.6f}, "
      f"expected_size={expected_probe_size}"
  )

  test_ds_batched = test_ds.batch(args.test_batch_size)

  keras.distribution.set_distribution(keras.distribution.DataParallel())

  # ---------- Probe (via new library API) ----------
  print("\n--- Probe pass ---")
  # Load model once for the probe.
  gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(args.model)
  gemma_lm.preprocessor.sequence_length = args.sequence_length

  # Candidates = every attention-projection Dense/EinsumDense kernel.
  candidate_layers = _get_lora_candidate_layers(
      gemma_lm.backbone, attn_only=True
  )
  candidate_kernel_ids = {id(l.kernel) for l in candidate_layers}
  trainable_vars = list(gemma_lm.trainable_variables)

  # Boolean list, same order as `trainable_vars` -> pytree-compatible.
  candidate_mask = [id(v) in candidate_kernel_ids for v in trainable_vars]
  num_candidates = sum(candidate_mask)
  select_top_k = max(1, round(num_candidates * args.top_k_percent / 100.0))
  print(f"num_candidates={num_candidates}   select_top_k={select_top_k}")

  # Loss compatible with `jax_privacy.clipped_grad`:
  # signature `loss_fn(params, batch)` where batch is a dict with the fields
  # the Gemma3 preprocessor produces (`x`, `y`, `sw`).
  ntvars = [v.value for v in gemma_lm.non_trainable_variables]
  loss_obj = keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction="sum_over_batch_size"
  )

  def loss_fn(params, batch):
    x, y, sw = batch["x"], batch["y"], batch["sw"]
    # pytype: disable=attribute-error
    # gemma_lm is None-rebound later in this function to free the ~4GB backbone
    # before per-config fine-tunes, which triggers a false-positive here.
    y_pred, _ = gemma_lm.stateless_call(params, ntvars, x, training=False)
    # pytype: enable=attribute-error
    return loss_obj(y, y_pred.astype(jax.numpy.float32), sample_weight=sw)

  preproc = gemma_lm.preprocessor

  # Select each member of the full population independently with probability
  # q. The caller-owned event above uses the same sampling strategy's public
  # probability, tying the implemented mechanism to its accounting.
  selection_mask = np.zeros(train_size, dtype=np.bool_)
  selection_mask[probe_indices] = True
  selection_mask = tf.convert_to_tensor(selection_mask)

  def keep_probe_example(index, unused_example):
    del unused_example
    return tf.gather(selection_mask, index)

  def drop_example_index(unused_index, example):
    del unused_index
    return example

  selected_probe_ds = (
      train_ds.enumerate().filter(keep_probe_example).map(drop_example_index)
  )

  def preprocess_probe_chunk(chunk, preprocessor):
    """Preprocesses one host chunk into the batched loss-function pytree."""
    x, y, sw = preprocessor(chunk)
    y = np.asarray(y)
    if sw is None:
      sw = np.ones_like(y, dtype=np.float32)
    return jax.tree.map(np.asarray, {"x": x, "y": y, "sw": sw})

  probe_chunks = [
      preprocess_probe_chunk(chunk, preproc)
      for chunk in selected_probe_ds.batch(
          args.probe_preprocessing_batch_size, drop_remainder=False
      )
  ]
  if probe_chunks:
    probe_batch = jax.tree.map(
        lambda *chunks: np.concatenate(chunks, axis=0), *probe_chunks
    )
  else:
    # A Poisson draw may be empty. Use one record only to obtain shapes; the
    # empty batch is handled without evaluating `loss_fn` in the probe.
    shape_chunk = next(iter(train_ds.take(1).batch(1)))
    probe_batch = jax.tree.map(
        lambda x: x[:0], preprocess_probe_chunk(shape_chunk, preproc)
    )

  materialized_probe_size = jax.tree.leaves(probe_batch)[0].shape[0]
  if materialized_probe_size != len(probe_indices):
    raise RuntimeError(
        "Poisson sample materialization did not preserve the sampled size."
    )

  probe_noise_seed = int(probe_noise_seed_sequence.generate_state(1)[0])

  probe_result = saliency.topk_vote_probe(
      loss_fn=loss_fn,
      dataset=probe_batch,
      params=[v.value for v in trainable_vars],
      vote_top_k=args.probe_topk,
      select_top_k=select_top_k,
      noise_multiplier=args.probe_noise_multiplier,
      candidate_mask=candidate_mask,
      prng_key=jax.random.PRNGKey(probe_noise_seed),
      microbatch_size=args.probe_microbatch_size,
  )
  print(f"probe complete: kept {select_top_k}/{num_candidates} layers.")
  # Translate the boolean pytree back to the set of `.path` strings that
  # `_enable_lora_on_paths` expects.
  probe_selected_paths = _mask_to_paths(
      probe_result.selected_mask, trainable_vars, candidate_layers
  )
  print("Top-scoring probe-selected layers:")
  for i, (idx, s) in enumerate(probe_result.ranked_scores[:select_top_k]):
    print(f"  #{i+1:3d} score={s:.3e}  {_index_to_path(idx, candidate_layers)}")

  # ---------- Baseline mask = keras_hub's `enable_lora()` default ----------
  # `default_lora_layer_names()` returns a family-agnostic list of leaf names
  # (e.g. ["query_dense", "value_dense", "query", "value"] for Gemma3), and
  # `Backbone.enable_lora()` matches by `layer.name`. That's typically
  # query+value on every attention block — a much stronger reference point
  # than "adapt every attn projection" since it's what a user gets out of
  # the box.
  default_names = set(gemma_lm.backbone.default_lora_layer_names())
  baseline_selected_paths = {
      l.path for l in candidate_layers if l.name in default_names
  }
  baseline_label = (
      f"baseline (keras_hub default: {sorted(default_names)}, "
      f"{len(baseline_selected_paths)} layers)"
  )
  print(
      f"baseline default_lora_layer_names={sorted(default_names)} -> "
      f"{len(baseline_selected_paths)} layers"
  )

  # Keep the full population on the host. DPTrainer selects actual Poisson
  # batches by index; only those batches are transferred to the accelerator.
  train_data = _materialize_training_data(
      train_ds, preproc, train_size, args.train_preprocessing_batch_size
  )

  # Free the probe-time model + JIT cache before we spin up per-run fine-tunes.
  # (We rebind to None rather than `del`-ing because pyflakes' F821 check gets
  # confused by `del` on names captured by nested closures earlier in scope.)
  gemma_lm = None
  trainable_vars = None
  ntvars = None
  loss_fn = None
  probe_batch = None
  probe_chunks = None
  probe_indices = None
  selection_mask = None
  candidate_layers = None
  preproc = None
  gc.collect()
  jax.clear_caches()

  # ---------- Two DP-SGD runs, matched (eps, delta) ----------
  runs = []
  if not args.skip_baseline:
    runs.append((baseline_label, baseline_selected_paths))
  if not args.skip_dpsapf:
    runs.append((
        f"DP-SAPF (top-{args.top_k_percent}% probe)",
        probe_selected_paths,
    ))

  results = {}
  for (label, selected_paths), run_seed in zip(
      runs, training_seed_sequence.spawn(len(runs))
  ):
    print(f"\n--- Fine-tune: {label} ---")
    print(f"selected {len(selected_paths)} layers")
    rouge = _run_one_config(
        args,
        selected_paths,
        train_size,
        train_data,
        test_ds_batched,
        probe_dp_event,
        seed=int(run_seed.generate_state(1)[0]),
    )
    results[label] = rouge
    print(f"{label} ROUGE: {rouge}")

  # ---------- Report ----------
  print("\n=========================================================")
  print("Summary")
  print("=========================================================")
  for label, rouge in results.items():
    print(f"  {label}")
    for k, v in rouge.items():
      print(f"    {k}: {v:.4f}")
  if len(results) == 2:
    # pylint: disable-next=unbalanced-tuple-unpacking
    base, dp = list(results.values())
    print("\n  DP-SAPF - baseline delta:")
    for k in base:
      print(f"    {k}: {dp[k] - base[k]:+.4f}")


def _materialize_training_data(dataset, preprocessor, num_examples, chunk_size):
  """Tokenizes the full population into host arrays for indexed sampling."""
  import jax
  import numpy as np

  if chunk_size <= 0:
    raise ValueError("--train_preprocessing_batch_size must be positive.")
  result = None
  offset = 0
  for chunk in dataset.batch(chunk_size, drop_remainder=False):
    x, y, sw = preprocessor(chunk)
    y = np.asarray(y)
    if sw is None:
      sw = np.ones_like(y, dtype=np.float32)
    batch = jax.tree.map(np.asarray, {"x": x, "y": y, "sw": sw})
    if result is None:
      result = jax.tree.map(
          lambda a: np.empty((num_examples, *a.shape[1:]), dtype=a.dtype),
          batch,
      )
    end = offset + y.shape[0]
    if end > num_examples:
      raise ValueError("Training data exceeds its declared cardinality.")
    for target, source in zip(jax.tree.leaves(result), jax.tree.leaves(batch)):
      target[offset:end] = source
    offset = end
  if result is None or offset != num_examples:
    raise ValueError("Training data does not match its declared cardinality.")
  return result


def _make_keras_loss(model):
  """Adapts a stateless Keras model to DPTrainer's (loss, aux) contract."""
  import jax.numpy as jnp
  import keras

  # Only LoRA variables are passed as params. Frozen pretrained weights stay
  # outside gradient, optimizer, and noise state. Current DPTrainer.fit hoists
  # these closure arrays into executable arguments instead of HLO constants.
  frozen = [v.value for v in model.non_trainable_variables]
  loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  def loss_fn(params, batch, prng):
    del prng
    # Gemma's deterministic forward needs no mutable training state/dropout.
    logits, _ = model.stateless_call(params, frozen, batch["x"], training=False)
    loss = loss_obj(
        batch["y"], logits.astype(jnp.float32), sample_weight=batch["sw"]
    )
    return loss, ()

  return loss_fn


def _composed_epsilon(probe_dp_event, train_dp_event, delta):
  """Accounts for the fixed probe followed by one fine-tune."""
  import dp_accounting

  accountant = dp_accounting.rdp.RdpAccountant()
  accountant.compose(
      dp_accounting.ComposedDpEvent([probe_dp_event, train_dp_event])
  )
  return accountant.get_epsilon(delta)


def _calibrate_training_config(
    *,
    train_size,
    batch_size,
    gradient_accumulation_steps,
    epochs,
    clipping_norm,
    probe_dp_event,
    epsilon,
    delta,
):
  """Calibrates one-band (DP-SGD) training with the probe cost included."""
  import dp_accounting
  from jax_privacy import accounting
  from jax_privacy import training
  import numpy as np

  if min(train_size, batch_size, gradient_accumulation_steps, epochs) <= 0:
    raise ValueError("Training size, batch sizes, and epochs must be positive.")
  effective_batch_size = batch_size * gradient_accumulation_steps
  if effective_batch_size > train_size:
    raise ValueError("The effective batch size must not exceed train_size.")
  if not np.isfinite(epsilon) or epsilon <= 0 or not 0 < delta < 1:
    raise ValueError(
        "epsilon must be positive and finite; delta must be in (0, 1)."
    )
  if not np.isfinite(clipping_norm) or clipping_norm <= 0:
    raise ValueError("clipping_norm must be positive and finite.")
  # Count optimizer updates, not physical microbatches. 'epochs' sets expected
  # passes through the population; Poisson sampling has no exact epoch split.
  iterations = epochs * (train_size // effective_batch_size)
  sampling_probability = effective_batch_size / train_size
  probe_accountant = dp_accounting.rdp.RdpAccountant()
  probe_accountant.compose(probe_dp_event)
  if probe_accountant.get_epsilon(delta) >= epsilon:
    raise ValueError("The probe already exhausts the target privacy budget.")

  def composed_event(sigma_train):
    train_event = accounting.dpsgd_event(
        noise_multiplier=sigma_train,
        iterations=iterations,
        sampling_prob=sampling_probability,
    )
    return dp_accounting.ComposedDpEvent([probe_dp_event, train_event])

  sigma_train = dp_accounting.calibrate_dp_mechanism(
      make_fresh_accountant=dp_accounting.rdp.RdpAccountant,
      make_event_from_param=composed_event,
      target_epsilon=epsilon,
      target_delta=delta,
  )
  return training.BandMFConfig(
      strategy=np.array([1.0]),  # One band: independent-noise DP-SGD.
      iterations=iterations,
      expected_participations=iterations * sampling_probability,
      noise_multiplier=sigma_train,
      l2_clip_norm=clipping_norm,
      rescale_to_unit_norm=False,
      normalize_by=effective_batch_size,
  )


def _training_padding_multiple(
    sampling_strategy, num_examples, *, microbatch_size, sampling_seed
):
  """Finds one padded shape for nonempty batches without consuming fit RNG."""
  import numpy as np

  if microbatch_size <= 0:
    raise ValueError("microbatch_size must be positive.")
  # Match DPTrainer.fit's RNG use: first draw the loss PRNG seed, then sample.
  # fit receives the original SeedSequence and independently replays the draws.
  rng = np.random.default_rng(sampling_seed)
  rng.integers(2**63)
  batches = sampling_strategy.batch_iterator(num_examples, rng=rng)
  max_batch_size = max((len(indices) for indices in batches), default=0)
  return max(1, -(-max_batch_size // microbatch_size)) * microbatch_size


def _fit_keras_model(
    model, dataset, config, *, microbatch_size, learning_rate, seed
):
  """Runs DPTrainer and writes only the trained adapters back to Keras."""
  import jax
  from jax_privacy import training
  import numpy as np
  import optax

  sampling_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
  performance_flags = training.PerformanceFlags(
      microbatch_size=microbatch_size,
      noise_seed=int(noise_seed.generate_state(1)[0]),
  )
  plan = config.make(performance_flags)
  padding_multiple = _training_padding_multiple(
      plan.batch_selection_strategy,
      jax.tree.leaves(dataset)[0].shape[0],
      microbatch_size=microbatch_size,
      sampling_seed=sampling_seed,
  )
  # Padding to 32 allows changing Poisson sizes to compile multiple large
  # Gemma train_step executables. Replay only the host sampler to find one
  # shape covering this run, without truncating any sampled batch. The clipper
  # skips trailing all-padding microbatches. Empty draws retain their zero
  # shape and noise-only update. Do not log sizes derived from private draws.
  trainer = training.DPTrainer(
      config=config,
      performance_flags=performance_flags,
      loss_fn=_make_keras_loss(model),
      optimizer=optax.adam(learning_rate),
      compilation_strategy=training.PadToMultiple(multiple=padding_multiple),
  )
  variables = model.trainable_variables
  if not variables:
    raise ValueError(
        "The selected paths did not produce any trainable adapters."
    )
  print(
      f"DPTrainer: microbatch_size={microbatch_size}; "
      "one padded shape for nonempty batches.",
      flush=True,
  )

  def on_step(step, state, aux):
    # Bound asynchronous dispatch: finish the update before staging another
    # batch, and surface accelerator errors at the step that triggered them.
    try:
      jax.block_until_ready((state, aux))
    except jax.errors.JaxRuntimeError as error:
      raise RuntimeError(
          f"DPTrainer update {step}/{config.iterations} failed."
      ) from error
    if step == 1 or step % 10 == 0 or step == config.iterations:
      print(
          f"DPTrainer update {step}/{config.iterations} completed.", flush=True
      )

  # Report only the public update count, not per-example losses or norms.
  # JIT mode still uses DPTrainer's frozen-constant hoisting context.
  state = trainer.fit(
      dataset,
      [v.value for v in variables],
      rng_or_seed=sampling_seed,
      precompile=False,
      callback=on_step,
  )
  for variable, value in zip(variables, state.params):
    variable.assign(value)
  return state


def _run_one_config(
    args,
    selected_paths,
    train_size,
    train_data,
    test_ds_batched,
    probe_dp_event,
    *,
    seed,
):
  """Runs one DPTrainer fine-tune and Keras ROUGE evaluation."""
  import gc
  import jax
  import keras_hub  # pytype: disable=import-error

  config = _calibrate_training_config(
      train_size=train_size,
      batch_size=args.batch_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      epochs=args.epochs,
      clipping_norm=args.clipping_norm,
      probe_dp_event=probe_dp_event,
      epsilon=args.total_epsilon,
      delta=args.delta,
  )
  # Check the event emitted by the actual execution config as well.
  epsilon = _composed_epsilon(
      probe_dp_event, config.make().dp_event, args.delta
  )
  print(
      f"sigma_train={config.noise_multiplier:.4f}; "
      f"composed eps={epsilon:.4f}; updates={config.iterations}"
  )

  gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(args.model)
  gemma_lm.preprocessor.sequence_length = args.sequence_length
  _enable_lora_on_paths(gemma_lm.backbone, selected_paths, args.lora_rank)
  state = _fit_keras_model(
      gemma_lm,
      train_data,
      config,
      microbatch_size=args.batch_size,
      learning_rate=args.learning_rate,
      seed=seed,
  )
  print(f"DPTrainer completed {int(state.step)} updates.")
  del state

  # ROUGE eval
  import tqdm

  gemma_lm.preprocessor.sequence_length = args.test_ds_sequence_length
  metric_fns = {
      "rouge_1": keras_hub.metrics.RougeN(order=1),
      "rouge_2": keras_hub.metrics.RougeN(order=2),
      "rouge_l": keras_hub.metrics.RougeL(),
  }

  def common_prefix(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
      i += 1
    return i

  for batch in tqdm.tqdm(test_ds_batched):
    prompts = [p.decode("utf-8") for p in batch["prompts"].numpy()]
    outputs = gemma_lm.generate(prompts)
    outputs = [o[common_prefix(p, o) :] for p, o in zip(prompts, outputs)]
    targets = [s.decode("utf-8") for s in batch["responses"].numpy()]
    for m in metric_fns.values():
      m.update_state(targets, outputs)

  rouge = {k: float(m.result()["f1_score"]) for k, m in metric_fns.items()}

  # Clean up before the next config.
  del gemma_lm
  gc.collect()
  jax.clear_caches()
  return rouge


def _index_to_path(candidate_local_index, candidate_layers):
  """Maps a probe-local candidate index back to the layer's `.path`."""
  return candidate_layers[candidate_local_index].path


def _mask_to_paths(selected_mask, trainable_vars, candidate_layers):
  """Boolean pytree over trainable vars -> set of layer `.path` strings."""
  id_to_path = {id(l.kernel): l.path for l in candidate_layers}
  selected_paths = set()
  for var, is_selected in zip(trainable_vars, selected_mask):
    if is_selected and id(var) in id_to_path:
      selected_paths.add(id_to_path[id(var)])
  return selected_paths


if __name__ == "__main__":
  main()
