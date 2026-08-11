#!/usr/bin/env python3
# Copyright 2026 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validate that `jax_privacy.saliency.topk_vote_probe` produces useful masks.

Runs two DP-SGD fine-tunes of Gemma3 under matched (eps, delta):

  (1) baseline: LoRA on every candidate attention projection (~ the
      keras_hub `enable_lora()` default of query+value everywhere, but
      extended to all attention leaves for a fairer 'adapt everything'
      reference)
  (2) DP-SAPF: LoRA on the top-`--top_k_percent`% attention layers selected
      by the DP probe

Reports ROUGE-1/2/L for both and the delta. `dp_sgd_keras_gemma3_dpsapf.ipynb`
demonstrates the full mechanism; this script exists so you can iterate on
the probe implementation without a Jupyter round-trip.

Typical run (single-GPU, ~10-20 min per config on 1B):
  python examples/dpsapf_validate.py \\
      --dataset cnn_dailymail --test_run \\
      --top_k_percent 5 --total_epsilon 4.0

`--test_run` swaps to Gemma3 1B. Drop it for the full 4B model.
"""

import argparse
import os

# Deferred imports below: the heavy JAX/Keras/TF stack is imported inside
# functions so `os.environ['KERAS_BACKEND']` and cache-dir env vars can be
# set first (they must be in place before `import keras`).
# pylint: disable=import-outside-toplevel


DEFAULT_CACHE_ROOT = "/bigtemp/fzv6en/diffuser_cache"


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
  p.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
  p.add_argument(
      "--dataset",
      default="cnn_dailymail",
      choices=["samsum", "cnn_dailymail", "xsum_hf"],
  )
  p.add_argument("--model", default="gemma3_instruct_4b_text")
  p.add_argument("--sequence_length", type=int, default=1024)
  p.add_argument("--test_ds_sequence_length", type=int, default=1024)
  p.add_argument("--epochs", type=int, default=1)
  p.add_argument("--batch_size", type=int, default=4)
  p.add_argument("--gradient_accumulation_steps", type=int, default=128)
  p.add_argument("--test_batch_size", type=int, default=4)
  p.add_argument("--lora_rank", type=int, default=64)
  p.add_argument("--learning_rate", type=float, default=3e-3)
  p.add_argument("--seed", type=int, default=0)
  # Probe
  p.add_argument("--probe_samples", type=int, default=50000)
  p.add_argument("--probe_topk", type=int, default=8)
  p.add_argument("--probe_noise_multiplier", type=float, default=20.0)
  p.add_argument("--probe_microbatch_size", type=int, default=4)
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
    args.probe_samples = min(args.probe_samples, 1000)
  return args


def main():
  args = apply_test_run(parse_args())
  _set_env_defaults(args.cache_root)

  # Heavy imports go here, after env vars are set.
  import gc
  import jax
  import keras
  import keras_hub  # pytype: disable=import-error

  from jax_privacy import saliency

  print(f"Dataset: {args.dataset}   Model: {args.model}")
  print(
      f"top_k_percent={args.top_k_percent}   total_epsilon={args.total_epsilon}"
  )

  # ---------- Data + model ----------
  dataset_cfg = DATASET_REGISTRY[args.dataset]
  fmt = _make_source_to_gemma3_format(dataset_cfg)
  train_ds = _load_dataset_split(dataset_cfg, "train").map(fmt)
  val_ds = _load_dataset_split(dataset_cfg, "validation").map(fmt)
  test_ds = _load_dataset_split(dataset_cfg, "test").map(fmt)

  train_size = int(train_ds.cardinality().numpy())
  print(f"train_size={train_size}")

  train_ds_batched = train_ds.shuffle(2048).batch(
      args.batch_size, drop_remainder=True
  )
  val_ds_batched = val_ds.batch(args.batch_size, drop_remainder=True)
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

  def probe_batches():
    """Yields microbatches (dict) shaped for `loss_fn`."""
    src = (
        train_ds_batched.unbatch()
        .take(args.probe_samples)
        .batch(args.probe_microbatch_size, drop_remainder=True)
    )
    for chunk in src:
      x, y, sw = preproc(chunk)
      # convert to jax arrays
      x = {
          k: jax.numpy.asarray(v.numpy() if hasattr(v, "numpy") else v)
          for k, v in x.items()
      }
      y = jax.numpy.asarray(y.numpy() if hasattr(y, "numpy") else y)
      sw = (
          jax.numpy.asarray(sw.numpy() if hasattr(sw, "numpy") else sw)
          if sw is not None
          else jax.numpy.ones_like(y, dtype=jax.numpy.float32)
      )
      yield {"x": x, "y": y, "sw": sw}

  probe_result = saliency.topk_vote_probe(
      loss_fn=loss_fn,
      dataset=probe_batches(),
      params=[v.value for v in trainable_vars],
      num_samples=args.probe_samples,
      vote_top_k=args.probe_topk,
      select_top_k=select_top_k,
      noise_multiplier=args.probe_noise_multiplier,
      candidate_mask=candidate_mask,
      prng_key=jax.random.PRNGKey(args.seed),
      sampling_probability=args.probe_samples / train_size,
      microbatch_size=args.probe_microbatch_size,
  )
  print(
      f"probe: {probe_result.n_seen} samples, "
      f"kept {select_top_k}/{num_candidates} layers."
  )
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

  # Free the probe-time model + JIT cache before we spin up per-run fine-tunes.
  # (We rebind to None rather than `del`-ing because pyflakes' F821 check gets
  # confused by `del` on names captured by nested closures earlier in scope.)
  gemma_lm = None
  trainable_vars = None
  ntvars = None
  loss_fn = None
  probe_batches = None
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
  for label, selected_paths in runs:
    print(f"\n--- Fine-tune: {label} ---")
    print(f"selected {len(selected_paths)} layers")
    rouge = _run_one_config(
        args,
        selected_paths,
        train_size,
        train_ds_batched,
        val_ds_batched,
        test_ds_batched,
        probe_result,
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


def _run_one_config(
    args,
    selected_paths,
    train_size,
    train_ds_batched,
    val_ds_batched,
    test_ds_batched,
    probe_result,
):
  """Runs one full DP-SGD fine-tune + ROUGE eval for the given mask.

  Uses `keras_api.make_private` for training. (An earlier version of this
  script tried to migrate to `jax_privacy.training.DPTrainer`, but bridging
  Keras stateless_call with DP-SGD hit a "closure-captured constants" vs
  "params-shaped runtime buffers" tradeoff neither of which we could fit —
  see the discussion with @ryan112358 on the PR. Reverting until we agree
  on the intended Keras + DPTrainer pattern.)
  """
  import gc
  import jax
  import keras
  import keras_hub  # pytype: disable=import-error

  import dp_accounting
  from jax_privacy import accounting
  from jax_privacy import keras_api

  gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(args.model)
  gemma_lm.preprocessor.sequence_length = args.sequence_length

  _enable_lora_on_paths(gemma_lm.backbone, selected_paths, args.lora_rank)

  steps_per_epoch = train_size // args.batch_size
  total_train_steps = args.epochs * steps_per_epoch
  effective_batch_size = args.batch_size * args.gradient_accumulation_steps

  # Calibrate sigma_train so composed (probe + training) matches target eps.
  def composed_eps(sigma_train):
    train_event = accounting.dpsgd_event(
        noise_multiplier=sigma_train,
        iterations=total_train_steps,
        sampling_prob=effective_batch_size / train_size,
    )
    total = dp_accounting.ComposedDpEvent([probe_result.dp_event, train_event])
    acc = dp_accounting.rdp.RdpAccountant()
    acc.compose(total)
    return acc.get_epsilon(args.delta)

  # Binary search for sigma_train.
  lo, hi = 0.1, 10.0
  while composed_eps(hi) > args.total_epsilon:
    lo, hi = hi, hi * 2
  while composed_eps(lo) <= args.total_epsilon and lo > 1e-3:
    hi, lo = lo, lo * 0.5
  for _ in range(64):
    mid = 0.5 * (lo + hi)
    if composed_eps(mid) > args.total_epsilon:
      lo = mid
    else:
      hi = mid
    if hi - lo < 1e-3:
      break
  sigma_train = hi
  print(
      f"total_epsilon={args.total_epsilon} -> sigma_train={sigma_train:.4f}; "
      f"composed eps ~= {composed_eps(sigma_train):.4f}"
  )

  dp_cfg = keras_api.DPKerasConfig(
      epsilon=args.total_epsilon,
      delta=args.delta,
      noise_multiplier=sigma_train,
      clipping_norm=args.clipping_norm,
      batch_size=args.batch_size,
      train_steps=total_train_steps,
      train_size=train_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      seed=args.seed,
  )
  gemma_lm = keras_api.make_private(gemma_lm, dp_cfg)

  optimizer = keras.optimizers.Adam(
      learning_rate=args.learning_rate,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
  )
  optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
  gemma_lm.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=optimizer,
      weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )
  gemma_lm.fit(
      x=train_ds_batched, epochs=args.epochs, validation_data=val_ds_batched
  )

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
  del gemma_lm, optimizer
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
