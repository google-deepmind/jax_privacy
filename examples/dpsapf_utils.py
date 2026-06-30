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

"""Selection-only helpers for the auto-LoRA + DP-SGD notebook.

Scope: the DP probe, layer-selection thresholding, and the probe-side DP
accounting. Data loading, model setup, training, and ROUGE eval stay in
the notebook so the moving parts are visible.

Public API:
  DATASET_REGISTRY                          # cnn_dailymail, xsum_hf
  make_source_to_gemma3_format(cfg)         # tf.data.map prompt/response fn
  load_dataset_split(cfg, split_spec)       # TFDS vs HuggingFace dispatcher

  get_lora_candidate_layers(backbone, ...)  # enumerate LoRA-eligible layers
  run_probe(gemma_lm, train_ds, top_k_percent, **kw) -> ProbeResult
  enable_lora_on_paths(backbone, paths, rank)

  compose_probe_and_training(...)           # RDP composition (probe + DP-SGD)
  calibrate_train_sigma(target_eps, **kw)   # binary search for sigma_train
"""

import dataclasses
import functools
import gc

import dp_accounting
import jax
import jax.numpy as jnp
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from jax_privacy.accounting import accountants

# HuggingFace `datasets` is only needed for the xsum_hf entry of
# DATASET_REGISTRY; keep it optional so TFDS-only users don't need it.
try:
  from datasets import load_dataset  # pytype: disable=import-error
except ImportError:
  load_dataset = None


# ---------------------------------------------------------------------------
# Dataset registry + minimal data helpers (needed to materialise the probe input
# in the right prompt/response format).
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
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


def make_source_to_gemma3_format(cfg):
  """Returns a tf.data.map fn that emits {prompts, responses} string dicts."""
  in_field, out_field = cfg["input_field"], cfg["output_field"]
  prefix, suffix = cfg["prompt_prefix"], cfg["prompt_suffix"]

  def fn(d):
    return {
        "prompts": tf.strings.join([prefix, d[in_field], suffix]),
        "responses": d[out_field],
    }

  return fn


def load_dataset_split(cfg, split_spec):
  """Loads one split, dispatching by cfg['loader']. Returns a
  tf.data.Dataset with known cardinality (so train_size is meaningful)."""
  if cfg.get("loader") == "hf":
    if load_dataset is None:
      raise ImportError(
          "The HuggingFace `datasets` package is required for the xsum_hf "
          "dataset. Install it with `pip install datasets`."
      )
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


# ---------------------------------------------------------------------------
# Layer enumeration + selective LoRA enabling.
# ---------------------------------------------------------------------------


def get_lora_candidate_layers(backbone, attn_only=False):
  """Dense / EinsumDense sublayers of `backbone` eligible for a LoRA adapter."""
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


def enable_lora_on_paths(backbone, paths, rank):
  """Enable LoRA on layers whose `.path` is in `paths`; freeze the rest."""
  p2l = {
      l.path: l for l in get_lora_candidate_layers(backbone, attn_only=False)
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
  return ids


# ---------------------------------------------------------------------------
# DP probe: per-sample top-k voting over LoRA candidates.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProbeResult:
  selected_paths: set  # paths kept after `top_k_percent` threshold
  ranked: list  # [(path, score), ...] descending
  n_seen: int  # number of probe samples actually processed
  topk: int  # k in the per-sample top-k vote
  noise_multiplier: float  # in units of sqrt(topk) sensitivity
  used_noise: bool


def _probe_topk_vote(
    gemma_lm,
    train_ds,
    *,
    num_probe_samples,
    topk,
    noise_multiplier,
    microbatch_size,
    use_noise,
    seed,
    attn_only,
):
  """Internal: returns (layer_scores: dict[path->score], n_seen)."""
  trainable_vars = list(gemma_lm.trainable_variables)
  ntvars = [v.value for v in gemma_lm.non_trainable_variables]
  tvars = [v.value for v in trainable_vars]

  candidates = get_lora_candidate_layers(gemma_lm.backbone, attn_only=attn_only)
  kid_to_path = {id(l.kernel): l.path for l in candidates}
  cand_indices = tuple(
      i for i, v in enumerate(trainable_vars) if id(v) in kid_to_path
  )
  cand_paths = [kid_to_path[id(trainable_vars[i])] for i in cand_indices]
  num_cand = len(cand_indices)
  if num_cand == 0:
    raise ValueError("No LoRA-candidate layers found.")
  if topk > num_cand:
    raise ValueError(f"topk={topk} > num_candidates={num_cand}.")

  loss_obj = keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction="sum_over_batch_size"
  )

  def _loss(tvars, x_b, y_b, sw_b):
    y_pred, _ = gemma_lm.stateless_call(tvars, ntvars, x_b, training=False)
    return loss_obj(y_b, y_pred.astype(jnp.float32), sample_weight=sw_b)

  per_grad = jax.grad(_loss, argnums=0)

  def _norms_one(tvars, x_one, y_one, sw_one):
    # vmap strips the batch axis; functional Gemma needs (1, seq, ...) shapes.
    x_b = jax.tree.map(lambda v: v[None, ...], x_one)
    grads = per_grad(tvars, x_b, y_one[None, ...], sw_one[None, ...])
    return jnp.stack(
        [jnp.linalg.norm(grads[i].astype(jnp.float32)) for i in cand_indices]
    )

  @functools.partial(jax.jit, donate_argnums=(0,))
  def _step(votes, tvars, x, y, sw):
    in_axes_x = jax.tree.map(lambda _: 0, x)
    norms = jax.vmap(_norms_one, in_axes=(None, in_axes_x, 0, 0))(
        tvars, x, y, sw
    )
    _, idx = jax.lax.top_k(norms, k=topk)
    return votes + jax.nn.one_hot(idx, num_cand, dtype=jnp.float32).sum((0, 1))

  def _to_jnp(v):
    return jnp.asarray(v.numpy() if hasattr(v, "numpy") else v)

  preproc = gemma_lm.preprocessor
  ds = (
      train_ds.unbatch()
      .take(num_probe_samples)
      .batch(microbatch_size, drop_remainder=True)
  )

  l2_sens = float(topk) ** 0.5
  print(
      f"  probe: top-{topk} vote over {num_probe_samples} samples, "
      f"{num_cand} candidates, L2 sensitivity={l2_sens:.3f}"
  )

  votes = jnp.zeros((num_cand,), dtype=jnp.float32)
  n_seen = 0
  for chunk in ds:
    x, y, sw = preproc(chunk)
    x = {k: _to_jnp(v) for k, v in x.items()}
    y = _to_jnp(y)
    sw = _to_jnp(sw) if sw is not None else jnp.ones_like(y, dtype=jnp.float32)
    votes = _step(votes, tvars, x, y, sw)
    n_seen += microbatch_size
    if (n_seen // microbatch_size) % 32 == 0:
      print(f"  probe: {n_seen}/{num_probe_samples}")

  if use_noise:
    noise = (noise_multiplier * l2_sens) * jax.random.normal(
        jax.random.PRNGKey(seed), (num_cand,), dtype=jnp.float32
    )
    votes = votes + noise

  return {cand_paths[i]: float(votes[i]) for i in range(num_cand)}, n_seen


def run_probe(
    gemma_lm,
    train_ds,
    *,
    top_k_percent,
    num_probe_samples=1024,
    topk=8,
    noise_multiplier=6.0,
    microbatch_size=1,
    use_noise=True,
    seed=0,
    attn_only=True,
    clear_caches_after=True,
):
  """DP probe + selection in one call.

  Per training sample, vote +1 on the topk LoRA candidates with the largest
  per-sample gradient L2 norm. Add Gaussian noise to the vote histogram
  (std = noise_multiplier * sqrt(topk)). Return the top `top_k_percent`%
  layers by noisy score, plus the full ranked list and DP metadata needed
  for downstream composition.

  Frees the probe-time JIT cache before returning unless told otherwise.
  """
  scores, n_seen = _probe_topk_vote(
      gemma_lm,
      train_ds,
      num_probe_samples=num_probe_samples,
      topk=topk,
      noise_multiplier=noise_multiplier,
      microbatch_size=microbatch_size,
      use_noise=use_noise,
      seed=seed,
      attn_only=attn_only,
  )
  ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
  num_keep = max(1, round(len(ranked) * top_k_percent / 100.0))
  selected = {p for p, _ in ranked[:num_keep]}

  print(
      f"\n{n_seen} samples; keeping top {num_keep}/{len(ranked)} "
      f"({top_k_percent}%):"
  )
  for i, (p, s) in enumerate(ranked[:num_keep]):
    print(f"  #{i+1:3d} score={s:.3e}  {p}")

  if clear_caches_after:
    gc.collect()
    jax.clear_caches()

  return ProbeResult(
      selected_paths=selected,
      ranked=ranked,
      n_seen=n_seen,
      topk=topk,
      noise_multiplier=noise_multiplier,
      used_noise=use_noise,
  )


# ---------------------------------------------------------------------------
# DP accounting: probe (1 Poisson-Gaussian) + DP-SGD (T Poisson-Gaussian).
# ---------------------------------------------------------------------------


def compose_probe_and_training(
    probe_sigma,
    num_probe_samples,
    train_sigma,
    effective_batch_size,
    train_steps,
    train_size,
    delta,
):
  """Renyi-DP composed epsilon at (delta) for probe + T DP-SGD steps."""
  acc = accountants.RdpAccountantConfig().create_accountant()
  acc.compose(
      dp_accounting.PoissonSampledDpEvent(
          sampling_probability=num_probe_samples / train_size,
          event=dp_accounting.GaussianDpEvent(probe_sigma),
      ),
      1,
  )
  acc.compose(
      dp_accounting.PoissonSampledDpEvent(
          sampling_probability=effective_batch_size / train_size,
          event=dp_accounting.GaussianDpEvent(train_sigma),
      ),
      train_steps,
  )
  return acc.get_epsilon(delta)


def calibrate_train_sigma(target_eps, **kw):
  """Binary-search sigma_train so compose(probe, T x train) == target_eps."""

  def at(s):
    return compose_probe_and_training(train_sigma=s, **kw)

  if at(1e10) >= target_eps:
    raise ValueError(
        "Probe alone exceeds target_eps; raise noise_multiplier or "
        "reduce num_probe_samples."
    )
  lo, hi = 0.1, 10.0
  while at(hi) > target_eps:
    lo, hi = hi, hi * 2
  while at(lo) <= target_eps and lo > 1e-3:
    hi, lo = lo, lo * 0.5
  for _ in range(80):
    mid = 0.5 * (lo + hi)
    if at(mid) > target_eps:
      lo = mid
    else:
      hi = mid
    if hi - lo < 1e-3:
      break
  return hi
