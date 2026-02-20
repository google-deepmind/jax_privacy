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

"""Methods for sampling from various PLDs arising from DP-BandMF.

For convenience, all methods assume C in DP-MF is a banded Toeplitz matrix,
which can be specified by the non-zero entries of its first column. The main
benefit of this assumption is that we can compute Cx efficiently via
convolution. The analysis behind the methods here can be extended to other
families of C, but that is beyond the current scope of this library.

Docstrings will refer to an example being included or excluded from the dataset,
i.e. the add-remove adjacency, but this is equivalent to the zero-out adjacency
as of now for all methods in this library.
"""

from jax_privacy import batch_selection
import numpy as np
import scipy as sp

Seed = int | np.random.Generator | None


def _validate_c_col(c_col: np.ndarray) -> None:
  """Validates c_col for DP-BandMF."""
  if c_col.ndim != 1:
    raise ValueError('c_col must be a 1D array.')
  if np.any(c_col < 0):
    raise ValueError('c_col must be non-negative.')


def _banded_c_times_x(c_col: np.ndarray, x: np.ndarray) -> np.ndarray:
  """Multiplies the banded Toeplitz matrix whose first column's non-zeros are c_col by x."""
  assert c_col.size <= x.size
  if c_col.size == 0:
    return np.zeros_like(x, dtype=x.dtype)
  if c_col.size == 1:
    return c_col[0] * x
  top_prod = sp.signal.convolve(c_col, x[: -c_col.size + 1])
  bot_block = sp.linalg.toeplitz(c_col[:-1], np.zeros(c_col.size - 1))
  bot_prod = np.matmul(bot_block, x[-c_col.size + 1 :])
  top_prod[-c_col.size + 1 :] += bot_prod
  return top_prod


def _generate_zero_mean_sample(
    iterations: int,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
  """Generates a sample from a zero-mean Gaussian."""
  return rng.normal(loc=0.0, scale=noise_multiplier, size=iterations)


def _generate_balls_in_bins_sample(
    iterations: int,
    cycle_length: int,
    noise_multiplier: float,
    c_col: np.ndarray,
    seed: Seed = None,
    positive_sample: bool = True,
) -> np.ndarray:
  """Generates a sample from the dominating pair for DP-BandMF using balls-in-bins sampling.

  See https://arxiv.org/abs/2410.06266 for details.

  Args:
    iterations: The number of iterations of DP-MF.
    cycle_length: The length of each cycle of balls-in-bins sampling.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of c_col.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    seed: The rng or seed to use for sampling.
    positive_sample: If True, we sample from the distribution in the dominating
      pair corresponding to the case where the sensitive example is included.
      Otherwise, we sample from the other case in the dominating pair, where the
      sensitive example is not included.

  Returns:
    A sample from the dominating PLD for DP-BandMF using balls-in-bins sampling.
  """
  if iterations <= 0:
    raise ValueError('iterations must be positive.')
  if noise_multiplier <= 0:
    raise ValueError('noise_multiplier must be positive.')
  if cycle_length <= 0:
    raise ValueError('cycle_length must be positive.')
  _validate_c_col(c_col)
  if c_col.size > iterations:
    c_col = c_col[:iterations]
  rng = np.random.default_rng(seed)
  if positive_sample:
    # Add Cx to the Gaussian noise, where x is a vector which is 1 in every
    # b-th coordinate starting at a random position in {0, 1, ..., b-1} and 0
    # otherwise.
    starting_index = rng.integers(0, cycle_length)
    x = np.arange(iterations) % cycle_length == starting_index
    # TODO - Precompute this once for efficiency.
    mode = _banded_c_times_x(c_col, x)
  else:
    mode = np.zeros(iterations)
  return rng.normal(loc=mode, scale=noise_multiplier)


def _generate_b_min_sep_sample(
    strategy: batch_selection.BMinSepSampling,
    noise_multiplier: float,
    c_col: np.ndarray,
    seed: Seed = None,
    positive_sample: bool = True,
) -> np.ndarray:
  """Generates a sample from the dominating pair for DP-BandMF using b-min-sep sampling.

  See https://arxiv.org/abs/2602.09338 for details.

  Args:
    strategy: The b-min-sep sampling strategy to use.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of c_col.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. It is assumed that the length of c_col is the same as the minimum
      separation parameter in the sampling scheme.
    seed: The rng or seed to use for sampling.
    positive_sample: If True, we sample from the distribution in the dominating
      pair corresponding to the case where the sensitive example is included.
      Otherwise, we sample from the other case in the dominating pair, where the
      sensitive example is not included.

  Returns:
    A sample from the dominating PLD for DP-BandMF using b-min-sep sampling.
  """
  # Aliases for readability of math.
  p = strategy.sampling_prob
  b = strategy.min_sep
  if strategy.iterations <= 0:
    raise ValueError('iterations must be positive.')
  if noise_multiplier <= 0:
    raise ValueError('noise_multiplier must be positive.')
  if not (0 < strategy.sampling_prob < 1):
    raise ValueError('sampling_prob must be in (0, 1).')
  _validate_c_col(c_col)
  if c_col.size > strategy.min_sep:
    raise ValueError('c_col must have length less than or equal to min_sep.')
  if c_col.size > strategy.iterations:
    c_col = c_col[: strategy.iterations]
  rng = np.random.default_rng(seed)
  if positive_sample:
    pre_filter_x = rng.binomial(1, p, size=strategy.iterations)
    x = np.zeros(strategy.iterations, dtype=np.float32)
    if strategy.warm_start:
      # We use a 'warm-start' such that each example is only available in the
      # first iteration w.p. 1 / (1 + (b - 1) * p), and otherwise is first
      # available in a uniformly random iteration from the next b - 1
      # iterations.
      i = rng.choice(
          range(b),
          p=[1 / (1 + (b - 1) * p)] + [p / (1 + (b - 1) * p)] * (b - 1),
      )
    else:
      i = 0
    while i < strategy.iterations:
      if pre_filter_x[i] == 1:
        x[i] = 1.0
        i += b
      else:
        i += 1
    mode = _banded_c_times_x(c_col, x)
  else:
    mode = np.zeros(strategy.iterations)
  return rng.normal(loc=mode, scale=noise_multiplier)


def generate_sample(
    strategy: batch_selection.BatchSelectionStrategy,
    noise_multiplier: float,
    c_col: np.ndarray,
    seed: Seed = None,
    positive_sample: bool = True,
) -> np.ndarray:
  """Generates a sample from the dominating pair DP-BandMF using the given batch selection strategy.

  TODO: Refactor to allow multiple samples in a single call.
  TODO: Explore benefits of Jax implementation.

  Args:
    strategy: The batch selection strategy to use.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of c_col.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    seed: The rng or seed to use for sampling.
    positive_sample: If True, we sample from the distribution in the dominating
      pair corresponding to the case where the sensitive example is included.
      Otherwise, we sample from the other case in the dominating pair, where the
      sensitive example is not included.

  Returns:
    A sample from the dominating PLD for DP-BandMF using balls-in-bins sampling.
  """
  if isinstance(strategy, batch_selection.BallsInBinsSampling):
    return _generate_balls_in_bins_sample(
        strategy.iterations,
        strategy.cycle_length,
        noise_multiplier,
        c_col,
        seed,
        positive_sample,
    )
  elif isinstance(strategy, batch_selection.BMinSepSampling):
    if strategy.truncated_batch_size:
      raise ValueError(
          'Truncated batch size is not supported for sample generation.'
      )
    return _generate_b_min_sep_sample(
        strategy,
        noise_multiplier,
        c_col,
        seed,
        positive_sample,
    )
  else:
    raise ValueError(f'Unsupported batch selection strategy: {type(strategy)}')


def _compute_balls_in_bins_privacy_loss(
    epoch_length: int,
    sample: np.ndarray,
    noise_multiplier: float,
    c_col: np.ndarray,
) -> float:
  """Computes the privacy loss for a sample from balls-in-bins sampling.

  Args:
    epoch_length: The length of each epoch (number of bins) for balls-in-bins
      sampling.
    sample: The sample, generated by _generate_balls_in_bins_sample.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of c_col.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. Assumed to match the epoch length of balls-in-bins.

  Returns:
    The privacy loss of the sample, assuming we sample from the distribution in
    the dominating pair where the sensitive example is included.
  """
  if epoch_length <= 0:
    raise ValueError('epoch_length must be positive.')
  if noise_multiplier <= 0:
    raise ValueError('noise_multiplier must be positive.')
  if sample.ndim != 1:
    raise ValueError('sample must be a 1D array.')
  _validate_c_col(c_col)
  if c_col.size > sample.size:
    c_col = c_col[: sample.size]
  # For each possible batch that the example could have been assigned to, we get
  # an x vector which is 1 in the every b-th coordinate starting from a
  # different position in {0, 1, ..., b-1} and 0 otherwise. We construct a
  # matrix whose columns are Cx for each of these x vectors.
  x = np.arange(sample.size) % epoch_length == 0
  first_mode = _banded_c_times_x(c_col, x)
  elementary_vector = np.zeros(epoch_length, dtype=np.float32)
  elementary_vector[0] = c_col[0]
  modes_matrix = sp.linalg.toeplitz(elementary_vector, first_mode)
  # For each Cx values, we compute the likelihood ratio of the output y, which
  # is exp((2 <y, Cx> - ||Cx||^2) / 2 * noise_multiplier ** 2), conditional on
  # the associated x vector being the one actually sampled.
  dot_products = np.dot(modes_matrix, sample)
  squared_mode_norms = (modes_matrix**2).sum(axis=1)
  per_mode_privacy_loss = (2 * dot_products - squared_mode_norms) / (
      2 * noise_multiplier**2
  )
  # The final privacy loss is the log-mean-exp of the per-mode privacy losses.
  return sp.special.logsumexp(per_mode_privacy_loss) - np.log(epoch_length)


def compute_privacy_loss(
    strategy: batch_selection.BatchSelectionStrategy,
    sample: np.ndarray,
    noise_multiplier: float,
    c_col: np.ndarray,
) -> float:
  """Computes the privacy loss on a sample from the dominating pair.

  This method reports the privacy loss assuming we sample from the distribution
  in the dominating pair where the sensitive example is included. To get the
  privacy loss for when the other example is excluded, we simply negate the
  privacy loss for the positive sample.

  Args:
    strategy: The batch selection strategy used to generate the sample.
    sample: The sample, generated by _generate_balls_in_bins_sample.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of c_col.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D. Assumed to match the epoch length of balls-in-bins.

  Returns:
    The privacy loss of the sample, assuming we sample from the distribution in
    the dominating pair where the sensitive example is included.
  """
  if isinstance(strategy, batch_selection.BallsInBinsSampling):
    if sample.size != strategy.iterations:
      raise ValueError(
          'Sample size must match the number of iterations of the strategy.'
      )
    return _compute_balls_in_bins_privacy_loss(
        strategy.cycle_length,
        sample,
        noise_multiplier,
        c_col,
    )
  else:
    raise ValueError(f'Unsupported batch selection strategy: {type(strategy)}')


def get_privacy_loss_sample(
    strategy: batch_selection.BatchSelectionStrategy,
    noise_multiplier: float,
    c_col: np.ndarray,
    seed: Seed = None,
    positive_sample: bool = True,
    also_return_sample: bool = False,
) -> float | tuple[float, np.ndarray]:
  """Returns a sample from DP-BandMF's dominating privacy loss distribution.

  Args:
    strategy: The batch selection strategy to use.
    noise_multiplier: The noise multiplier of DP-MF. This is multiplied by the
      clip norm, not accounting for the norm of c_col.
    c_col: The non-zero entries in the first column of C. Should be non-negative
      and 1D.
    seed: The seed to use for sampling.
    positive_sample: True if we sample from the distribution in the dominating
      pair corresponding to the case where the sensitive example is included.
      Otherwise, we sample from the other case in the dominating pair, where the
      sensitive example is not included.
    also_return_sample: If True, we also return the sample.
  """
  sample = generate_sample(
      strategy,
      noise_multiplier,
      c_col,
      seed,
      positive_sample,
  )
  privacy_loss = compute_privacy_loss(
      strategy,
      sample,
      noise_multiplier,
      c_col,
  )
  if not positive_sample:
    privacy_loss = -privacy_loss
  if also_return_sample:
    return privacy_loss, sample
  else:
    return privacy_loss
