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

"""Uses Monte Carlo accounting to calibrate noise multiplier for DP-SGD.

We use a small training setup for the purpose of making this example easy to
run. With larger setups, sample generation would be slower and we would need
more samples. In such cases, we recommend parallelizing sample generation. It
may also be good to discretize the samples by rounding up to the nearest
multiplier of some float and store the discretized samples as a histogram, since
this reduces the memory overhead.
"""

import math

from absl import app
import dp_accounting
from jax_privacy import batch_selection
from jax_privacy.experimental import accounting
from jax_privacy.experimental.monte_carlo import delta_calculation
from jax_privacy.experimental.monte_carlo import sample_generation
import numpy as np


ITERATIONS = 100
EPOCH_LENGTH = 10
EPSILON = 1.0
DELTA = 1e-2
BASE_DELTA = DELTA / 2  # Make this closer to DELTA for more accuracy, further
# from DELTA for more speed.

# The non-zero entries of the first column of the banded Toeplitz matrix C
# used in DP-MF. If this vector is longer than EPOCH_LENGTH, the calculation of
# nm_upper_bound needs to be updated. Use [1.0] for DP-SGD.
C_COL = np.array([1.0, 1 / 2, 1 / 4, 1 / 8])
C_COL = C_COL / np.linalg.norm(C_COL)


def main(_) -> None:
  minimum_samples = delta_calculation.minimum_samples_to_calibrate(
      BASE_DELTA, DELTA
  )
  print(f'Minimum samples to calibrate: {minimum_samples}')

  # We figure out a lower and upper bound on the noise multiplier necessary to
  # achieve (EPSILON, BASE_DELTA)-DP, and then define a sweep between these
  # bounds. Note that this is calibrating to BASE_DELTA, not DELTA.

  # For the lower bound, we use the noise multiplier required for DP-SGD with
  # Poisson sampling. This is a reasonable lower bound for DP-MF since DP-MF
  # generally requires more noise than DP-SGD.
  nm_lower_bound = dp_accounting.calibrate_dp_mechanism(
      dp_accounting.pld.PLDAccountant,
      lambda nm: accounting.dpsgd_event(
          noise_multiplier=nm,
          iterations=ITERATIONS,
          sampling_prob=1 / EPOCH_LENGTH,
      ),
      EPSILON,
      BASE_DELTA,
  )

  # For the upper bound, we calculate the needed noise multiplier if we assume
  # no amplification, in which case DP-SGD is just a Gaussian mechanism with
  # sensitivity math.ceil(ITERATIONS / EPOCH_LENGTH) ** 0.5.
  nm_upper_bound = (
      dp_accounting.calibrate_dp_mechanism(
          dp_accounting.pld.PLDAccountant,
          dp_accounting.GaussianDpEvent,
          EPSILON,
          BASE_DELTA,
      )
      * math.ceil(ITERATIONS / EPOCH_LENGTH) ** 0.5
  )

  print(f'Sweeping noise multiplier from {nm_lower_bound} to {nm_upper_bound}')

  sweep_size = math.ceil(np.log(nm_upper_bound / nm_lower_bound) / np.log(1.1))
  # delta_calculation's calibration function assumes that the parameters are
  # ordered from highest to lowest privacy, so we create a sweep in that order.
  # We know nm_upper_bound is a valid noise multiplier, so we don't need to
  # include it in the sweep.
  nm_sweep = [nm_upper_bound / 1.1**i for i in range(1, sweep_size + 1)]

  strategy = batch_selection.BallsInBinsSampling(
      cycle_length=EPOCH_LENGTH, iterations=ITERATIONS
  )
  positive_samples = []
  negative_samples = []
  # This double for loop is massively parallelizable! In this small example, it
  # is not necessary, but for larger sample sizes it is recommended to
  # parallelize this, across multiple cores or machines, in whatever manner
  # is befitting for your use case.
  for nm in nm_sweep:
    per_nm_positive_samples = []
    per_nm_negative_samples = []
    for _ in range(minimum_samples):
      positive_sample = sample_generation.get_privacy_loss_sample(
          strategy=strategy,
          noise_multiplier=nm,
          c_col=C_COL,
          positive_sample=True,
      )
      negative_sample = sample_generation.get_privacy_loss_sample(
          strategy=strategy,
          noise_multiplier=nm,
          c_col=C_COL,
          positive_sample=False,
      )
      per_nm_positive_samples.append(positive_sample)
      per_nm_negative_samples.append(negative_sample)
    positive_samples.append(per_nm_positive_samples)
    negative_samples.append(per_nm_negative_samples)

  passes_verification, best_nm_index = (
      delta_calculation.perform_calibration_from_samples(
          EPSILON,
          DELTA,
          positive_samples=positive_samples,
          negative_samples=negative_samples,
      )
  )
  if passes_verification:
    print(f'Best noise multiplier: {nm_sweep[best_nm_index]}')
  else:
    # If all the verifications fail, we fall back to nm_upper_bound, which we
    # know achieves (EPSILON, BASE_DELTA)-DP.
    print(f'Best noise multiplier: {nm_upper_bound}')


if __name__ == '__main__':
  app.run(main)
