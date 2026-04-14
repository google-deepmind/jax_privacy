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

"""Numerical optimization of DP-MF strategies and evaluation of expected error.

This is a simple binary that invokes the various optimization routines
for computing DP-MF strategies that we support in JAX privacy under the
matrix_factorization directory.  This binary also evaluates the expected
errors of the optimized strategies.

NOTE: The expected errors reported here correspond to *Unamplified DP-MF*
and do not take into account the noise multiplier.  The relative difference
between different strategies holds for all privacy budgets / noise multipliers.

Better expected errors can be obtained by using Amplified DP-MF, such as
DP-BandMF (via banded.py or toeplitz.py).
"""

import time

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from jax_privacy.matrix_factorization import banded
from jax_privacy.matrix_factorization import buffered_toeplitz
from jax_privacy.matrix_factorization import dense
from jax_privacy.matrix_factorization import optimization
from jax_privacy.matrix_factorization import streaming_matrix
from jax_privacy.matrix_factorization import toeplitz

# pylint: disable=invalid-name
jax.config.update('jax_enable_x64', True)

_STRATEGY = flags.DEFINE_enum(
    'strategy',
    'dense',
    [
        'dense',
        'banded',
        'banded-toeplitz',
        'banded-inverse-toeplitz',
        'blt',
        'banded-sqrt',
        'banded-inverse-sqrt',
        'normalized-banded-toeplitz',
    ],
    'Strategy class to optimize over.  See code for details.',
)
_ITERATIONS = flags.DEFINE_integer(
    'iterations', 128, 'Number of training iterations'
)
_PARTICIPATIONS = flags.DEFINE_integer(
    'participations',
    1,
    'The number of participations per user or example.  If using example-level'
    'DP this corresponds to the number of training epochs.  If using user-level'
    'DP this corresponds to the number of iterations a user participates, which'
    'can be much larger than the number of epochs when users contribute only a'
    'portion of their data each time they participate.',
)
_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    'mean',
    ['mean', 'max'],
    'Objective to optimize.',
)


def optimize_strategy(
    *,
    strategy: str,
    n: int,
    participations: int = 1,
    objective: str = 'mean',
):
  """Compute matrix factorization+error metrics for the given configuration."""
  sep = n // participations
  t0 = time.time()
  loss = None

  if objective not in ['mean', 'max']:
    raise ValueError(f'Unknown objective {objective}')

  reduction_fn = jnp.mean if objective == 'mean' else jnp.max
  match strategy:
    case 'banded-toeplitz':
      # https://arxiv.org/abs/2405.15913
      strategy_coef = toeplitz.optimize_banded_toeplitz(
          n, bands=sep, max_optimizer_steps=1000, reduction_fn=reduction_fn
      )
      sensitivity_squared = toeplitz.minsep_sensitivity_squared(
          strategy_coef, min_sep=sep, max_participations=participations
      )
      pqe = toeplitz.per_query_error(strategy_coef=strategy_coef, n=n)
      loss = reduction_fn(pqe) * sensitivity_squared

    case 'banded-inverse-toeplitz':
      # https://arxiv.org/pdf/2505.12128
      weight_decay = 1.0
      momentum = 0.0
      workload_coef = toeplitz.multiply(
          weight_decay ** jnp.arange(n),
          momentum ** jnp.arange(n),
          n=n,
          skip_checks=True,
      )
      noising_coef = toeplitz.optimize_banded_inverse_toeplitz(
          n=n,
          min_sep=sep,
          num_bands=sep,
          weight_decay=weight_decay,
          momentum=momentum,
          max_optimizer_steps=1000,
          reduction_fn=reduction_fn,
      )
      sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
          n=n,
          noising_coef=noising_coef,
          min_sep=sep,
          max_participations=participations,
      ) 
      pqe = toeplitz.per_query_error(
          noising_coef=noising_coef, n=n, workload_coef=workload_coef
      )
      loss = reduction_fn(pqe) * sensitivity_squared

    case 'normalized-banded-toeplitz':
      # https://arxiv.org/abs/2405.15913
      def loss_fn(coef):  # pylint: disable=function-redefined
        C_inv = toeplitz.inverse_as_streaming_matrix(
            coef, column_normalize_for_n=n
        )
        A = streaming_matrix.prefix_sum()
        B = A @ C_inv
        return reduction_fn(B.row_norms_squared(n))

      initial_strategy_coef = toeplitz.optimal_max_error_strategy_coefs(sep)
      strategy_coef = optimization.optimize(
          loss_fn, initial_strategy_coef, max_optimizer_steps=1000
      )

      # this is just equal to # participations
      sensitivity_squared = toeplitz.minsep_sensitivity_squared(
          # Strategy is normalized in toeplitz.inverse_as_streaming_matrix.
          strategy_coef=strategy_coef / jnp.linalg.norm(strategy_coef),
          min_sep=sep,
          max_participations=participations,
      )
      loss = loss_fn(strategy_coef) * sensitivity_squared

    case 'banded-sqrt':
      # https://arxiv.org/abs/2202.11205
      # https://arxiv.org/abs/2405.13763
      strategy_coef = toeplitz.optimal_max_error_strategy_coefs(sep)
      strategy_coef = jnp.concatenate([strategy_coef, jnp.zeros(n - sep)])
      sensitivity_squared = toeplitz.minsep_sensitivity_squared(
          strategy_coef, min_sep=sep, max_participations=participations
      )
      pqe = toeplitz.per_query_error(strategy_coef=strategy_coef, n=n)
      loss = reduction_fn(pqe) * sensitivity_squared

    case 'banded-inverse-sqrt':
      # https://arxiv.org/pdf/2505.12128
      weight_decay = 1.0
      momentum = 0.0
      workload_coef = toeplitz.multiply(
          weight_decay ** jnp.arange(n),
          momentum ** jnp.arange(n),
          n=n,
          skip_checks=True,
      )
      noising_coef = toeplitz.banded_inverse_square_root_noising_coefs(
          sep, weight_decay=weight_decay, momentum=momentum
      )
      sensitivity_squared = toeplitz.compute_banded_inverse_sensitivity_squared(
          n=n,
          noising_coef=noising_coef,
          min_sep=sep,
          max_participations=participations,
      )
      pqe = toeplitz.per_query_error(
          noising_coef=noising_coef, n=n, workload_coef=workload_coef
      )
      loss = reduction_fn(pqe) * sensitivity_squared

    case 'banded':
      # https://arxiv.org/abs/2306.08153
      # https://arxiv.org/abs/2405.15913
      loss_reduction_fn = jnp.mean if objective == 'mean' else jnp.max
      strategy = banded.optimize(
          n,
          bands=sep,
          max_optimizer_steps=1000,
          reduction_fn=loss_reduction_fn,
          scan_fn='dinosaur',
      )
      sensitivity_squared = banded.minsep_sensitivity_squared(
          strategy, min_sep=sep, max_participations=participations
      )
      squared_error = loss_reduction_fn(banded.per_query_error(strategy))
      loss = squared_error * sensitivity_squared

    case 'dense':
      # https://arxiv.org/abs/2202.08312
      # https://arxiv.org/abs/2211.06530
      # https://arxiv.org/abs/2306.08153
      if objective == 'max':
        raise ValueError('Max Error is not supported for dense strategies.')
      C = dense.optimize(n, epochs=_PARTICIPATIONS.value, equal_norm=False)
      loss = dense.per_query_error(strategy_matrix=C).mean()  # sensitivity = 1

    case 'blt':
      # https://arxiv.org/abs/2404.16706
      # https://arxiv.org/abs/2408.08868
      strategy = buffered_toeplitz.optimize(
          n=n, min_sep=sep, max_participations=participations, error=objective
      )
      loss_fn = buffered_toeplitz.LossFn.build_min_sep(
          n, objective, sep, participations
      )
      loss = loss_fn.loss(strategy)

  loss = float(jnp.sqrt(loss))
  t1 = time.time()

  print(f'{strategy=} {n=} {participations=} {objective=} {loss=} time={t1-t0}')


def main(_) -> None:
  optimize_strategy(
      strategy=_STRATEGY.value,
      n=_ITERATIONS.value,
      participations=_PARTICIPATIONS.value,
      objective=_OBJECTIVE.value,
  )


if __name__ == '__main__':
  app.run(main)
