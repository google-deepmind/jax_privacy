# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
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

"""Configuration for the DP Optimization Algorithm, or, mechanism."""

import abc
import dataclasses
import enum
import hashlib
import json
import os
from typing import Any, Literal

import chex
import einshape
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax_privacy.dp_sgd import gradients
from jax_privacy.dp_sgd import typing

_exists = os.path.exists
_open = open


def _check_symmetric(matrix: typing.SquareMatrix, **allclose_kwargs) -> None:
  if not jnp.allclose(matrix, matrix.T, **allclose_kwargs):
    raise ValueError(f'Matrix must be symmetric, got: {matrix}')


class AlgorithmConfig(metaclass=abc.ABCMeta):
  """Configuration for the mechanism."""

  @property
  @abc.abstractmethod
  def noise_multiplier(self) -> float | None:
    """The noise multiplier used to generate additive noise in the algorithm."""


@dataclasses.dataclass(kw_only=True, slots=True)
class NoDpConfig(AlgorithmConfig):
  """Configuration for the (here, non)-DP Mechanism."""

  noise_multiplier: float = dataclasses.field(init=False, default=0.0)


# TODO: Rename this to DpBandMfConfig
@dataclasses.dataclass(kw_only=True, slots=True)
class DpsgdConfig(AlgorithmConfig):
  """Configuration for the DP-SGD mechanism."""

  noise_multiplier: float | None
  num_bands: int | None = None

  def __post_init__(self):
    if self.noise_multiplier is None:
      self.noise_multiplier = 0.0


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class SensitivityConfig(metaclass=abc.ABCMeta):
  """Configures the sensitivity definition."""
  # TODO: Extend this to not require a matrix.

  @abc.abstractmethod
  def sensitivity(self, encoder_matrix: jnp.ndarray) -> chex.Numeric:
    """Returns the sensitivity of the `encoder_matrix`."""


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class KbParticipation(SensitivityConfig):
  """When data is seen at most `k` times and exactly b steps apart.

  Here, b is implicitly defined by the matrix shape and the choice of `k` as
  b = `encoder_matrix`.shape[0] / `k`.

  Note that we require elementwise non-negativity for the sensitivity in the
  multidimensional case to reduce to the sensitivity of the scalar case, i.e.,
  for DP on a ML model with > 1 parameter to be valid. This is ensured herein by
  taking the absolute value of the gram matrix.

   See https://arxiv.org/pdf/2211.06530 for more details about both the
   participation pattern and how sensitivity is computed.
  """

  k: int  # max participations

  def __post_init__(self):
    if self.k <= 0:
      raise ValueError(f'k={self.k} must be positive.')

  def _check_epoch_aligned(self, num_steps: chex.Numeric):
    if num_steps % self.k != 0:
      raise ValueError(f'`k`={self.k} must divide `num_steps`={num_steps}.')

  def sensitivity(
      self,
      encoder_matrix: typing.SquareMatrix,
  ) -> chex.Numeric:
    """Returns sensitivity of `encoder_matrix` for (k,b)-participation."""
    jax.debug.callback(gradients.check_is_matrix, encoder_matrix)
    jax.debug.callback(gradients.check_square, encoder_matrix)
    jax.debug.callback(self._check_epoch_aligned, encoder_matrix.shape[0])

    blocked_encoder_matrix = einshape.jax_einshape(
        'n(kr)->nkr', encoder_matrix, k=self.k
    )
    gram_matrices = jnp.einsum(
        'nkr,nlr->rkl', blocked_encoder_matrix, blocked_encoder_matrix
    )
    squared_sensitivity = jnp.abs(gram_matrices).sum(axis=(1, 2))
    return jnp.sqrt(jnp.max(squared_sensitivity))


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class BMinSep(SensitivityConfig):
  """When data is seen at most at least `b` steps apart.

   See https://arxiv.org/abs/2306.08153 for more details about both the
   participation pattern and how sensitivity is computed.
  """

  b: int  # minimum number of steps between each participation

  def sensitivity(self, encoder_matrix: jnp.ndarray) -> chex.Numeric:
    """Returns the sensitivity of the `encoder_matrix`."""
    raise NotImplementedError('Not implemented yet.')


class NormType(enum.Enum):
  TWO = enum.auto()
  INF = enum.auto()


@enum.unique
class MatrixType(enum.Enum):
  GENERAL = enum.auto()
  TOEPLITZ = enum.auto()
  BLT = enum.auto()


@enum.unique
class OptimizationType(enum.Enum):
  GRAD_DESCENT = enum.auto()
  SDP = enum.auto()
  ANALYTICAL = enum.auto()


@enum.unique
class ObjectiveCombinationType(enum.Enum):
  """Defines how to combine the objectives in the optimization.

  This can be either arithmetic mean of squared sensitivity and squared error,
    or geometric mean of the two.

  Attributes:
    ARITHMETIC: Arithmetic mean of squared sensitivity and squared error.
    GEOMETRIC: Geometric mean of squared sensitivity and squared error.
  """
  ARITHMETIC = enum.auto()
  GEOMETRIC = enum.auto()


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class OptimizationConfig:
  """Config for the optimizer used to optimize parameters of a matrix mechanism.

  Attributes:
    optimizer_name: Name of JAXOPT optimizer used
    optimizer_kwargs: Kwargs for the Optimizer
    max_iter: Maximum number of iterations of optimizer to run
    regularization_strength: How strong to set the regularizer (if any)
    objective_combination: How to combine sensitivity and error metrices into
      a single objective
  """
  optimizer_name: str
  # TODO: Figure out why using Mapping here causes a type
  #                              error
  optimizer_kwargs: dict[str, Any]
  max_iter: int
  regularization_strength: float
  objective_combination: ObjectiveCombinationType


@dataclasses.dataclass(kw_only=True, slots=True)
class GenerationConfig:
  optimization_type: OptimizationType
  matrix_type: MatrixType
  norm_type: NormType


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class MatrixMechanismConfig(metaclass=abc.ABCMeta):
  """Configuration for the Matrix Mechanism.

  Subclasses must be a dataclass.

  Attributes:
    num_updates: int representing the number of updates that the matrix
      mechanism should be generated for.
    cache_base_path: Absolute path to the base cache directory.
    sensitivity_config: Sensitivity configuration used to generate matrices.
      Must also be a dataclass.
    workload_matrix_type: The type of workload matrix used to generate the
      matrix mechanism.
    manual_cache_file_name: Manually overwrites the filename used below. This
      may be useful if matrices are generated outside jax_privacy, and don't yet
      have a corresponding definition here, and want to be pointed to for
      experimental settings.
    generation_config: (Optional) Configuration for the matrix generation.
  """

  num_updates: int
  cache_base_path: str  # TODO: b/333769339 - Add in generation when completed.
  sensitivity_config: SensitivityConfig  # Must also be a dataclass.
  # TODO: Add in the other workload matrix types.
  # TODO: Figure out how to make this an enum that can is amenable
  #                   to JSON serialization.
  workload_matrix_type: Literal['PREFIX_SUM', 'DECAYING_SUM'] = 'PREFIX_SUM'
  manual_cache_file_name: str | None = None
  generation_config: GenerationConfig | None = None

  def _assert_positive(self, attr: str) -> None:
    """Raises ValueError if attr is not positive."""
    val = getattr(self, attr)
    if val < 1:
      raise ValueError(f'`{attr}`={val} must be positive.')

  def __post_init__(self):
    if not _exists(self.cache_base_path):
      raise ValueError(f'`path`={self.cache_base_path} must exist.')
    self._assert_positive('num_updates')

    # Otherwise, caching will not work.
    if not dataclasses.is_dataclass(self):
      raise TypeError('Subclasses must be a dataclass.')
    if not dataclasses.is_dataclass(self.sensitivity_config):
      raise TypeError('Sensitivity config must be a dataclass.')

  @property
  def cache_path(self) -> str:
    """Full path in the filesystem to the matrix defined by this config.

    All matrix mechanisms are stored under a set of subdirectories grouping
    the cached matrix mechanisms by, in order, the mechanism type, number of
    updates it's calibrated for, and the sensitivity used in generation. The
    filename is then automatically generated by hashing the remaining
    mechanism config args.
    """
    config = dataclasses.asdict(self)
    config.pop('path', None)  # Ignored for caching
    steps = config.pop('num_updates')  # Handled separately
    sensitivity_def = self.sensitivity_config.__class__.__name__
    mechanism_def = self.__class__.__name__

    subdirs = [f'{mechanism_def=}', f'{steps=}', f'{sensitivity_def=}']
    if self.manual_cache_file_name is not None:
      unique_name = self.manual_cache_file_name
    else:
      unique_name = hashlib.md5(
          json.dumps(config, sort_keys=True).encode()
      ).hexdigest()
    return os.path.join(self.cache_base_path, *subdirs, f'{unique_name}.npy')


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class OptimalMultiEpoch(MatrixMechanismConfig):
  max_participations: int
  steps_until_reparticipation: int

  def __post_init__(self):
    self._assert_positive('max_participations')
    self._assert_positive('steps_until_reparticipation')


@dataclasses.dataclass(kw_only=True, slots=True)
class DpftrlConfig(AlgorithmConfig):
  """Configuration for the DP-FTRL mechanisms via the Matrix Mechanism.

  Attributes:
    noise_multiplier: The noise multiplier relative to the clipping norm.
    mechanism_config: The dataclass configuration for the matrix mechanism.
    sensitivity_config: The sensitivity configuration for the DP Training
      experiment. Matrices are normalized with respect to this specification.
    correlation_unroll: Controls how many steps of the generation are unrolled
      for jit compilation in the jax.lax.scan. Setting higher values can improve
      generation speed but may come at a cost in memory. Default sets this to no
      unroll.
  """

  noise_multiplier: float | None
  mechanism_config: MatrixMechanismConfig  # Must be an implemented subclass.
  sensitivity_config: SensitivityConfig  # Must be an implemented subclass
  # Configures unrolling on dp_sgd.gradients.DpftrlGradientComputer.
  correlation_unroll: float | None = None

  def encoder_matrix(self) -> jnp.ndarray:
    """Returns the encoder matrix C."""
    cache_path = self.mechanism_config.cache_path
    if _exists(cache_path):
      with _open(cache_path, mode='rb') as infile:
        encoder = jnp.load(infile, allow_pickle=False)
    else:
      raise ValueError(f'No encoder found cached at path `{cache_path}`.')

    encoder /= self.sensitivity_config.sensitivity(encoder)
    return encoder


def correlation_matrix(
    encoder_matrix: typing.SquareMatrix,
) -> typing.SquareMatrix:
  gradients.check_is_matrix(encoder_matrix)
  gradients.check_square(encoder_matrix)
  gradients.check_lower_triangular(encoder_matrix)
  n = encoder_matrix.shape[0]
  return jax.scipy.linalg.solve_triangular(
      encoder_matrix, jnp.eye(n), lower=True
  )
