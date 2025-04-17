# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
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

"""Common functions used to define model architectures."""

import abc
import dataclasses
from typing import Any, Callable, NamedTuple, Protocol

import haiku as hk
import jax
from image_classification.models import common


def _rescale_fn(
    fn: Callable[[jax.Array], jax.Array],
    scale: float
) -> Callable[[jax.Array], jax.Array]:
  """Rescales the activation a function by a multiplicative factor."""
  return lambda x: scale * fn(x)


class HkFn(Protocol):
  def __call__(
      self,
      image: jax.Array,
      is_training: bool,
  ) -> jax.Array:
    """Haiku function to transform."""


class InitFn(Protocol):
  def __call__(
      self,
      rng_key: jax.Array,
      image: jax.Array,
      is_training: bool,
  ) -> tuple[hk.Params, hk.State]:
    """Initializes the model parameters and state."""


class ApplyFn(Protocol):
  def __call__(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng_per_example: jax.Array,
      images: jax.Array,
      is_training: bool,
  ) -> jax.Array:
    """Forward pass on the model."""


class Model(NamedTuple):
  """Model."""

  init: InitFn
  apply: ApplyFn

  @classmethod
  def from_hk_module(
      cls,
      model: Callable[..., HkFn],
      *args: Any,
      **kwargs: Any,
  ) -> 'Model':
    """Returns a model given a Haiku module."""
    def net(image: jax.Array, is_training: bool):
      return model(*args, **kwargs)(image, is_training)
    transformed = hk.transform_with_state(net)
    return cls(
        init=transformed.init,
        apply=transformed.apply,
    )


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ModelConfig(abc.ABC):
  """Config for the model."""

  @abc.abstractmethod
  def make(self, num_classes: int) -> Model:
    """Returns a new instance of the model."""


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class WithRestoreModelConfig(ModelConfig):
  """Configuration for restoring the model.

  Attributes:
    model: Model to use.
    path: Path to the model to restore.
    params_key: (dictionary) Key identifying the parameters in the checkpoint to
      restore.
    network_state_key: (dictionary) Key identifying the model state in the
      checkpoint to restore.
    layer_to_ignore: Optional identifying name of the layer to ignore when
      loading the checkpoint (useful for ignoring the classification layer to
      use a different number of classes for example).
  """

  model: ModelConfig
  path: str | None = None
  params_key: str | None = None
  network_state_key: str | None = None
  layer_to_ignore: str | None = None

  def _should_reset_layer(self, module_name, *_):
    return module_name == self.layer_to_ignore

  def make(self, num_classes: int) -> Model:
    model = self.model.make(num_classes=num_classes)

    if self.path:
      params_loaded, network_state_loaded = common.restore_from_path(
          restore_path=self.path,
          params_key=self.params_key,
          network_state_key=self.network_state_key,
      )
      _, params_loaded = hk.data_structures.partition(
          self._should_reset_layer, params_loaded)
      _, network_state_loaded = hk.data_structures.partition(
          self._should_reset_layer, network_state_loaded)
    else:
      params_loaded, network_state_loaded = {}, {}

    def init(rng_key: jax.Array, image: jax.Array, is_training: bool):
      params, network_state = model.init(rng_key, image, is_training)
      if self.path:
        # Note that the 'loaded' version must be last in the merge to get
        # priority.
        params = hk.data_structures.merge(params, params_loaded)
        network_state = hk.data_structures.merge(
            network_state, network_state_loaded)
      return params, network_state
    return Model(init=init, apply=model.apply)
