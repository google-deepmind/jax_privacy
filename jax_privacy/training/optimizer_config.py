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

"""Optim utils."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import haiku as hk
import jax
from jax_privacy.training import experiment_config
import optax


@dataclasses.dataclass(kw_only=True, slots=True)
class LearningRateConfig:
  """Configuration for the learning-rate.

  Attributes:
    name: name of the optax decay schedule to use.
    kwargs: keyword arguments for the optax decay schedule.
    relative_kwargs: name of a keyword argument provided in
      kwargs that is defined only relatively to the total number
      of model updates. Its value will later be multiplied by the number of
      model updates so that it can be correctly interpreted by optax.
  """
  name: str
  kwargs: Mapping[str, Any]
  relative_kwargs: Sequence[str] | None = None


def constant_lr_config(
    value: float,
) -> LearningRateConfig:
  return LearningRateConfig(
      name='constant_schedule',
      kwargs={'value': value},
  )


def cosine_decay_lr_config(
    *,
    init_value: float,
    alpha: float = 0.0,
) -> LearningRateConfig:
  return LearningRateConfig(
      name='cosine_decay_schedule',
      kwargs={
          'init_value': init_value,
          'alpha': alpha,
          'decay_steps': 1.0,
      },
      relative_kwargs=['decay_steps'],
  )


@dataclasses.dataclass(kw_only=True, slots=True)
class OptimizerConfig:
  """Configuration for the optimizer.

  Attributes:
    name: Name of the optax optimizer to use.
    kwargs: Keyword arguments for the optax optimizer.
    lr: Learning-rate configuration.
  """

  name: str
  lr: LearningRateConfig
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def make_lr_schedule_fn(self, max_num_updates: int) -> optax.Schedule:
    """Creates the learning-rate schedule based on the number of updates."""
    if isinstance(self.lr, float):
      return optax.constant_schedule(self.lr)
    else:
      kwargs = {**self.lr.kwargs}
      if self.lr.relative_kwargs is not None:
        # Adapt relative arguments by multiplying them by `max_num_updates`.
        for kwarg_name in self.lr.relative_kwargs:
          rel_val = kwargs[kwarg_name]
          abs_val = rel_val * max_num_updates
          kwargs[kwarg_name] = abs_val

      return getattr(
          optax,
          self.lr.name,
      )(**kwargs)

  def make_optimizer(
      self,
      max_num_updates: int,
  ) -> optax.GradientTransformation:
    optimizer = getattr(optax, self.name)
    return optimizer(
        self.make_lr_schedule_fn(max_num_updates),
        **self.kwargs,
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class AgcOptimizerConfig(OptimizerConfig):
  """Configuration for Adaptive Gradient Clipping optimizer.

  This is useful in particular to stabilize the training of NF-ResNets at
  large batch-sizes and NFNets.

  References:
    [Brock, De, Smith, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization. (https://arxiv.org/abs/2102.06171)

  Attributes:
    filter_fn: On which parameters to enable AGC. If set to None, this
      corresponds to enabling AGC on all parameters.
    name: Name of the optax optimizer to use.
    kwargs: Keyword arguments for the optax optimizer.
    lr: Learning-rate configuration.
    clipping: The maximum allowed ratio of update norm to parameter norm.
    eps: An epsilon term to prevent clipping of zero-initialized params. Usually
      significantly larger than the epsilon used for Adam (defauilt: 1e-3).
  """

  name: str
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  lr: LearningRateConfig | float
  clipping: float = 0.01
  eps: float = 1e-3
  filter_fn: experiment_config.FilterFn | None = None

  def make_optimizer(
      self,
      max_num_updates: int,
  ) -> optax.GradientTransformation:
    # TODO: investigate use of super() here.
    base_optimizer = getattr(optax, self.name)(
        self.make_lr_schedule_fn(max_num_updates),
        **self.kwargs,
    )

    # The AGC optimizer clips the gradient with the AGC rule before applying
    # the transformation of `base_optimizer`.
    agc_optimizer = optax.chain(
        optax.adaptive_grad_clip(self.clipping, self.eps),
        base_optimizer,
    )

    def label_parameters(tree: hk.Params):
      if self.filter_fn is None:
        # If no filter_fn is provided, all leaves of the tree are tagged as
        # 'agc'.
        return jax.tree_util.tree_map(lambda x: 'agc', tree)
      else:
        # Leaves for which `self.filter_fn` returns True are tagged as 'agc',
        # and other leaves are tagged as 'no_agc'.
        label_map = {True: 'agc', False: 'no_agc'}
        return hk.data_structures.map(
            lambda *args: label_map[self.filter_fn(*args)], tree)

    return optax.multi_transform(
        {'agc': agc_optimizer, 'no_agc': base_optimizer},
        param_labels=label_parameters,
    )


def adam_config(
    *,
    lr: LearningRateConfig,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> OptimizerConfig:
  return OptimizerConfig(
      name='adam',
      lr=lr,
      kwargs={'eps': eps, 'b1': b1, 'b2': b2},
  )


def sgd_config(
    *,
    lr: LearningRateConfig,
    momentum: float | None = None,
    nesterov: bool = False,
) -> OptimizerConfig:
  return OptimizerConfig(
      name='sgd',
      lr=lr,
      kwargs={'momentum': momentum, 'nesterov': nesterov},
  )


def agc_config(
    *,
    lr: LearningRateConfig,
    filter_fn: experiment_config.FilterFn | None = None,
    momentum: float | None = None,
    nesterov: bool = False,
    clipping: float = 0.01,
    eps: float = 1e-3,
) -> AgcOptimizerConfig:
  return AgcOptimizerConfig(
      name='sgd',
      lr=lr,
      kwargs={'momentum': momentum, 'nesterov': nesterov},
      clipping=clipping,
      eps=eps,
      filter_fn=filter_fn,
  )
