# coding=utf-8
# Copyright 2023 DeepMind Technologies Limited.
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

"""Defines train and evaluation functions that compute losses and metrics."""

from typing import Mapping

import chex
import haiku as hk
import jax.numpy as jnp
from jax_privacy.experiments import image_data as data
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import forward
from jax_privacy.src.training import metrics as metrics_module
import optax

# TODO: investigate the pytype bug below.


class MultiClassForwardFn(
    # pytype: disable=not-indexable
    forward.ForwardFn[data.DataInputs, hk.Params, hk.State],
    # pytype: enable=not-indexable
):
  """Defines forward passes for multi-class classification."""

  def __init__(self, net: hk.TransformedWithState):
    """Initialization function.

    Args:
      net: haiku model to use for the forward pass.
    """
    self._net = net

  def train_init(
      self,
      rng_key: chex.PRNGKey,
      inputs: data.DataInputs,
  ) -> tuple[hk.Params, hk.State]:
    """Initializes the model.

    Args:
      rng_key: random number generation key used for the random initialization.
      inputs: model inputs to infer shapes of the model parameters.
        Images are expected to be of shape [NKHWC] (K is augmult).
    Returns:
      Initialized model parameters and state.
    """
    # Images has shape [NKHWC] (K is augmult).
    return self._net.init(rng_key, inputs.image[:, 0], is_training=True)

  def train_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng_per_example: chex.PRNGKey,
      inputs: data.DataInputs,
  ) -> tuple[typing.Loss, tuple[hk.State, typing.Metrics]]:
    """Forward pass per example (training time).

    Args:
      params: model parameters that should get updated during training.
      network_state: model state.
      rng_per_example: a random number generation key specific for a device and
        accumulation step. It can be used to create a unique seed per
        individual example by the user.
      inputs: model inputs, where the labels are one-hot encoded. Images are
        expected to be of shape [NKHWC] (K is augmult), and labels of shape
        [NKO].
    Returns:
      loss: loss function computed per-example on the mini-batch (averaged over
        the K augmentations).
      network_state: new model state
      metrics: metrics computed on the current mini-batch, including the loss
        value per-example.
    """
    images, labels = inputs.image, inputs.label

    # `images` has shape [NKHWC] (K is augmult), while model accepts [NHWC], so
    # we use a single larger batch dimension.
    reshaped_images = images.reshape((-1,) + images.shape[2:])
    reshaped_labels = labels.reshape((-1,) + labels.shape[2:])

    logits, network_state = self._net.apply(
        params, network_state, rng_per_example, reshaped_images,
        is_training=True)
    loss = self._loss(logits, reshaped_labels)

    # We reshape back to [NK] and average across augmentations.
    loss = loss.reshape(images.shape[:2])
    loss = jnp.mean(loss, axis=1)

    # Accuracy computation is performed with the first augmentation.
    logits = logits.reshape(images.shape[:2] + logits.shape[1:])
    selected_logits = logits[:, 0, :]
    labels = jnp.mean(labels, axis=1)

    metrics = typing.Metrics(
        scalars_avg=self._train_metrics(selected_logits, labels),
        per_example={'loss': loss},
    )
    return jnp.mean(loss), (network_state, metrics)

  def eval_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng: chex.PRNGKey,
      inputs: data.DataInputs,
  ) -> typing.Metrics:
    """Forward pass per example (evaluation time).

    Args:
      params: model parameters that should get updated during training.
      network_state: model state.
      rng: random number generation key.
      inputs: model inputs (of format `{'images': images, 'labels': labels}`),
        where the labels are one-hot encoded. `images` is expected to be of
        shape [NHWC], and `labels` of shape [NO].
    Returns:
      per_example: metrics computed per-example on the mini-batch (logits only).
      aggregated: metrics computed and aggregated on the current mini-batch.
    """
    logits, unused_network_state = self._net.apply(
        params, network_state, rng, inputs.image)
    loss = jnp.mean(self._loss(logits, inputs.label))

    return typing.Metrics(
        per_example={'logits': logits},
        scalars_avg={'loss': loss, **self._eval_metrics(logits, inputs.label)},
    )

  def _loss(self, logits: chex.Array, labels: chex.Array) -> chex.Array:
    """Compute the loss per-example.

    Args:
      logits: logits vector of expected shape [...O].
      labels: one-hot encoded labels of expected shape [...O].
    Returns:
      Cross-entropy loss computed per-example on leading dimensions.
    """
    return optax.softmax_cross_entropy(logits, labels)

  def _train_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    return self._topk_accuracy_metrics(logits, labels)

  def _eval_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    return self._topk_accuracy_metrics(logits, labels)

  def _topk_accuracy_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    """Evaluates topk accuracy."""
    # NB: labels are one-hot encoded.
    acc1, acc5 = metrics_module.topk_accuracy(logits, labels, topk=(1, 5))
    metrics = {'acc1': 100 * acc1, 'acc5': 100 * acc5}
    return metrics
