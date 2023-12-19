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

import abc
from typing import Mapping, Sequence

import chex
import haiku as hk
import jax.numpy as jnp
from jax_privacy.experiments import image_data
from jax_privacy.experiments.image_classification import metrics as metrics_module
from jax_privacy.experiments.image_classification.models import base
from jax_privacy.src.dp_sgd import typing
from jax_privacy.src.training import forward
import optax


class MultiClassForwardFn(
    forward.ForwardFn[image_data.DataInputs, hk.Params, hk.State],
):
  """Defines forward passes for learning tasks."""

  def __init__(self, net: base.Model):
    self._net = net

  def train_init(
      self,
      rng_key: chex.PRNGKey,
      inputs: image_data.DataInputs,
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
      inputs: image_data.DataInputs,
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
      network_state: new model state.
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

  @abc.abstractmethod
  def _loss(self, logits: chex.Array, labels: chex.Array) -> chex.Array:
    raise NotImplementedError()

  @abc.abstractmethod
  def _train_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    raise NotImplementedError()

  @abc.abstractmethod
  def _eval_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    raise NotImplementedError()


class MultiClassSingleLabelForwardFn(MultiClassForwardFn):
  """Defines forward passes for learning tasks with one label."""

  def __init__(self, net: base.Model, label_smoothing: float):
    super().__init__(net)
    self._label_smoothing = label_smoothing

  def eval_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng: chex.PRNGKey,
      inputs: image_data.DataInputs,
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
      Metrics computed on the current mini-batch, including the logits computed
        per-example on the mini-batch.
    """
    logits, unused_network_state = self._net.apply(
        params, network_state, rng, inputs.image, is_training=False)
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
    # NB: labels are one-hot encoded.
    if self._label_smoothing:
      labels = optax.smooth_labels(labels, alpha=self._label_smoothing)
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


class MultiClassMultiLabelForwardFn(MultiClassForwardFn):
  """Defines forward passes for learning tasks with multiple labels."""

  def __init__(
      self,
      net: base.Model,
      *,
      all_label_names: Sequence[str],
      class_indices_for_eval: Sequence[int],
  ):
    super().__init__(net)
    self._all_label_names = all_label_names
    self._class_indices_for_eval = class_indices_for_eval

  def eval_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng: chex.PRNGKey,
      inputs: image_data.DataInputs,
  ) -> typing.Metrics:
    """Forward pass per example (evaluation time).

    Args:
      params: model parameters that should get updated during training.
      network_state: model state.
      rng: random number generation key.
      inputs: model inputs, where the labels are one-hot encoded. Images are
        expected to be of shape [NHWC], and `labels` of shape [NO].
    Returns:
      Metrics computed on the current mini-batch, including the logits computed
        per-example on the mini-batch.
    """
    logits, unused_network_state = self._net.apply(
        params, network_state, rng, inputs.image, is_training=False)
    logits = logits[..., jnp.asarray(self._class_indices_for_eval)]
    loss = jnp.mean(self._loss(logits, inputs.label))

    return typing.Metrics(
        per_example={'logits': logits},
        scalars_avg={'loss': loss, **self._eval_metrics(logits, inputs.label)},
    )

  def _loss(self, logits: chex.Array, labels: chex.Array) -> chex.Array:
    # NB: labels are multilabel binary classification problem.
    # For each example, average loss of all labels.
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean(-1)

  def _train_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    return self._avg_and_per_class_accuracy_metrics(logits, labels)

  def _eval_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
  ) -> Mapping[str, chex.Numeric]:
    return self._avg_and_per_class_accuracy_metrics(
        logits, labels, class_subset=self._class_indices_for_eval)

  def _avg_and_per_class_accuracy_metrics(
      self,
      logits: chex.Array,
      labels: chex.Array,
      *,
      class_subset: Sequence[int] | None = None,
  ) -> Mapping[str, chex.Numeric]:
    """Returns the accuracy for non-mutually exclusive labels (not one-hot).

    Args:
      logits: [batch size, number of classes in subset].
      labels: [batch size, number of classes in subset].
      class_subset: Indices of classes in subset.

    Returns:
      Average accuracy over all classes and per-class.
    """
    acc_over_classes = metrics_module.per_class_acc(logits, labels)
    if class_subset is not None:
      label_names = [self._all_label_names[i] for i in class_subset]
    else:
      label_names = self._all_label_names
    acc_by_label_name = {
        'acc_' + name: 100.0 * acc
        for name, acc in zip(acc_over_classes, label_names, strict=True)
    }
    return {
        'avg_acc': 100.0 * jnp.mean(acc_over_classes),
        **acc_by_label_name,
    }
