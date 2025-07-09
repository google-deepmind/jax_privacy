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

"""Tools to enable privacy auditing."""

import dataclasses
import enum
import functools
import itertools
from typing import Any, Iterator, Mapping, Sequence

import chex
from einshape import jax_einshape as einshape
import haiku as hk
import jax
import jax.numpy as jnp
from jax_privacy.dp_sgd import typing
import image_data
from image_classification import evaluator
from image_classification import metrics as metrics_module
from image_data import base
from image_data import loader
from jax_privacy.training import experiment_config
from jax_privacy.training import forward
from jax_privacy.training import updater
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import typing_extensions


class CanaryType(enum.Enum):
  RANDOM_GRADIENT = 'random_gradient'
  MISLLABELLED_INPUT = 'mislabelled_input'
  CUSTOM = 'custom'


@dataclasses.dataclass(kw_only=True, slots=True)
class AuditingConfig:
  """Configuration for auditing.

  Attributes:
    canary_type: Type of canaries to use.
    canary_every_n_steps: How often to inject a canary example.
    num_canaries: Number of canaries to use for input canaries.
    in_canary_eval: Data loader for in-canary evaluation.
    out_canary_eval: Data loader for out-canary evaluation.
  """

  canary_type: CanaryType
  # Configs below are only used if gradient_canary is True.
  canary_every_n_steps: int | None = None
  # Configs below are only used if gradient_canary is False.
  num_canaries: int | None = None
  in_canary_eval: loader.DataLoader | None = None
  out_canary_eval: loader.DataLoader | None = None

  def __post_init__(self):
    match self.canary_type:
      case CanaryType.RANDOM_GRADIENT:
        if self.canary_every_n_steps is None:
          raise ValueError(
              'canary_every_n_steps must be set for random gradient canaries.'
          )
      case CanaryType.MISLLABELLED_INPUT:
        if self.num_canaries is None:
          raise ValueError(
              'num_canaries must be set for mislabelled input canaries.'
          )
      case CanaryType.CUSTOM:
        if self.in_canary_eval is None:
          raise ValueError('custom_canary_fn must be set for custom canaries.')
      case _:
        raise ValueError(f'Unsupported canary type: {self.canary_type}')


class AuditingLoggingConfig(experiment_config.LoggingConfig):
  """Logging configuration for auditing."""

  def __init__(self, base_logging_config: experiment_config.LoggingConfig):
    super().__init__(**dataclasses.asdict(base_logging_config))

  @typing_extensions.override
  def additional_training_metrics(
      self, params: hk.Params, grads: hk.Params
  ) -> Mapping[str, chex.Numeric]:
    del params
    canary_grad = jax.tree.map(jnp.ones_like, grads)
    return {'canary_dot_prod': _dot_product(canary_grad, grads)}


# TODO: b/411323638 - Enable support for grain datasets in auditing.


def data_inputs_to_dataset(
    data_inputs: Sequence[base.DataInputs],
) -> tf.data.Dataset:
  """Converts a list of DataInputs to a tf.data.Dataset using from_tensor_slices."""
  # Separate the fields into lists
  images = [inputs.image for inputs in data_inputs]
  labels = [inputs.label for inputs in data_inputs]

  # Convert the lists to tensors
  images_tensor = tf.cast(tf.stack(images), dtype=images[0].dtype)
  labels_tensor = tf.cast(tf.stack(labels), dtype=labels[0].dtype)

  # # Create the dataset from the tensors
  dataset = tf.data.Dataset.from_tensor_slices({
      'image': images_tensor,
      'label': labels_tensor,
  })

  return dataset.map(base.DataInputs.from_dict)


def load_raw_data_mixed_with_canaries(
    canaries: Sequence[base.DataInputs],
    num_examples: int,
    base_load_raw_data_fn,
    data_loader,
    *,
    shuffle_files: bool,
) -> tf.data.Dataset:
  """Loads raw data mixed with canaries."""
  del data_loader
  assert num_examples >= len(canaries)
  canaries_ds = data_inputs_to_dataset(canaries)
  ds = base_load_raw_data_fn(shuffle_files=shuffle_files)
  ds_without_replaced = ds.skip(len(canaries))
  combined_ds = canaries_ds.concatenate(ds_without_replaced)
  return combined_ds


def data_loader_with_examples_replaced_by_canaries(
    data_loader: loader.DataLoaderType,
    canaries: Sequence[base.DataInputs] | loader.DataLoader,
) -> loader.DataLoaderType:
  """Wraps a data loader to add canaries at a fixed rate."""

  if isinstance(canaries, loader.DataLoader):
    canaries = canaries.load_raw_data_tf(shuffle_files=False)
    ds = tfds.as_numpy(canaries)
    canaries = [base.DataInputs(*example) for example in ds]

  new_load_raw_data_fn = functools.partial(
      load_raw_data_mixed_with_canaries,
      canaries,
      data_loader.config.num_samples,
      data_loader.load_raw_data_tf,
  )
  return dataclasses.replace(
      data_loader,
      load_raw_data_tf_fn=new_load_raw_data_fn,
  )


def data_loader_with_only_canaries(
    data_loader: loader.DataLoaderType,
    canaries: Sequence[base.DataInputs],
) -> loader.DataLoaderType:
  """Wraps a data loader to load only canaries."""

  def new_load_raw_data_fn(
      data_loader,
      *,
      shuffle_files,
  ):
    del data_loader, shuffle_files
    return data_inputs_to_dataset(canaries)

  new_config = dataclasses.replace(
      data_loader.config, num_samples=len(canaries)
  )
  return dataclasses.replace(
      data_loader,
      load_raw_data_tf_fn=new_load_raw_data_fn,
      config=new_config,
  )


def load_dataset_with_fixed_rate_canaries(
    canary_every_n_steps: int,
    base_load_dataset_fn: loader.LoadDatasetFn,
    data_loader: loader.DataLoader,
    *,
    is_training: bool,
    shard_data: bool,
    batch_dims: Sequence[int],
    drop_metadata: bool = True,
    max_num_batches: int | None = None,
) -> Iterator[base.DataInputs]:
  """Wraps a LoadDatasetFn to add canaries at a fixed rate.

  Args:
    canary_every_n_steps: How often to inject a canary example.
    base_load_dataset_fn: The base load dataset function to wrap.
    data_loader: The data loader to use.
    is_training: If set to true, data augmentation may be applied to each batch
      of data.
    shard_data: Whether to shard data across hosts, i.e. to partition the data
      with each host only seeing its own subset (shard) of the partition. It
      should be enabled if and only if data is not batched across hosts.
    batch_dims: The size of each dimension to be batched.
    drop_metadata: Whether to drop the metadata in the batch (True by default).
      This can be useful when the metadata does not have the consistent shapes
      required by pmapped functions.
    max_num_batches: Maximum number of batches to load.

  Returns:
    A TFDS numpy iterator.
  """
  del drop_metadata
  train_data_iterator = base_load_dataset_fn(
      data_loader,
      is_training=is_training,
      shard_data=shard_data,
      batch_dims=batch_dims,
      drop_metadata=False,  # Canary annotation is in metadata.
      max_num_batches=max_num_batches,
  )

  per_host_batch_size = int(np.prod(batch_dims))

  def maybe_insert_canary(index: int, inputs: image_data.DataInputs):
    inject_step = index % canary_every_n_steps == 0
    if jax.process_index() == 0 and inject_step:
      is_canary = jnp.array(
          [1] + [0] * (per_host_batch_size - 1), dtype=jnp.bool
      ).reshape((jax.local_device_count(), -1))
      inputs = dataclasses.replace(
          inputs,
          metadata={
              'is_canary': is_canary,
              **{k: v for k, v in inputs.metadata.items() if k != 'is_canary'},
          },
      )

    return inputs

  train_data_iterator = itertools.starmap(
      maybe_insert_canary, enumerate(train_data_iterator)
  )
  return train_data_iterator


def data_loader_with_fixed_rate_canaries(
    data_loader: loader.DataLoaderType,
    canary_every_n_steps: int,
) -> loader.DataLoaderType:
  """Wraps a data loader to add canaries at a fixed rate."""

  new_load_dataset_fn = functools.partial(
      load_dataset_with_fixed_rate_canaries,
      canary_every_n_steps,
      data_loader.load_dataset_fn,
  )
  return dataclasses.replace(
      data_loader,
      load_dataset_fn=new_load_dataset_fn,
  )


def generate_in_out_mislabelled_example_canaries(
    data: loader.DataLoader,
    num_canaries: int,
) -> tuple[list[base.DataInputs], list[base.DataInputs]]:
  """Generates in and out canaries by mislabeling examples.

  Arguments:
    data: The data loader to use.
    num_canaries: The number of canaries to generate in each dataset.

  Returns:
    A tuple of two lists of canaries, the first being the in canaries and the
    second being the out canaries.
  """
  dataset = data.load_raw_data_tf(shuffle_files=True).take(2 * num_canaries)
  ds = tfds.as_numpy(dataset)

  canaries = []
  for example in ds:
    updated_example = base.DataInputs(
        image=example.image,
        label=np.random.choice(
            np.setdiff1d(np.arange(data.config.num_classes), example.label)
        ),
    )
    canaries.append(updated_example)

  return canaries[:num_canaries], canaries[num_canaries:]


class CanaryGradientForwardFnWrapper(
    forward.ForwardFn[image_data.DataInputs, hk.Params, hk.State]
):
  """Wrapper for adding canary gradients to a forward function."""

  def __init__(
      self,
      base_forward_fn: forward.ForwardFn[
          image_data.DataInputs, hk.Params, hk.State
      ],
  ):
    self._base_forward_fn = base_forward_fn

  def _canary_loss(self, params: hk.Params):
    canary_grad = jax.tree.map(jnp.ones_like, params)
    return _dot_product(canary_grad, params)

  @typing_extensions.override
  def train_forward(
      self,
      params: hk.Params,
      network_state: hk.State,
      rng_per_example: chex.PRNGKey,
      inputs: image_data.DataInputs,
  ) -> tuple[typing.Loss, tuple[hk.State, typing.Metrics]]:
    batch_size = inputs.image.shape[0]

    is_canary = inputs.metadata.get(
        'is_canary', jnp.zeros(batch_size, dtype=jnp.bool)
    )
    weights = inputs.metadata.get('weights', jnp.ones(batch_size))
    inputs = dataclasses.replace(
        inputs,
        metadata={
            'weights': weights * (1 - is_canary),
            **{k: v for k, v in inputs.metadata.items() if k != 'weights'},
        },
    )

    loss, (network_state, metrics) = self._base_forward_fn.train_forward(
        params, network_state, rng_per_example, inputs
    )
    canary_loss = self._canary_loss(params) * jnp.sum(is_canary) / batch_size
    loss += canary_loss

    metrics = dataclasses.replace(
        metrics,
        scalars_avg={'canary_loss': canary_loss, **metrics.scalars_avg},
        scalars_sum={'canary_count': jnp.sum(is_canary), **metrics.scalars_sum},
        per_example={'is_canary': is_canary, **metrics.per_example},
    )

    return loss, (network_state, metrics)

  @typing_extensions.override
  def train_init(
      self,
      rng_key: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> tuple[typing.ParamsT, typing.ModelStateT]:
    return self._base_forward_fn.train_init(rng_key, inputs)

  @typing_extensions.override
  def eval_forward(
      self,
      params: typing.ParamsT,
      network_state: typing.ModelStateT,
      rng: chex.PRNGKey,
      inputs: typing.InputsT,
  ) -> typing.Metrics:
    return self._base_forward_fn.eval_forward(
        params, network_state, rng, inputs
    )


def _dot_product(u: hk.Params, v: hk.Params) -> chex.Numeric:
  assert jax.tree.all(jax.tree.map(lambda x, y: x.shape == y.shape, u, v))
  return jnp.array(
      jax.tree.leaves(
          jax.tree.map(
              lambda x, y: jnp.sum(x * y),
              u,
              v,
          )
      )
  ).sum()


class ImageClassificationAuditEvaluator(evaluator.ImageClassificationEvaluator):
  """Evaluator used when auditing is enabled."""

  def audit_dataset(
      self,
      updater_state: updater.UpdaterState,
      ds_iterator: Iterator[image_data.DataInputs],
  ) -> Mapping[str, Any]:
    """Audit a full dataset in an iterative fashion.

    Args:
      updater_state: Updater state.
      ds_iterator: Data iterator of objects of shape (num_hosts,
        num_devices_per_host, batch_size_per_device, *individual_shape).

    Returns:
      metrics.
    """

    num_samples = 0
    host_id = jax.process_index()

    all_params = ['last', *updater_state.params_avg]
    logits_by_params = {
        k: metrics_module.ArrayConcatenater() for k in all_params
    }
    labels_by_params = {
        k: metrics_module.ArrayConcatenater() for k in all_params
    }

    for inputs in ds_iterator:

      num_hosts, num_devices_per_host, batch_size_per_device, *_ = (
          inputs.image.shape
      )
      batch_size = num_hosts * num_devices_per_host * batch_size_per_device
      num_samples += batch_size
      local_inputs = jax.tree_util.tree_map(lambda x: x[host_id], inputs)

      metrics_by_params = self._evaluate_batch(
          updater_state=updater_state, inputs=local_inputs
      )

      for params_name, metrics in metrics_by_params.items():

        # Returned logits are across all devices (flattened over all hosts).
        logits = einshape(
            '(hd)bk->(hdb)k',
            metrics.per_example['logits'],
            h=num_hosts,
            d=num_devices_per_host,
            b=batch_size_per_device,
        )
        logits_by_params[params_name].append(logits)
        # Use labels across all devices on all hosts.
        labels = einshape(
            'hdbk->(hdb)k',
            inputs.label,
            h=num_hosts,
            d=num_devices_per_host,
            b=batch_size_per_device,
        )
        labels_by_params[params_name].append(labels)

    metrics = {}
    metrics['num_samples'] = num_samples

    for params_name, logits in logits_by_params.items():
      logits_array = logits.asarray()
      if params_name in labels_by_params:
        labels_array = labels_by_params[params_name].asarray()
        loss_values = np.sum(
            -jax.nn.log_softmax(logits_array) * labels_array, axis=-1
        )
        for i, loss_value in enumerate(loss_values):
          metrics[f'loss_{i}_{params_name}'] = float(loss_value)

    return metrics
