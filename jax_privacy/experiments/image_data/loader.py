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

"""Image dataset loader with typical pre-processing and advanced augs."""

import abc
import dataclasses
import functools
import itertools
from typing import Iterator, Sequence

import jax
from jax_privacy.experiments.image_data import augmult
from jax_privacy.experiments.image_data import base
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass(kw_only=True, slots=True)
class DataLoader(metaclass=abc.ABCMeta):
  """Create a data loader.

  Attributes:
    config: Dataset configuration.
    augmult_config: Configuration for data augmentation. If set to None, no
      data augmentation is applied. NOTE: data augmentation is ignored in
      evaluation mode.
    debug: Whether to load fake data for debugging purposes.
    cache_train: Whether the training dataset should be cached. This should only
      be set to True for small datasets.
  """

  config: base.DatasetConfig
  augmult_config: augmult.AugmultConfig | None = None
  debug: bool = False
  cache_train: bool = False

  @abc.abstractmethod
  def load_raw_data(self, shuffle_files: bool) -> tf.data.Dataset:
    """Method for loading the raw tensorflow dataset."""

  def load_dataset(
      self,
      *,
      is_training: bool,
      shard_data: bool,
      batch_dims: Sequence[int],
      drop_metadata: bool = True,
      max_num_batches: int | None = None,
  ) -> Iterator[base.DataInputs]:
    """Loads the dataset and preprocesses it.

    In training mode, each batch has the shape `num_local_devices x
    batch_size_per_device x augmult x example_shape.`
    In evaluation mode, each batch has the shape `num_processes x
    num_local_devices x batch_size_per_device x augmult x example_shape.`

    Args:
      is_training: If set to true, data augmentation may be applied to each
        batch of data.
      shard_data: Whether to shard data across hosts, i.e. to partition the data
        with each host only seeing its own subset (shard) of the partition. It
        should be enabled if and only if data is not batched across hosts.
      batch_dims: The size of each dimension to be batched.
      drop_metadata: Whether to drop the metadata in the batch (True by
        default). This can be useful when the metadata does not have the
        consistent shapes required by pmapped functions.
      max_num_batches: Maximum number of batches to load.

    Yields:
      A TFDS numpy iterator.
    """
    if self.debug:
      ds = self.config.make_fake_data()
    else:
      ds = self.load_raw_data(shuffle_files=is_training)
    if shard_data:
      ds = ds.shard(jax.process_count(), jax.process_index())

    if drop_metadata:
      ds = ds.map(lambda x: base.DataInputs(image=x.image, label=x.label))

    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    if is_training:
      if self.cache_train:
        ds = ds.cache()
      ds = ds.repeat()
      ds = ds.shuffle(
          buffer_size=10*np.prod(batch_dims),
          seed=None,
      )

    ds = ds.map(
        functools.partial(
            self.config.preprocess,
            augmult_config=self.augmult_config,
            is_training=is_training,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    for dim in reversed(batch_dims):
      ds = ds.batch(dim, drop_remainder=is_training)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    if max_num_batches is not None:
      ds = itertools.islice(ds, max_num_batches)
    yield from ds
