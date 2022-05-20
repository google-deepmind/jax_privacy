# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from jax_privacy.src.training.image_classification.data import augmult as augmult_module
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_image_dataset(
    *,
    ds: tf.data.Dataset,
    normalize_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    label_preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    num_samples: int,
    is_training: bool,
    batch_dims: Sequence[int],
    dtype: jnp.dtype = jnp.float32,
    transpose: bool = False,
    image_size: Tuple[int, int] = (224, 224),
    random_crop: bool = False,
    random_flip: bool = False,
    augmult: int = 0,
) -> Iterator[Dict[str, chex.Array]]:
  """Loads the given split of the dataset.

  Args:
    ds: Dataset without any sharding, pre-processing or augmentation.
    normalize_fn: function to normalize the images.
    label_preprocess_fn: function to pre-process the labels.
    num_samples: number of samples in the dataset split getting loaded.
    is_training: If true, use training preproc and augmentation.
    batch_dims: List indicating how to batch the dataset (typically expected to
      be of shape (num_devices, bs_per_device)
    dtype: One of float32 or bfloat16 (bf16 may not be supported fully)
    transpose: If true, employs double transpose trick.
    image_size: Final image size returned by dataset pipeline. Note that the
      exact procedure to arrive at this size will depend on the chosen preproc.
    random_crop: whether to use a random crop of the image or a center crop.
    random_flip: whether to perform random horizontal flips of the image.
    augmult: The augmentation multiplicity. Values > 1 will result in each
      batch being composed of  `batch_size // augmult` unique underlying images,
      each one copied and separately augmented `augmult` times.

  Yields:
    A TFDS numpy iterator.
  """
  # Do no apply data augmentation if augmult == 0.
  random_flip = random_flip and augmult
  random_crop = random_crop and augmult

  total_batch_size = np.prod(batch_dims)
  ds = ds.shard(jax.process_count(), jax.process_index())

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_optimization.map_parallelization = True
  options.experimental_optimization.parallel_batch = True

  # If using augmult, we use determinism.
  if is_training and augmult == 1:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.process_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=None)
  elif num_samples % total_batch_size != 0:
    raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  def preprocess(image, label):
    if random_crop:
      # Use a larger decoding size to take random crops within decoded image,
      # e.g. decoding size of 268x268 for 224x224 crops.
      decoding_size = [int(x * 1.2) for x in image_size]
    else:
      decoding_size = image_size
    image = _decode_and_crop(
        image_bytes=image,
        image_size=decoding_size,
    )
    if normalize_fn:
      image = normalize_fn(image)
    label = tf.cast(label, tf.int32)
    if label_preprocess_fn:
      label = label_preprocess_fn(label)
    if is_training:
      image, label = augmult_module.apply_augmult(
          image=image,
          label=label,
          image_size=list(decoding_size) + [3],
          crop_size=list(image_size) + [3],
          augmult=augmult,
          random_flip=random_flip,
          random_crop=random_crop,
          pad=None,
      )
    else:
      image = tf.reshape(image, list(decoding_size) + [3])

    return image, label

  ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)

  def transpose_fn(image, label):
    image = tf.transpose(image, (1, 2, 3, 0))
    return image, label

  def cast_fn(image, label):
    image = tf.cast(image, tf.dtypes.as_dtype(dtype))
    return image, label

  for i, batch_size in enumerate(reversed(batch_dims)):
    ds = ds.batch(batch_size)
    if i == 0:
      # Transpose and cast as needbe.
      if transpose:
        ds = ds.map(transpose_fn)  # NHWC -> HWCN
      # NOTE: You may be tempted to move the casting earlier on in the pipeline,
      # but for bf16 some operations will end up silently placed on the TPU and
      # this causes stalls while TF and JAX battle for the accelerator.
      ds = ds.map(cast_fn)

  ds = ds.map(lambda images, labels: {'images': images, 'labels': labels})
  ds = ds.prefetch(AUTOTUNE)
  ds = tfds.as_numpy(ds)
  yield from ds


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
    image_size: Sequence[int] = (224, 224),
) -> tf.Tensor:
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  # Pad the image with at least 32px on the short edge and take a
  # crop that maintains aspect ratio.
  scale = tf.minimum(
      tf.cast(image_height, tf.float32) / (image_size[0] + 32),
      tf.cast(image_width, tf.float32) / (image_size[1] + 32))
  padded_center_crop_height = tf.cast(scale * image_size[0], tf.int32)
  padded_center_crop_width = tf.cast(scale * image_size[1], tf.int32)
  offset_height = ((image_height - padded_center_crop_height) + 1) // 2
  offset_width = ((image_width - padded_center_crop_width) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_height,
      padded_center_crop_width
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def _decode_and_crop(
    image_bytes: tf.Tensor,
    image_size: Sequence[int],
) -> tf.Tensor:
  """Returns processed and resized images."""
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = _decode_and_center_crop(image_bytes, image_size=image_size)
  assert image.dtype == tf.uint8
  image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)
  return image


def preprocess_32x32(
    image: tf.Tensor,
    label: tf.Tensor,
    *,
    num_classes: int,
    is_training: bool,
    normalization_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    augmult: int,
    random_crop: bool,
    random_flip: bool,
    image_resize: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Pre-process a 32x32 image and its label.

  Args:
    image: image to preprocess. In this function it gets potentially normalized,
      augmented and resized.
    label: label to preprocess (specified as an integer in
      {0, ..., num_classes-1}). It gets encoded as a one-hot vector in this
      function.
    num_classes: number of possible classes.
    is_training: True for training mode, False for evaluation mode. When set to
      False, this deactivates data augmentation.
    normalization_fn: function to normalize the function before the
      data augmentation.
    augmult: number of augmentation multiplicities to use. `augmult=0`
      corresponds to no augmentation at all, `augmult=1` to standard data
      augmentation (one augmented view per mini-batch) and `augmult>1` to having
      several augmented view of each sample within the mini-batch.
    random_crop: whether to use random crops for data augmentation.
    random_flip: whether to use random horizontal flips for data augmentation.
    image_resize: size to which the image should be resized after the data
      augmentation.
  Returns:
    images: pre-processed and augmented images.
    labels: one-hot encoded labels.
  """
  image = tf.image.convert_image_dtype(image, tf.float32)
  label = tf.squeeze(tf.one_hot(label, num_classes))

  if normalization_fn is not None:
    image = normalization_fn(image)

  # Transformations that are valid only in training.
  if is_training:
    images, labels = augmult_module.apply_augmult(
        image=image,
        label=label,
        image_size=(32, 32, 3),
        crop_size=(32, 32, 3),
        augmult=augmult,
        random_flip=random_flip,
        random_crop=random_crop,
        pad=4,
    )
  else:
    images, labels = image, label

  if image_resize:
    images = tf.image.resize(images, image_resize)

  return images, labels


def load_32x32_image_dataset(
    ds: tf.data.Dataset,
    is_training: bool,
    batch_size_per_device_per_step: int,
    preprocess_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tf.data.Dataset:
  """Load dataset of 32x32 images.

  We specialize this function to 32x32 images because these require less heavy
  data pre-processing than large images, which makes the data loading simpler.

  Args:
    ds: dataset to load.
    is_training: whether in training mode.
    batch_size_per_device_per_step: batch-size to fit on each device at every
      iteration.
    preprocess_fn: function to apply to each sample being loaded.
  Returns:
    tfds dataset containing samples in numpy format.
  """
  if is_training:
    ds = ds.shard(jax.process_count(), jax.process_index())
    # Shuffle before repeat ensures all examples seen in an epoch.
    # See https://www.tensorflow.org/guide/data_performance#repeat_and_shuffle.
    ds = ds.shuffle(buffer_size=50000)
    ds = ds.repeat()

  if preprocess_fn is not None:
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
  ds = ds.batch(
      batch_size_per_device_per_step, drop_remainder=True)
  if is_training:
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
  ds = ds.map(lambda images, labels: {'images': images, 'labels': labels})
  ds = ds.prefetch(AUTOTUNE)
  ds = tfds.as_numpy(ds)
  return ds
