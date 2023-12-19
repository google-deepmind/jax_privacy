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

"""MIMIC-CXR dataset."""

import abc
import dataclasses
import functools
from typing import Sequence

from jax_privacy.experiments.image_data import base
from jax_privacy.experiments.image_data import loader
import tensorflow as tf


# MIMIC-CXR has the same labels as CheXpert but in a different order.
ALL_LABEL_NAMES = (
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices',
)

PATIENT_ID_10P = 10940805  # 10% percentile of patient ID
PATIENT_ID_20P = 11849767  # 20% percentile of patient ID


def decode_record(key: str, record: tf.train.Example) -> base.DataInputs:
  """Decodes the record."""
  feature_description = {
      'image': tf.io.FixedLenFeature([], tf.dtypes.string),
      'label': tf.io.VarLenFeature(tf.dtypes.float32),
  }
  parsed = tf.io.parse_single_example(record, feature_description)
  image_grayscale = tf.image.decode_image(parsed['image'], channels=1)
  image_shape = tf.shape(image_grayscale)
  image = tf.broadcast_to(
      image_grayscale,
      (image_shape[0], image_shape[1], 3),
  )

  label = tf.sparse.to_dense(parsed['label'])
  # Convert NaNs to the uncertain label (-1).
  label = tf.where(tf.math.is_nan(label), -1.0, label)
  return base.DataInputs(image=image, label=label, metadata={'key': key})


@dataclasses.dataclass(kw_only=True, slots=True)
class MimicCxrConfig(base.ImageDatasetConfig):
  """Dataset configuration for MIMIC-CXR.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    split_content: Subset split, e.g. "train".
    name: Unique identifying name for the dataset.
    image_size: Image resolution.
    all_label_names: names of all the possible classes. This must be ordered
      in a consistent way with the dataset, so that the k-th entry of a label
      vector corresponds to the k-th entry of `all_label_names`.
    select_label_names: names of the label names to use. This must be a subset
      of `all_label_names` or None. If set to None, this field will be set to
      the same value as `all_label_names`.
    indices_in_select_label_order: gives the index of each member of
      `select_label_names` within `all_label_names`.
  """
  split_content: str
  num_samples: int
  name: str = 'mimic_cxr'
  image_size: tuple[int, int] = (224, 224)
  num_classes: int = 14
  all_label_names: Sequence[str] = ALL_LABEL_NAMES
  select_label_names: Sequence[str] | None = None
  patient_id_min: int = tf.int32.min
  patient_id_max: int = tf.int32.max

  def __post_init__(self):
    if self.select_label_names is None:
      self.select_label_names = tuple(self.all_label_names)
    self.num_classes = len(self.select_label_names)

  @property
  def indices_in_select_label_order(self) -> Sequence[int]:
    return tuple(
        self.all_label_names.index(name) for name in self.select_label_names)

  @property
  def class_names(self) -> Sequence[str]:
    return self.select_label_names

  def _preprocess_label(self, label: tf.Tensor) -> tf.Tensor:
    """Pre-processes the input label."""
    # This reshape ensures that the label has the expected size.
    label = tf.reshape(label, [len(self.all_label_names)])
    label = tf.cast(label, tf.int32)
    # Sub-selects only the required classes.
    label = tf.gather(label, self.indices_in_select_label_order)
    # -1=uncertain/not mentioned, 0=negative, 1=positive
    return tf.maximum(label, 0)

  def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
    """Normalizes the input image."""
    return base.center_image(image)

  def make_fake_data(self) -> tf.data.Dataset:
    fake_data = base.DataInputs(
        image=tf.random.normal(shape=(*self.image_size, 3)),
        label=tf.random.uniform(
            shape=(len(self.all_label_names),),
            minval=0,
            maxval=2,
            dtype=tf.int32,
        ),
    )
    return tf.data.Dataset.from_tensors(fake_data).repeat(self.num_samples)

  def select_patient(self, patient_id: tf.Tensor) -> tf.Tensor:
    return tf.logical_and(
        tf.greater_equal(patient_id, self.patient_id_min),
        tf.less(patient_id, self.patient_id_max),
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class AbstractMimicCxrLoader(loader.DataLoader, metaclass=abc.ABCMeta):
  """Data loader for MIMIC-CXR."""

  config: MimicCxrConfig

  def _keep_example(self, key: tf.Tensor, record: tf.Tensor) -> tf.Tensor:
    """To use as a filter to decide whether to keep the sample."""
    del record  # decision based on the key only
    patient_id_str = tf.strings.split(key, '/')[0]
    patient_id = tf.strings.to_number(patient_id_str, tf.int32)
    return self.config.select_patient(patient_id)

  @abc.abstractmethod
  def load_raw_data(self, shuffle_files: bool) -> tf.data.Dataset:
    """Load a supervised MIMIC-CXR data set.

    Args:
      shuffle_files: whether the file names should be shuffled before being
        read.

    Returns:
      The dataset with raw examples. Each example is expected to be a
      `base.DataInputs` instance, where image is a float tensor with values in
      [0, 1], and label is a binary tensor of shape [14].
    """
    raise NotImplementedError()


MimicCxrTrainInternalConfig = functools.partial(
    MimicCxrConfig,
    patient_id_min=PATIENT_ID_20P,
    num_samples=259_094,
    split_content='train@618',
)
MimicCxrValidInternalConfig = functools.partial(
    MimicCxrConfig,
    patient_id_min=PATIENT_ID_10P,
    patient_id_max=PATIENT_ID_20P,
    num_samples=32_378,
    split_content='train@618',
)
MimicCxrTestInternalConfig = functools.partial(
    MimicCxrConfig,
    patient_id_max=PATIENT_ID_10P,
    num_samples=32_384,
    split_content='train@618',
)
MimicCxrTrainOfficialConfig = functools.partial(
    MimicCxrConfig,
    num_samples=323_856,
    split_content='train@618',
)
MimicCxrValidOfficialConfig = functools.partial(
    MimicCxrConfig,
    num_samples=2_508,
    split_content='valid@36',
)
MimicCxrTestOfficialConfig = functools.partial(
    MimicCxrConfig,
    num_samples=4_201,
    split_content='test@60',
)
