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

"""CheXpert dataset with typical pre-processing and advanced augs."""

import abc
import dataclasses
import enum
import functools
from typing import Mapping, Sequence

import jax
from jax_privacy.experiments.image_data import base
from jax_privacy.experiments.image_data import loader
import tensorflow as tf


ALL_LABEL_NAMES = (
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
)
TRAIN_LABEL_NAMES = ALL_LABEL_NAMES
EVAL_LABEL_NAMES = (
    'Atelectasis', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Pleural Effusion',
)

MEAN_GRAYSCALE = 0.54725
STDDEV_GRAYSCALE = 0.2719352

# NOTE: the number of samples in the various configurations at the bottom of
# this file will need to be re-computed offline if this seed changes: since
# every patient is associated with a variable number of images, the total number
# of images per split needs to be manually evaluated (e.g. by running through
# the data loaded at batch-size 1 and counting the number of iterations).
_PATIENT_PERMUTATION_SEED = 127634
_NUM_PATIENTS = 64_540


class LabelMeaning(enum.Enum):
  """How to interpret label values."""

  UNCERTAIN = (0, -1)
  POSITIVE = (1, 1)
  NEGATIVE = (2, 0)
  UNMENTIONED = (3, -1)

  def meaning(self, split: str) -> int:
    if split in ('train', 'validation'):
      return self.value[0]
    elif split == 'test':
      return self.value[1]
    else:
      raise ValueError(f'Invalid split {split}.')


def _normalize_image(
    image: tf.Tensor,
    preprocess_name: str,
) -> tf.Tensor:
  """Returns normalisation function given image_size and preprocess_name."""
  if preprocess_name == 'center':
    return base.center_image(image)
  elif preprocess_name == 'standardise':
    return base.standardize_image_per_channel(
        image,
        mean_per_channel=MEAN_GRAYSCALE,
        stddev_per_channel=STDDEV_GRAYSCALE,
    )
  else:
    raise NotImplementedError(f'Preprocessing with {preprocess_name} not '
                              'implemented for ChexPert.')


@dataclasses.dataclass(kw_only=True, slots=True)
class ChexpertConfig(base.ImageDatasetConfig):
  r"""Dataset configuration for CheXpert.

  Attributes:
    num_samples: Number of examples in the dataset split.
    num_classes: Number of label classes for the dataset.
    split_content: Subset split, e.g. "train".
    name: Unique identifying name for the dataset.
    image_size: Image resolution.
    preprocess_name: Name of preprocessing function.
    all_label_names: names of all the possible classes. This must be ordered
      in a consistent way with the dataset, so that the k-th entry of a label
      vector corresponds to the k-th entry of `all_label_names`.
    select_label_names: names of the label names to use. This must be a subset
      of `all_label_names`.
    indices_in_select_label_order: gives the index of each member of
      `select_label_names` within `all_label_names`.
    uncertain_to_positive: names of the labels to map to a positive label. The
      other labels will be mapped to a negative label.
    uncertain_label_smoothing: label smoothing to apply to the uncertain class.
      For example, when set to 0.2, positive classes will use a target of 0.8
      and negative classes a target of 0.2. This should be set to 0.0 at
      evaluation time.
    patient_percentile_min: min percentile of patient ID to use to sub-select
      patient for the split.
    patient_percentile_max: max percentile of patient ID to use to sub-select
      patient for the split.
    label_dtype: TensorFlow type to use for the label. For training, using a
      float type may be useful to model uncertainty (through label smoothing),
      and for evaluation, an integer type is required.
  """
  split_content: str
  num_samples: int
  image_size: tuple[int, int] = (224, 224)
  name: str = 'chexpert'
  num_classes: int = -1
  preprocess_name: str = 'center'
  all_label_names: Sequence[str] = ALL_LABEL_NAMES
  select_label_names: Sequence[str]
  uncertain_to_positive: Sequence[str] = (
      'Atelectasis', 'Edema', 'Pleural Effusion')
  uncertain_label_smoothing: float = 0.0
  patient_percentile_min: float = 0.0
  patient_percentile_max: float = 1.0

  def __post_init__(self):
    self.num_classes = len(self.select_label_names)
    if (self.patient_percentile_min != 0.0
        or self.patient_percentile_max != 1.0):
      assert (
          0
          <= self.patient_percentile_min
          <= self.patient_percentile_max
          <= 1.0
      )
    if not 0.0 <= self.uncertain_label_smoothing <= 0.5:
      raise ValueError(
          f'Invalid smoothing value: {self.uncertain_label_smoothing}.'
      )

  @property
  def label_dtype(self) -> tf.dtypes.DType:
    return tf.int32 if self.uncertain_label_smoothing == 0.0 else tf.float32

  @property
  def indices_in_select_label_order(self) -> Sequence[int]:
    return tuple(
        self.all_label_names.index(name) for name in self.select_label_names)

  @property
  def uncertain_labels(self) -> tf.Tensor:
    """Maps uncertain labels to (smoothed) positive / negative label."""
    labels = [self.uncertain_label_smoothing for _ in self.all_label_names]
    for label_name in self.uncertain_to_positive:
      label_idx = self.all_label_names.index(label_name)
      labels[label_idx] = 1.0 - self.uncertain_label_smoothing
    return tf.cast(labels, dtype=self.label_dtype)

  @property
  def class_names(self) -> Sequence[str]:
    return self.select_label_names

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

  def _preprocess_label(self, label: tf.Tensor) -> tf.Tensor:
    """Pre-processes the input labels (expecting a Tensor of shape [14])."""

    label = tf.cast(label, self.label_dtype)
    zero = tf.zeros((), dtype=self.label_dtype)
    one = tf.ones((), dtype=self.label_dtype)

    # Map each uncertain label to its (potentially smoothed) value.
    label = tf.where(
        tf.equal(label, LabelMeaning.UNCERTAIN.meaning(self.split_content)),
        self.uncertain_labels,
        label,
    )
    # Map all positive labels to 1.
    label = tf.where(
        tf.equal(label, LabelMeaning.POSITIVE.meaning(self.split_content)),
        one, label)
    # Map all negative and unmentioned labels to 0.
    label = tf.where(
        tf.equal(label, LabelMeaning.NEGATIVE.meaning(self.split_content)),
        zero, label)
    label = tf.where(
        tf.equal(label, LabelMeaning.UNMENTIONED.meaning(self.split_content)),
        zero, label)

    # Retain only desired labels.
    label = tf.gather(label, self.indices_in_select_label_order)
    return label

  def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
    """Normalizes the input image."""
    return _normalize_image(
        image,
        preprocess_name=self.preprocess_name,
    )

  def select_patient(self, patient_id: tf.Tensor) -> tf.Tensor:
    """Whether the patient should be selected based on min and max percentiles.

    Args:
      patient_id: single int Tensor containing the patient id, with values in
        [1, _NUM_PATIENTS].
    Returns:
      Whether the patient should be selected.
    """
    if self.split_content != 'train':
      raise ValueError(f'Not implemented for split "{self.split_content}".')

    permuted_patient_id = tf.random.experimental.index_shuffle(
        index=tf.reshape(patient_id-1, [1]),  # 0-indexing and required ndim=1
        seed=jax.random.PRNGKey(_PATIENT_PERMUTATION_SEED),
        max_index=_NUM_PATIENTS-1,  # result in {0, ..., _NUM_PATIENTS-1}
    )
    permuted_patient_id = tf.cast(
        tf.reshape(permuted_patient_id, []), tf.float32)

    min_patient_id = self.patient_percentile_min * _NUM_PATIENTS
    max_patient_id = self.patient_percentile_max * _NUM_PATIENTS
    return tf.logical_and(
        tf.greater_equal(permuted_patient_id, min_patient_id),
        tf.less(permuted_patient_id, max_patient_id),
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class AbstractChexpertLoader(loader.DataLoader, metaclass=abc.ABCMeta):
  """Data loader for CheXpert.

  Attributes:
    config: Dataset configuration.
    debug: Whether to load fake data for debugging.
    cache_train: Whether to cache the training dataset.
    preprocessed_resolution: Resolution of the images in the dataset to load.
      Should be set to 300 or 512.
    subset_boundaries: If not None, only load a subset of the data, defined by
      the slice boundaries in this tuple.
  """

  config: ChexpertConfig
  debug: bool = False
  cache_train: bool = False
  preprocessed_resolution: int = 300

  def __post_init__(self):
    if self.preprocessed_resolution not in (300, 512):
      raise ValueError(
          'Dataset not available for'
          f' {self.preprocessed_resolution} resolution.'
      )

  def filter_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Returns a dataset filtered by `config.select_patient`."""

    def keep_example(example: Mapping[str, tf.Tensor]) -> tf.Tensor:
      # Maps the filename, which has the format
      # 'CheXpert-v1.0/train/patient50625/study2/view1_frontal.jpg',
      # to `patient50625`.
      patient_str = tf.strings.split(example['name'], '/')[2]
      # Maps `patient50625` to `50625`.
      patient_id = tf.strings.regex_replace(patient_str, 'patient', '')
      patient_id = tf.strings.to_number(patient_id, tf.int32)
      return self.config.select_patient(patient_id)

    if (self.config.patient_percentile_min != 0.0
        or self.config.patient_percentile_max != 1.0):
      return ds.filter(keep_example)
    else:
      return ds

  @abc.abstractmethod
  def load_raw_data(self, shuffle_files: bool) -> tf.data.Dataset:
    """Loads a supervised chexpert data set.

    Args:
      shuffle_files: whether the file names should be shuffled before being
        read.

    Returns:
      The dataset with raw examples. Each example is expected to be a
      `base.DataInputs` instance, where image is a float tensor with values in
      [0, 1], and label is a binary tensor of shape [14].
    """
    raise NotImplementedError()


ChexpertTrainOfficialConfig = functools.partial(
    ChexpertConfig,
    num_samples=223_414,
    split_content='train',
    select_label_names=TRAIN_LABEL_NAMES,
    uncertain_label_smoothing=0.2,
)
ChexpertValidOfficialConfig = functools.partial(
    ChexpertConfig,
    num_samples=234,
    split_content='validation',
    select_label_names=EVAL_LABEL_NAMES,
)
ChexpertTestOfficialConfig = functools.partial(
    ChexpertConfig,
    num_samples=668,
    split_content='test',
    select_label_names=EVAL_LABEL_NAMES,
)
ChexpertTrainInternalConfig = functools.partial(
    ChexpertConfig,
    num_samples=178_677,
    split_content='train',
    select_label_names=TRAIN_LABEL_NAMES,
    patient_percentile_min=0.2,
    patient_percentile_max=1.0,
    uncertain_label_smoothing=0.2,
)
ChexpertValidInternalConfig = functools.partial(
    ChexpertConfig,
    num_samples=22_351,
    split_content='train',
    select_label_names=EVAL_LABEL_NAMES,
    patient_percentile_min=0.0,
    patient_percentile_max=0.1,
)
ChexpertTestInternalConfig = functools.partial(
    ChexpertConfig,
    num_samples=22_386,
    split_content='train',
    select_label_names=EVAL_LABEL_NAMES,
    patient_percentile_min=0.1,
    patient_percentile_max=0.2,
)
