# Copyright 2026 DeepMind Technologies Limited.
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

import collections.abc
import os

os.environ["KERAS_BACKEND"] = "jax"
# pylint: disable=g-import-not-at-top, wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
from jax_privacy import keras_api
import keras
import numpy as np
import tensorflow as tf
# pylint: enable=g-import-not-at-top, wrong-import-position


class DictPyDataset(keras.utils.PyDataset):
  """A PyDataset that yields batches of dictionary inputs and targets.

  This is useful for testing functional models that take dictionary inputs
  (like Gemma) using Keras' PyDataset API.
  """

  def __init__(
      self, x_dict: dict[str, np.ndarray], y: np.ndarray, batch_size: int
  ):
    super().__init__()
    self.x_dict = x_dict
    self.y = y
    self.batch_size = batch_size

  def __len__(self) -> int:
    first_key = next(iter(self.x_dict))
    return (
        len(self.x_dict[first_key]) + self.batch_size - 1
    ) // self.batch_size

  def __getitem__(self, idx: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    low = idx * self.batch_size
    high = min(low + self.batch_size, len(self.y))
    batch_x = {k: v[low:high] for k, v in self.x_dict.items()}
    return (batch_x, self.y[low:high])


class StandardPyDataset(keras.utils.PyDataset):
  """A PyDataset that yields batches of dense inputs and targets."""

  def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int) -> None:
    super().__init__()
    self.x = x
    self.y = y
    self.batch_size = batch_size

  def __len__(self) -> int:
    return (len(self.y) + self.batch_size - 1) // self.batch_size

  def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
    low = idx * self.batch_size
    high = min(low + self.batch_size, len(self.y))
    return self.x[low:high], self.y[low:high]


def _to_tf_dataset(
    x: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
  """Converts numpy arrays to a tf.data.Dataset."""
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


def _to_generator(
    x: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    batch_size: int,
) -> collections.abc.Generator[tuple[np.ndarray, np.ndarray], None, None]:
  """Converts numpy arrays to an infinite generator."""

  def gen():
    while True:
      for i in range(0, len(y), batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]

  return gen()


def _to_generator_from_dict(
    x: dict[str, np.ndarray],
    y: np.ndarray,
    batch_size: int,
) -> collections.abc.Generator[
    tuple[dict[str, np.ndarray], np.ndarray], None, None
]:
  """Converts numpy arrays (in dictionary) to an infinite generator."""

  def gen():
    while True:
      for i in range(0, len(y), batch_size):
        yield {k: v[i : i + batch_size] for k, v in x.items()}, y[
            i : i + batch_size
        ]

  return gen()


def _to_py_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> keras.utils.PyDataset:
  """Converts numpy arrays to a Keras PyDataset."""
  return StandardPyDataset(x, y, batch_size)


def _to_py_dataset_from_dict(
    x: dict[str, np.ndarray],
    y: np.ndarray,
    batch_size: int,
) -> keras.utils.PyDataset:
  """Converts numpy arrays (in dictionary) to a Keras PyDataset."""
  return DictPyDataset(x, y, batch_size)


class KerasApiE2ETest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="numpy", dataset_type="numpy"),
      dict(testcase_name="tf_dataset", dataset_type="tf_dataset"),
      dict(testcase_name="generator", dataset_type="generator"),
      dict(testcase_name="py_dataset", dataset_type="py_dataset"),
  )
  def test_dp_fit_regression(self, dataset_type: str) -> None:
    """Verifies DP regression fit across different dataset types.

    Input data: 32 samples of 4 features. `y = 2x_0 + 0.5x_1`.
    Expectation: Model should learn this linear relationship and reduce MSE
    significantly.

    Args:
      dataset_type: The type of dataset to use for training (numpy, tf_dataset,
        generator, py_dataset).
    """
    np.random.seed(42)
    train_size = 32
    batch_size = 8
    epochs = 20
    num_features = 4

    inputs = keras.Input(shape=(num_features,), dtype="float32")
    outputs = keras.layers.Dense(1)(inputs)
    model_raw = keras.Model(inputs=inputs, outputs=outputs)
    loss = "mse"
    metrics = ["mse"]

    x_np = np.random.uniform(0, 1, (train_size, num_features)).astype("float32")
    y_np = (
        (2.0 * x_np[:, 0] + 0.5 * x_np[:, 1]).reshape(-1, 1).astype("float32")
    )

    x_train, y_train = x_np, y_np
    fit_kwargs = {"batch_size": batch_size}

    if dataset_type == "tf_dataset":
      x_train = _to_tf_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}
    elif dataset_type == "generator":
      x_train = _to_generator(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {"steps_per_epoch": train_size // batch_size}
    elif dataset_type == "py_dataset":
      x_train = _to_py_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}

    dp_params = keras_api.DPKerasConfig(
        epsilon=10.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=epochs * (train_size // batch_size),
        train_size=train_size,
    )

    model = keras_api.make_private(model_raw, dp_params)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=metrics,
    )

    history = model.fit(x_train, y_train, epochs=epochs, **fit_kwargs)

    self.assertIsNotNone(history.history)
    self.assertIn("loss", history.history)
    self.assertLess(history.history["loss"][-1], history.history["loss"][0])
    self.assertLess(history.history["loss"][-1], 0.2)

  @parameterized.named_parameters(
      dict(testcase_name="numpy", dataset_type="numpy"),
      dict(testcase_name="tf_dataset", dataset_type="tf_dataset"),
      dict(testcase_name="generator", dataset_type="generator"),
      dict(testcase_name="py_dataset", dataset_type="py_dataset"),
  )
  def test_dp_fit_binary_classification(self, dataset_type: str) -> None:
    """Verifies DP binary classification fit.

    Input data: 32 samples of 4 features, where `y = (x[:, 0] > 0.5)`.
    Expectation: Model should learn this simple linear separation and achieve
    accuracy > 0.6.

    Args:
      dataset_type: The type of dataset to use for training (numpy, tf_dataset,
        generator, py_dataset).
    """
    np.random.seed(42)
    train_size = 32
    batch_size = 8
    epochs = 20
    num_features = 4

    inputs = keras.Input(shape=(num_features,), dtype="float32")
    outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
    model_raw = keras.Model(inputs=inputs, outputs=outputs)
    loss = "binary_crossentropy"
    metrics = ["accuracy"]

    x_np = np.random.uniform(0, 1, (train_size, num_features)).astype("float32")
    y_np = (x_np[:, 0] > 0.5).astype("float32").reshape(-1, 1)

    x_train, y_train = x_np, y_np
    fit_kwargs = {"batch_size": batch_size}

    if dataset_type == "tf_dataset":
      x_train = _to_tf_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}
    elif dataset_type == "generator":
      x_train = _to_generator(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {"steps_per_epoch": train_size // batch_size}
    elif dataset_type == "py_dataset":
      x_train = _to_py_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}

    dp_params = keras_api.DPKerasConfig(
        epsilon=10.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=epochs * (train_size // batch_size),
        train_size=train_size,
    )

    model = keras_api.make_private(model_raw, dp_params)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=metrics,
    )

    history = model.fit(x_train, y_train, epochs=epochs, **fit_kwargs)

    self.assertIsNotNone(history.history)
    self.assertIn("loss", history.history)
    self.assertLess(history.history["loss"][-1], history.history["loss"][0])
    accuracy_key = "accuracy"
    self.assertGreater(history.history[accuracy_key][-1], 0.6)

  @parameterized.named_parameters(
      dict(testcase_name="numpy", dataset_type="numpy"),
      dict(testcase_name="tf_dataset", dataset_type="tf_dataset"),
      dict(testcase_name="generator", dataset_type="generator"),
      dict(testcase_name="py_dataset", dataset_type="py_dataset"),
  )
  def test_dp_fit_multilabel_classification(self, dataset_type: str) -> None:
    """Verifies DP multi-label classification fit across all dataset types.

    This task consists of 3 independent binary classifications sharing inputs.
    Input data: 32 samples of 4 features. Three binary labels, each independent
    and depends on the corresponding feature: `y_k = (x_k > 0.5)`.
    Expectation: Model should learn this relationship and achieve accuracy >
    0.45.

    Args:
      dataset_type: The type of dataset to use for training (numpy, tf_dataset,
        generator, py_dataset).
    """
    np.random.seed(42)
    train_size = 32
    batch_size = 8
    epochs = 20
    num_features = 4
    num_classes = 3

    inputs = keras.Input(shape=(num_features,), dtype="float32")
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(inputs)
    model_raw = keras.Model(inputs=inputs, outputs=outputs)
    loss = "binary_crossentropy"
    metrics = ["accuracy"]

    x_np = np.random.uniform(0, 1, (train_size, num_features)).astype("float32")
    y_np = np.zeros((train_size, num_classes), dtype="float32")
    y_np[:, 0] = (x_np[:, 0] > 0.5).astype("float32")
    y_np[:, 1] = (x_np[:, 1] > 0.5).astype("float32")
    y_np[:, 2] = (x_np[:, 2] > 0.5).astype("float32")

    x_train, y_train = x_np, y_np
    fit_kwargs = {"batch_size": batch_size}

    if dataset_type == "tf_dataset":
      x_train = _to_tf_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}
    elif dataset_type == "generator":
      x_train = _to_generator(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {"steps_per_epoch": train_size // batch_size}
    elif dataset_type == "py_dataset":
      x_train = _to_py_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}

    dp_params = keras_api.DPKerasConfig(
        epsilon=10.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=epochs * (train_size // batch_size),
        train_size=train_size,
    )

    model = keras_api.make_private(model_raw, dp_params)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=metrics,
    )

    history = model.fit(x_train, y_train, epochs=epochs, **fit_kwargs)

    self.assertIsNotNone(history.history)
    self.assertIn("loss", history.history)
    self.assertLess(history.history["loss"][-1], history.history["loss"][0])
    accuracy_key = "accuracy"
    self.assertGreater(history.history[accuracy_key][-1], 0.45)

  @parameterized.named_parameters(
      dict(testcase_name="numpy", dataset_type="numpy"),
      dict(testcase_name="tf_dataset", dataset_type="tf_dataset"),
      dict(testcase_name="generator", dataset_type="generator"),
      dict(testcase_name="py_dataset", dataset_type="py_dataset"),
  )
  def test_dp_fit_multiclass_classification(self, dataset_type: str) -> None:
    """Verifies DP multi-class classification fit across all dataset types.

    Task: Classify into 3 classes based on argmax of features.
    Input data: 32 samples of 4 features.
    Expectation: Model should learn this rule and achieve accuracy > 0.6.

    Args:
      dataset_type: The type of dataset to use for training (numpy, tf_dataset,
        generator, py_dataset).
    """
    np.random.seed(42)
    train_size = 32
    batch_size = 8
    epochs = 20
    num_features = 4
    num_classes = 3

    inputs = keras.Input(shape=(num_features,), dtype="float32")
    outputs = keras.layers.Dense(num_classes)(inputs)
    model_raw = keras.Model(inputs=inputs, outputs=outputs)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["sparse_categorical_accuracy"]

    x_np = np.random.uniform(0, 1, (train_size, num_features)).astype("float32")
    y_np = (
        (np.argmax(x_np, axis=1) % num_classes).astype("int32").reshape(-1, 1)
    )

    x_train, y_train = x_np, y_np
    fit_kwargs = {"batch_size": batch_size}

    if dataset_type == "tf_dataset":
      x_train = _to_tf_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}
    elif dataset_type == "generator":
      x_train = _to_generator(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {"steps_per_epoch": train_size // batch_size}
    elif dataset_type == "py_dataset":
      x_train = _to_py_dataset(x_np, y_np, batch_size)
      y_train = None
      fit_kwargs = {}

    dp_params = keras_api.DPKerasConfig(
        epsilon=10.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=epochs * (train_size // batch_size),
        train_size=train_size,
    )

    model = keras_api.make_private(model_raw, dp_params)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=metrics,
    )

    history = model.fit(x_train, y_train, epochs=epochs, **fit_kwargs)

    self.assertIsNotNone(history.history)
    self.assertIn("loss", history.history)
    self.assertLess(history.history["loss"][-1], history.history["loss"][0])
    accuracy_key = "sparse_categorical_accuracy"
    self.assertGreater(history.history[accuracy_key][-1], 0.6)

  @parameterized.named_parameters(
      dict(testcase_name="numpy_dict", dataset_type="numpy_dict"),
      dict(testcase_name="tf_dataset_dict", dataset_type="tf_dataset_dict"),
      dict(testcase_name="generator_dict", dataset_type="generator_dict"),
      dict(testcase_name="py_dataset_dict", dataset_type="py_dataset_dict"),
  )
  def test_dp_fit_seq2seq(self, dataset_type: str) -> None:
    """Verifies DP seq2seq fit across different dataset types (including dict).

    Input data: 32 samples of sequences of length 5. Vocabulary size 100.
    Targets are token IDs modulo `num_classes`.
    Expectation: Model should learn this simple lookup relationship and achieve
    accuracy > 0.6.

    Args:
      dataset_type: The type of dataset to use for training (numpy, tf_dataset,
        generator, py_dataset).
    """
    np.random.seed(42)
    train_size = 32
    batch_size = 8
    epochs = 20
    num_classes = 3
    sequence_length = 5
    vocab_size = 100

    inputs_dict = {
        "token_ids": keras.Input(shape=(sequence_length,), dtype="int32")
    }
    x = keras.layers.Embedding(vocab_size, 16)(inputs_dict["token_ids"])
    outputs = keras.layers.Dense(num_classes)(x)
    model_raw = keras.Model(inputs=inputs_dict, outputs=outputs)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["sparse_categorical_accuracy"]

    x_np = np.random.randint(
        0, vocab_size, (train_size, sequence_length)
    ).astype("int32")
    y_np = (x_np % num_classes).astype("int32")

    x_train, y_train = {"token_ids": x_np}, y_np
    fit_kwargs = {"batch_size": batch_size}

    if dataset_type == "tf_dataset_dict":
      x_train = _to_tf_dataset({"token_ids": x_np}, y_np, batch_size)
      y_train = None
      fit_kwargs = {}
    elif dataset_type == "generator_dict":
      x_train = _to_generator_from_dict({"token_ids": x_np}, y_np, batch_size)
      y_train = None
      fit_kwargs = {"steps_per_epoch": train_size // batch_size}
    elif dataset_type == "py_dataset_dict":
      x_train = _to_py_dataset_from_dict({"token_ids": x_np}, y_np, batch_size)
      y_train = None
      fit_kwargs = {}

    dp_params = keras_api.DPKerasConfig(
        epsilon=10.0,
        delta=1e-5,
        clipping_norm=1.0,
        batch_size=batch_size,
        gradient_accumulation_steps=1,
        train_steps=epochs * (train_size // batch_size),
        train_size=train_size,
    )

    model = keras_api.make_private(model_raw, dp_params)

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=metrics,
    )

    history = model.fit(x_train, y_train, epochs=epochs, **fit_kwargs)

    self.assertIsNotNone(history.history)
    self.assertIn("loss", history.history)
    self.assertLess(history.history["loss"][-1], history.history["loss"][0])
    accuracy_key = "sparse_categorical_accuracy"
    self.assertGreater(history.history[accuracy_key][-1], 0.6)


if __name__ == "__main__":
  absltest.main()
