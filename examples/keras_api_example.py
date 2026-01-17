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

"""Example of using Keras API with DP-SGD on MNIST.

This is the fork of
https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
from absl import app
from jax_privacy import keras_api  # pylint: disable=g-import-not-at-top
import keras
from keras import layers
import numpy as np

num_classes = 10
input_shape = (28, 28, 1)


def get_model():
  return keras.Sequential([
      keras.Input(shape=input_shape),
      layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dropout(0.5),
      layers.Dense(num_classes, activation="softmax"),
  ])


def load_data():
  """Loads the MNIST data and returns the train and test sets."""
  # Load the data and split it between train and test sets
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255
  # Make sure images have shape (28, 28, 1)
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)
  # convert class vectors to "one hot encoding"
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)


def main(_):
  # Marker to insert the main part of the example into ReadTheDocs.
  # [START example]
  (x_train, y_train), (x_test, y_test) = load_data()
  model = get_model()

  epsilon = 1.1
  delta = 1e-5
  batch_size = 256
  epochs = 5
  train_size = len(x_train)
  dp = True
  clipping_norm = 1.0

  if dp:
    steps_per_epoch = train_size // batch_size
    dp_params = keras_api.DPKerasConfig(
        epsilon=epsilon,
        delta=delta,
        clipping_norm=clipping_norm,
        batch_size=batch_size,
        train_steps=epochs * steps_per_epoch,
        train_size=train_size,
        seed=0,
        gradient_accumulation_steps=1,
    )
    # Compile the model before making it private to initialize optimizer state.
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    # Run a single step to initialize metrics.
    model.train_on_batch(x_train[:batch_size], y_train[:batch_size])

    model = keras_api.make_private(model, dp_params)
    print(
        f"DP training:{epsilon=} {delta=} {clipping_norm=} {batch_size=} "
        f" {epochs=} {train_size=}"
    )

    data_source = keras_api.create_poisson_data_source(
        x_train, y_train, dp_params=dp_params
    )

    # Manual training loop
    state = (
        [v.value for v in model.trainable_variables],
        [v.value for v in model.non_trainable_variables],
        [v.value for v in model.optimizer.variables],
        [v.value for m in model.metrics for v in m.variables],
    )

    for epoch in range(epochs):
      for _ in range(steps_per_epoch):
        batch = next(data_source)
        logs, state = model.train_step(state, batch)
      print(f"Epoch {epoch + 1}/{epochs} - logs: {logs}")

  else:
    print("Non-DP training")
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )
  # [END example]
  print("DP: expected train accuracy: ~96%, val accuracy: ~92%")
  print("Non-DP: expected train accuracy: ~98%, val accuracy: ~98%")


if __name__ == "__main__":
  app.run(main)
