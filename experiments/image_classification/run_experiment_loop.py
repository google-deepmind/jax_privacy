# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
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

r"""Runs the experiment as a loop without a Jaxline experiment.

Usage example (see README.md for setup and details):
  python run_experiment_loop.py --config=configs/mnist.py
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import json
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jax
from image_classification import config_base
from image_classification import experiment as experiment_py
from ml_collections import config_flags
import numpy as np


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    'configs/mnist.py',
    help_string='Experiment configuration file.',
)

_SAVE_TRAINING_METRICS_PATH = flags.DEFINE_string(
    'save_training_metrics_path',
    None,
    'Path to save training metrics.',
)

_SAVE_EVAL_METRICS_PATH = flags.DEFINE_string(
    'save_eval_metrics_path',
    None,
    'Path to save training metrics.',
)

_SAVE_CONFIG_PATH = flags.DEFINE_string(
    'save_config_path',
    None,
    'Path to save configs.',
)


def _sanitize_dict(d: Mapping[Any, Any]) -> Mapping[Any, Any]:
  """Recursively sanitizes a dictionary for serialization.

  This function creates a new dictionary where all values are converted to
  primitive types (int, float, bool, str, None).  Non-primitive values
  are converted to strings.

  Args:
    d: The dictionary to sanitize.

  Returns:
    A new dictionary with all values converted to primitive types.
  """
  new_d = {}
  for k, v in d.items():
    if isinstance(v, Mapping):
      new_d[k] = _sanitize_dict(v)
    elif isinstance(v, (int, float, bool, str, type(None))):
      new_d[k] = v
    else:
      new_d[k] = str(v)
  return new_d


def train_eval(config: config_base.ExperimentConfig) -> None:
  """Runs the experiment as a loop without a Jaxline experiment."""
  # Setup.
  logging.info('[Setup] Setting up experiment with config %s', config)

  experiment = experiment_py.ImageClassificationExperiment(config)

  # Initialization.
  logging.info('[Init] Starting initialization...')
  state, step_on_host, train_data = experiment.initialize(
      jax.random.PRNGKey(134)
  )
  logging.info('[Init] Initialization complete.')

  if _SAVE_CONFIG_PATH.value:
    config_dict = _sanitize_dict(dataclasses.asdict(config))
    with open(_SAVE_CONFIG_PATH.value, 'w') as f:
      json.dump(config_dict, f)

  # Define e2e train metrics.
  train_metrics_to_log = collections.defaultdict(list)
  eval_metrics_to_log = collections.defaultdict(list)

  # Training loop.
  while step_on_host < experiment.max_num_updates:
    state, train_metrics, step_on_host = experiment.update(
        state=state,
        step_on_host=step_on_host,
        inputs_producer=functools.partial(next, train_data),
    )
    if step_on_host % 10 == 0:
      logging.info('[Train] %s', train_metrics)

    # Add per step metrics.
    for key, value in train_metrics.items():
      train_metrics_to_log[key].append(float(value))

    # Evaluation loop.
    if step_on_host % 100 == 0:
      eval_metrics = experiment.evaluate(
          state=state,
          step_on_host=step_on_host,
      )
      logging.info('[Eval] %s', eval_metrics)

      # Add per step metrics.
      for key, value in eval_metrics.items():
        eval_metrics_to_log[key].append(
            value.tolist() if isinstance(value, np.ndarray) else value
        )

  # Final evaluation.
  eval_metrics = experiment.evaluate(
      state=state,
      step_on_host=step_on_host,
  )
  logging.info('[Eval] %s', eval_metrics)

  # Add per step metrics.
  for key, value in eval_metrics.items():
    eval_metrics_to_log[key].append(
        value.tolist() if isinstance(value, np.ndarray) else value
    )

  logging.info('Training and evaluation complete.')

  if _SAVE_TRAINING_METRICS_PATH.value:
    with open(_SAVE_TRAINING_METRICS_PATH.value, 'w') as f:
      json.dump(train_metrics_to_log, f)

  if _SAVE_EVAL_METRICS_PATH.value:
    with open(_SAVE_EVAL_METRICS_PATH.value, 'w') as f:
      json.dump(eval_metrics_to_log, f)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  jaxline_config = _CONFIG.value
  experiment_config = jaxline_config.experiment_kwargs.config
  train_eval(experiment_config)


if __name__ == '__main__':
  app.run(main)
