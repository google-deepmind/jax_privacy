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

r"""Runs the experiment as a loop without a Jaxline experiment.

Usage example:
  python run_experiment_loop.py --config=configs/mnist.py
"""

import functools
from typing import Sequence

from absl import app
from absl import logging
import jax
from jax_privacy.experiments.image_classification import config_base
from jax_privacy.experiments.image_classification import experiment as experiment_py
from ml_collections import config_flags


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    'configs/mnist.py',
    help_string='Experiment configuration file.',
)


def train_eval(config: config_base.ExperimentConfig) -> None:
  """Runs the experiment as a loop without a Jaxline experiment."""
  # Setup.
  logging.info('[Setup] Setting up experiment with config %s', config)

  experiment = experiment_py.ImageClassificationExperiment(config)

  # Initialization.
  logging.info('[Init] Starting initialization...')
  state, step_on_host, train_data = experiment.initialize(
      jax.random.PRNGKey(134))
  logging.info('[Init] Initialization complete.')

  # Training loop.
  while step_on_host < experiment.max_num_updates:
    state, train_metrics, step_on_host = (
        experiment.update(
            state=state,
            step_on_host=step_on_host,
            inputs_producer=functools.partial(next, train_data),
        )
    )
    if step_on_host % 10 == 0:
      logging.info('[Train] %s', train_metrics)

    # Evaluation loop.
    if step_on_host % 100 == 0:
      eval_metrics = experiment.evaluate(
          state=state,
          step_on_host=step_on_host,
      )
      logging.info('[Eval] %s', eval_metrics)
  logging.info('Training and evaluation complete.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  jaxline_config = _CONFIG.value
  experiment_config = jaxline_config.experiment_kwargs.config
  train_eval(experiment_config)


if __name__ == '__main__':
  app.run(main)
