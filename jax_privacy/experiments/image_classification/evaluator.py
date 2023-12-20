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

"""The evaluator computes evaluation metrics.

For image classification we define an additional class function that iterates
over batches and aggregates metrics.
"""

import collections
from typing import Any, Iterator, Mapping, Sequence
import chex
from einshape import jax_einshape as einshape
import jax
from jax_privacy.experiments import image_data as data
from jax_privacy.experiments.image_classification import metrics as metrics_module
from jax_privacy.src.training import devices
from jax_privacy.src.training import evaluator as evaluator_py
from jax_privacy.src.training import forward
from jax_privacy.src.training import updater


class ImageClassificationEvaluator(evaluator_py.AbstractEvaluator):
  """Defines and applies the update, potentially in parallel across devices."""

  def __init__(
      self,
      *,
      forward_fn: forward.ForwardFn,
      rng: chex.PRNGKey,
      device_layout: devices.DeviceLayout = devices.DeviceLayout(),
      eval_auc: bool = False,
      eval_disparity: bool = False,
      class_names: Sequence[str],
  ):

    super().__init__(
        forward_fn=forward_fn,
        rng=rng,
        device_layout=device_layout,
    )

    self._eval_auc = eval_auc
    self._eval_disparity = eval_disparity
    self._class_names = class_names

  def evaluate_dataset(
      self,
      updater_state: updater.UpdaterState,
      ds_iterator: Iterator[data.DataInputs],
  ) -> Mapping[str, Any]:
    """Evaluates a full dataset in an iterative fashion.

    Args:
      updater_state: Updater state.
      ds_iterator: Data iterator of objects of shape (num_hosts,
        num_devices_per_host, batch_size_per_device, *individual_shape).

    Returns:
      metrics.
    """
    avg_metrics = collections.defaultdict(metrics_module.Avg)

    num_samples = 0
    host_id = jax.process_index()

    if self._eval_auc or self._eval_disparity:
      all_params = ['last', *updater_state.params_avg]
      logits_by_params = {
          k: metrics_module.ArrayConcatenater() for k in all_params}
      labels_by_params = {
          k: metrics_module.ArrayConcatenater() for k in all_params}

    for inputs in ds_iterator:

      num_hosts, num_devices_per_host, batch_size_per_device, *_ = (
          inputs.image.shape)
      batch_size = num_hosts * num_devices_per_host * batch_size_per_device
      num_samples += batch_size
      local_inputs = jax.tree_map(lambda x: x[host_id], inputs)

      metrics_by_params = self._evaluate_batch(
          updater_state=updater_state, inputs=local_inputs)

      for params_name, metrics in metrics_by_params.items():
        # Update the accumulated average for each metric.
        for metric_name, val in metrics.scalars.items():
          avg_metrics[f'{metric_name}_{params_name}'].update(val, n=batch_size)

        if self._eval_auc or self._eval_disparity:
          # Returned logits are across all devices (flattened over all hosts).
          logits = einshape(
              '(hd)bk->(hdb)k', metrics.per_example['logits'],
              h=num_hosts, d=num_devices_per_host, b=batch_size_per_device)
          logits_by_params[params_name].append(logits)
          # Use labels across all devices on all hosts.
          labels = einshape(
              'hdbk->(hdb)k', inputs.label,
              h=num_hosts, d=num_devices_per_host, b=batch_size_per_device)
          labels_by_params[params_name].append(labels)

    metrics = {k: v.avg for k, v in avg_metrics.items()}
    metrics['num_samples'] = num_samples

    # Compute AUC and / or disparity with concatenated logits and labels.
    if self._eval_auc:
      for params_name in logits_by_params:
        auc_metrics = metrics_module.avg_and_per_class_auc(
            logits=logits_by_params[params_name].asarray(),
            labels=labels_by_params[params_name].asarray(),
            class_names=self._class_names,
        )
        for metric_name, val in auc_metrics.items():
          metrics[f'{metric_name}_{params_name}'] = val

    if self._eval_disparity:
      for params_name in logits_by_params:
        disparity_metrics = metrics_module.per_class_disparity(
            logits=logits_by_params[params_name].asarray(),
            labels=labels_by_params[params_name].asarray(),
            class_names=self._class_names,
        )
        for metric_name, val in disparity_metrics.items():
          metrics[f'{metric_name}_{params_name}'] = val

    return metrics
