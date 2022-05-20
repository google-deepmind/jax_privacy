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

"""Training an NF-ResNet-50 on ImageNet with (8.0, 8e-7)-DP."""

import haiku.initializers as hk_init
import jax.numpy as jnp
from jax_privacy.src.training.image_classification import config_base
from jax_privacy.src.training.image_classification import data
from ml_collections import config_dict as configdict


@config_base.wrap_get_config
def get_config(config):
  """Experiment config."""

  config.experiment_kwargs = configdict.ConfigDict(
      dict(
          config=dict(
              num_updates=71589,
              optimizer=dict(
                  name='sgd',
                  lr=dict(
                      init_value=4.0,
                      decay_schedule_name=None,
                      decay_schedule_kwargs=None,
                      relative_schedule_kwargs=None,
                  ),
                  kwargs=dict(
                  ),
              ),
              model=dict(
                  model_type='nf_resnet',
                  model_kwargs=dict(
                      variant='ResNet50',
                      drop_rate=None,  # dropout-rate
                      fc_init=hk_init.RandomNormal(0.01, 0),
                      skipinit_gain=jnp.ones,
                  ),
                  restore=dict(
                      path=None,
                      params_key=None,
                      network_state_key=None,
                      layer_to_reset=None,
                  ),
              ),
              training=dict(
                  batch_size=dict(
                      init_value=16384,
                      per_device_per_step=32,
                      scale_schedule=None,  # example: {'2000': 8, '4000': 16},
                  ),
                  weight_decay=0.0,  # L-2 regularization,
                  train_only_layer=None,  # None
                  dp=dict(
                      target_delta=8e-7,
                      clipping_norm=1.0,  # float('inf') or None to deactivate
                      stop_training_at_epsilon=8.0,  # None,
                      rescale_to_unit_norm=True,
                      noise=dict(
                          std_relative=2.5,  # noise multiplier
                          ),
                      # Set the following flag to auto-tune one of:
                      # * 'batch_size'
                      # * 'std_relative'
                      # * 'stop_training_at_epsilon'
                      # * 'num_updates'
                      # Set to `None` to deactivate auto-tunning
                      auto_tune=None,
                      ),
                  logging=dict(
                      grad_clipping=True,
                      grad_alignment=False,
                      snr_global=True,  # signal-to-noise ratio across layers
                      snr_per_layer=False,  # signal-to-noise ratio per layer
                  ),
              ),
              averaging=dict(
                  ema=dict(
                      coefficient=0.99999,
                      start_step=0,
                  ),
                  polyak=dict(
                      start_step=0,
                  ),
              ),
              data=dict(
                  dataset=data.get_dataset(
                      name='imagenet',
                      train_split='train',  # 'train' or 'train_valid'
                      eval_split='valid',  # 'valid' or 'test'
                  ),
                  image_size=dict(
                      train=(224, 224),
                      eval=(224, 224),
                  ),
                  augmult=4,  # implements arxiv.org/abs/2105.13343
                  random_flip=True,
                  random_crop=True,
                  ),
              evaluation=dict(
                  batch_size=100,
              ))))

  return config
