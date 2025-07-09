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

"""API for adding DP-SGD to a Keras model."""

import dataclasses
import enum
import functools
import inspect
import types
import typing
import jax
import jax.numpy as jnp
from jax_privacy.accounting import accountants
from jax_privacy.accounting import analysis
from jax_privacy.accounting import calibrate
from jax_privacy.dp_sgd import grad_clipping as jp_grad_clipping
from jax_privacy.dp_sgd import gradients as jp_gradients
from jax_privacy.dp_sgd import typing as jp_typing
import keras
from keras.src import backend
from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
import numpy as np


class ClippingMethod(enum.Enum):
  """The type of clipping norm."""

  # Individual gradients are computed in parallel with jax.vmap.
  SPEED_OPTIMIZED = 1

  # Individual gradients are computed one by one.
  MEMORY_OPTIMIZED = 2


@dataclasses.dataclass(frozen=True)
class DPKerasConfig:
  """Parameters for adding DP-SGD to a Keras model.

  Attributes:
      epsilon: The epsilon that defines the differential-privacy budget. It
        should be in (0; +infinity) range. 0 means perfect privacy guarantee
        (not achievable in practice due to infinite noise), +infinity means no
        privacy guarantee. A commonly used value is ln(3) (smaller value more
        noise). You should set this value before training and only based on the
        privacy guarantees you have to achieve. You should not increase the
        epsilon only because of poor model performance.
      delta: The delta that defines the differential-privacy budget. The value
        of it means the probablility of full disclosure, no-privacy. It should
        be in (0, 1] and be as small as possible (e.g. 1e-5, smaller value more
        noise). You should set this value before training and only based on the
        privacy guarantees you have to achieve. You should not increase the
        delta only because of poor model performance.
      clipping_norm: The clipping norm for the gradients. TODO: how to choose
        it?
      batch_size: The batch size for the training.
      gradient_accumulation_steps: The number of gradient accumulation steps.
        This is the number of batches to accumulate before adding noise and
        performing an optimizer step. 1 means that there is no gradient
        accumulation, each optimizer step is performed after a single batch.
        This parameter defines the effective batch size = (physical) batch_size
        * gradient_accumulation_steps, i.e. the real accumulated batch size used
        for the model update. Usually DP training provides better accuracy with
        larger effective batch size, therefore it is recommended to set
        gradient_accumulation_steps to a value larger than 1. In many cases, you
        won't be able to set the physical batch size to a large enough value due
        to memory constraints, therefore gradient accumulation technique is very
        useful during DP training.
      train_steps: The number of training steps (optimizer update steps). If you
        try to train the model for more steps, it will fail. If you train by
        epochs, then it is epochs * (train_size // batch_size). If you train
        while the dataset iterator is not over then it is the length of the
        dataset iterator.
      train_size: The number of training examples in the dataset. If you repeat
        the examples in your dataset iterator, it should be the number of
        training examples in the original dataset before repeating.
      noise_multiplier: The noise multiplier for the gradients. If None
        (recommended), the noise multiplier will be automatically calculated
        based on epsilon, delta, effective_batch_size, train_steps and
        train_size. The noise added to the average of gradients per total batch
        is normal with mean 0 and stddev = noise_multiplier * clipping_norm /
        effective_batch_size.
      rescale_to_unit_norm: Whether to rescale the gradients to unit norm.
        Simplifies learning-rate tuning, see https://arxiv.org/abs/2204.13650.
      seed: The seed for the random number generator. If None, a random seed is
        used. It must be an int64. Useful for reproducibility.
      clipping_method: To optimize memory or speed when computing clipped
        gradients. Defaults to SPEED_OPTIMIZED, usually no need to change it.
  """

  epsilon: float
  delta: float
  clipping_norm: float
  batch_size: int
  gradient_accumulation_steps: int
  train_steps: int
  train_size: int
  noise_multiplier: float | None = None
  rescale_to_unit_norm: bool = True
  clipping_method: ClippingMethod = ClippingMethod.SPEED_OPTIMIZED
  seed: int | None = None

  _accountant = analysis.DpsgdTrainingAccountant(
      dp_accountant_config=accountants.PldAccountantConfig(
          # Smaller values result in higher precision but slower computation.
          value_discretization_interval=1e-3
      )
  )

  @property
  def effective_batch_size(self) -> int:
    """The effective batch size which is used for the model update.

    It equals to batch_size * gradient_accumulation_steps.
    """
    return self.batch_size * self.gradient_accumulation_steps

  def update_with_calibrated_noise_multiplier(self) -> 'DPKerasConfig':
    """Calculates the noise multiplier for the given DP training parameters.

    Returns:
      A copy (new instance) of DPKerasConfig with the noise multiplier set to
      the calibrated value.
    """
    print(
        f'Calculating noise multiplier for: {self.epsilon=},'
        f' {self.delta=}, {self.effective_batch_size=}, {self.train_steps=},'
        f' {self.train_size=}. This might take a few minutes.'
    )
    calculated_noise_multiplier = calibrate.calibrate_noise_multiplier(
        target_epsilon=self.epsilon,
        target_delta=self.delta,
        accountant=self._accountant,
        batch_sizes=self.effective_batch_size,
        num_updates=self.train_steps,
        num_samples=self.train_size,
    )
    print(
        'Finished calculating noise multiplier:'
        f' {calculated_noise_multiplier=}.'
    )
    return dataclasses.replace(
        self, noise_multiplier=calculated_noise_multiplier
    )

  def __post_init__(self):
    self._validate_params()

  def _validate_params(self):
    """Validates the parameters for DP-SGD training."""
    if self.epsilon <= 0:
      raise ValueError(f'Epsilon {self.epsilon} must be positive.')
    if self.delta <= 0:
      raise ValueError(f'Delta {self.delta} must be positive.')
    if self.clipping_norm <= 0:
      raise ValueError(f'Clipping norm {self.clipping_norm} must be positive.')
    if self.batch_size <= 0:
      raise ValueError(f'Batch size {self.batch_size} must be positive.')
    if self.train_steps <= 0:
      raise ValueError(f'Train steps {self.train_steps} must be positive.')
    if self.train_size <= 0:
      raise ValueError(f'Train size {self.train_size} must be positive.')
    if self.gradient_accumulation_steps <= 0:
      raise ValueError(
          f'Gradient accumulation steps {self.gradient_accumulation_steps} must'
          ' be positive.'
      )
    if self.noise_multiplier is not None:
      if self.noise_multiplier <= 0:
        raise ValueError(
            f'Noise multiplier {self.noise_multiplier} must be positive.'
        )
      try:
        resulting_epsilon = self._accountant.compute_epsilon(
            self.train_steps,
            analysis.DpParams(
                noise_multipliers=self.noise_multiplier,
                batch_size=self.batch_size,
                num_samples=self.train_size,
                delta=self.delta,
            ),
        )
      except ValueError as e:
        raise ValueError(
            'Value error occured while calculating epsilon based on the'
            f' provided {self.noise_multiplier=}. Maybe the noise multiplier is'
            f' too small? Original error: {e}'
        ) from e
      tolerance = 1e-1
      if resulting_epsilon > self.epsilon + tolerance:
        raise ValueError(
            f'Provided {self.noise_multiplier=} will lead to privacy'
            ' budget exceed because the resulting epsilon will be'
            f' {resulting_epsilon=} > target_epsilon={self.epsilon}. You need'
            ' to set a greater noise multiplier (greater epsilon means more'
            ' noise and more budget). Or you can leave noise multiplier unset'
            ' at all and let the API to automatically calculate the optimal'
            ' one.'
        )


def make_private(model: keras.Model, params: DPKerasConfig):
  """Adds DP-SGD training to a Keras model without modifying its API.

  This function modifies `model` in-place by adding attributes and replaces
  methods (e.g. it replaces train_step) and returns the modified model. The API
  of the model is not modified, i.e. you can use it as a usual Keras model.

  Args:
    model: The Keras model to add DP-SGD training to.
    params: The parameters for DP-SGD training.

  Returns:
    The Keras model with overloaded methods for DP-SGD training.
  """
  _validate_model(model)

  # Adding DP-SGD to the model works in the following way:
  # 1. We add attributes to the model:
  #    - _dp_params: The parameters for DP-SGD training.
  #    - _gradient_computer: The gradient computer for DP-SGD training.
  #    - _rng: The random number generator for DP-SGD training.
  # 2. We replace the model.fit with a new method that performs validation for
  #    DP-SGD training and calls the original fit method.
  # 3. We replace the model.train_step method with a new method that performs
  #    DP-SGD training. This method differs from the original, only in the
  #    gradient computation (clipped and noised).
  # 4. We replace the model._update_metrics_variables method with a new method
  #    that updates the metrics variables for DP-SGD training.

  _add_dp_sgd_attributes(model, params)
  model.fit = types.MethodType(
      _create_fit_fn_with_validation(model.fit, params), model
  )
  model.train_step = types.MethodType(_dp_train_step, model)
  if not hasattr(model, '_update_metrics_variables'):
    # _update_metrics_variables was extracted from train_step recently in
    # https://github.com/keras-team/keras/pull/20805/. Since in our train_step
    # we use it, we need to add it if it's not present. In the future, when
    # will stop support old versions of Keras, we can remove this.
    model._update_metrics_variables = types.MethodType(  # pylint: disable=protected-access
        _update_metrics_variables, model
    )
  return model


def _validate_model(model: keras.Model):
  if not isinstance(model, keras.Model):
    raise ValueError(f'Model {model} is not a Keras model.')
  if not isinstance(model, backend.jax.trainer.JAXTrainer):
    raise ValueError(f'Model {model} must use Jax backend.')
  # TODO: Add validation that the model does not contain layers
  # that are not compatible with DP-SGD, e.g. batch norm.


def _validate_optimizer(model: keras.Model, params: DPKerasConfig):
  optimizer_gradient_accumulation_steps = (
      model.optimizer.gradient_accumulation_steps or 1
  )
  dp_params_gradient_accumulation_steps = params.gradient_accumulation_steps
  if (
      optimizer_gradient_accumulation_steps
      != dp_params_gradient_accumulation_steps
  ):
    raise ValueError(
        'optimizer.gradient_accumulation_steps ='
        f' {optimizer_gradient_accumulation_steps} must be equal to'
        ' DPKerasConfig.gradient_accumulation_steps ='
        f' {dp_params_gradient_accumulation_steps}.'
    )


def _add_dp_sgd_attributes(model: keras.Model, params: DPKerasConfig) -> None:
  """Adds DP-SGD training attributes to the Keras model."""
  model._dp_params = params  # pylint: disable=protected-access
  model._gradient_computer = _get_gradient_computer(params)  # pylint: disable=protected-access
  prng_key = _get_random_int64() if params.seed is None else params.seed
  model.add_weight(
      name='_rng',
      shape=(2,),
      dtype='uint32',
      initializer=lambda shape, dtype: jax.random.PRNGKey(prng_key),
      trainable=False,
  )
  model.add_weight(
      name='_optimizer_steps',
      shape=(1,),
      initializer=lambda shape, dtype: jnp.zeros(shape, dtype=dtype),
      dtype='uint32',
      trainable=False,
  )


def _get_gradient_computer(
    params: DPKerasConfig,
) -> jp_gradients.DpsgdGradientComputer:
  """Creates the gradient computer for DP-SGD training."""
  match params.clipping_method:
    case ClippingMethod.SPEED_OPTIMIZED:
      clip_method = jp_grad_clipping.VECTORIZED
    case ClippingMethod.MEMORY_OPTIMIZED:
      clip_method = jp_grad_clipping.UNROLLED
    case _:
      raise ValueError(f'Unknown clipping method: {params.clipping_method}')
  noise_multiplier = (
      params.noise_multiplier
      if params.noise_multiplier is not None
      else params.update_with_calibrated_noise_multiplier().noise_multiplier
  )
  # We use the additivity of Gaussian random variables to calculate the noise
  # multiplier per batch.
  noise_multiplier_per_batch = noise_multiplier / np.sqrt(
      params.gradient_accumulation_steps
  )
  return jp_gradients.DpsgdGradientComputer(
      clipping_norm=params.clipping_norm,
      noise_multiplier=noise_multiplier_per_batch,
      rescale_to_unit_norm=params.rescale_to_unit_norm,
      per_example_grad_method=clip_method,
  )


def _create_fit_fn_with_validation(
    original_fit_fn: typing.Callable[..., typing.Any],
    params: DPKerasConfig,
):
  """Creates a fit function with validation for DP-SGD training.

   It validates that:
   1. DP parameters are aligned with the fit() arguments (e.g. batch sizes are
   equal).
   2. The number of optimizer steps that will be performed during fitting will
   not exceed the maximum number of optimizer steps.

  Args:
    original_fit_fn: The original fit function of the Keras model.
    params: The parameters for DP-SGD training.

  Returns:
    The fit function with same signature as original_fit_fn but with validation
    for DP-SGD training.
  """

  @functools.wraps(original_fit_fn)
  def fit_fn_with_validation(
      self,
      *args,
      **kwargs,
  ):
    _validate_optimizer(self, self._dp_params)  # pylint: disable=protected-access
    fit_signature = inspect.signature(original_fit_fn)

    # batch_size is not set explicitely in the fit() call if the input dataset
    # is already batched. In this case, we assume that the batch sizes are
    # aligned and use the batch size from the DP parameters. We will check that
    # the batch sizes are aligned in the train_step function.
    batch_size = (
        _get_param(fit_signature, 'batch_size', *args, **kwargs)
        or params.batch_size
    )
    # Default values are set according to the Keras documentation.
    epochs = _get_param(fit_signature, 'epochs', *args, **kwargs) or 1
    initial_epoch = (
        _get_param(fit_signature, 'initial_epoch', *args, **kwargs) or 0
    )
    steps_per_epoch = _get_param(
        fit_signature, 'steps_per_epoch', *args, **kwargs
    )

    # Note accessing self._dp_params is safe because it's added in
    # _add_dp_sgd_attributes, but requires disabling pylint because this
    # function is not a method within a class.
    _check_dp_params_aligned_with_fit_args(
        self._dp_params,  # pylint: disable=protected-access
        batch_size,
    )

    performed_optimizer_steps = (
        _get_non_trainable_weight('_optimizer_steps', self).numpy().item()
    )
    optimizer_steps_to_perform = _calculate_optimizer_steps_to_perform_in_fit(
        self._dp_params.train_size,  # pylint: disable=protected-access
        batch_size,
        epochs,
        initial_epoch,
        steps_per_epoch,
    )
    if (
        performed_optimizer_steps + optimizer_steps_to_perform
        > self._dp_params.train_steps  # pylint: disable=protected-access
    ):
      raise RuntimeError(
          'fit() cannot be performed because you will run out of privacy'
          ' budget. Currently, you have already performed'
          f' {performed_optimizer_steps} optimizer training steps and you are'
          f' trying to perform {optimizer_steps_to_perform} more. However, you'
          f' can perform in total only {self._dp_params.train_steps} training'  # pylint: disable=protected-access
          ' steps (optimizer updates). If you fit() the model with current'
          ' parameters, training steps will exceed the maximum number of'
          f' training steps: {performed_optimizer_steps=} +'
          f' {optimizer_steps_to_perform=} ='
          f' {performed_optimizer_steps + optimizer_steps_to_perform} >'
          f' total_train_steps={self._dp_params.train_steps}.'  # pylint: disable=protected-access
      )
    return original_fit_fn(
        *args,
        **kwargs,
    )

  return fit_fn_with_validation


def _check_dp_params_aligned_with_fit_args(
    dp_params: DPKerasConfig,
    batch_size: int,
):
  """Checks that the DP parameters are aligned with the fit() arguments."""
  if dp_params.batch_size != batch_size:
    raise ValueError(
        'The batch size in the DP parameters is not equal to the batch size'
        f' passed to fit(): {dp_params.batch_size=} != {batch_size=}. Please'
        ' make sure that the batch size in the DP parameters is equal to the'
        ' batch size passed to fit().'
    )


def _dp_train_step(self, state, data):
  """Performs a single training step.

  This function replaces Keras model train_step (that's why it has self arg).
  It differs from model.train_step only in the gradient computation (clipped
  and noised).

  Args:
    self: The Keras model.
    state: The state of the model. As in model.train_step.
    data: The data for the model. As in model.train_step.

  Returns:
    logs: The logs for the training step. As in model.train_step.
    state: The new state of the model. As in model.train_step.
  """
  (
      trainable_variables,
      _,
      optimizer_variables,
      _,
  ) = state
  x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

  dp_batch_size = self._dp_params.batch_size  # pylint: disable=protected-access
  actual_batch_size = jax.tree_util.tree_leaves(x)[0].shape[0]
  if dp_batch_size != actual_batch_size:
    # it is ok to throw an exception even though we are in a jit function
    # because the check is based on the static values, i.e. they won't
    # change between invocations, and if the condition is violated, it will
    # always fail during the tracing (first invocation) of this function.
    raise ValueError(
        'The batch size in the DP parameters is not equal to the batch size of'
        f' the actual data: {dp_batch_size=} !='
        f' actual_batch_size={actual_batch_size}. Please make sure that the'
        ' batch size in the DP parameters is equal to the batch size of the'
        ' data you supplied in the fit() call.'
    )  # pylint: disable=protected-access

  (_, aux), grads = _noised_clipped_grads(
      self.compute_loss_and_updates,
      self._dp_params,  # pylint: disable=protected-access
      self._gradient_computer,  # pylint: disable=protected-access
      state,
      data,
  )
  (
      unscaled_loss,  # unscaled means sum of losses, not divided by batch size
      y_pred,
      non_trainable_variables,
      metrics_variables,
  ) = aux

  (
      trainable_variables,
      optimizer_variables,
  ) = self.optimizer.stateless_apply(
      optimizer_variables, grads, trainable_variables
  )
  # TODO: access it and update it by name.
  non_trainable_variables[1] = non_trainable_variables[1] + 1

  logs, metrics_variables = self._update_metrics_variables(  # pylint: disable=protected-access
      metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
  )

  state = self._enforce_jax_state_sharding(  # pylint: disable=protected-access
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables,
  )
  return logs, state


def _noised_clipped_grads(
    compute_loss_and_updates_fn,
    dp_params: DPKerasConfig,
    gradient_computer: jp_gradients.GradientComputer,
    state,
    data,
):
  """Computes noised and clipped gradients.

  Args:
    compute_loss_and_updates_fn: The function that computes the loss and updates
      for the given state and data.
    dp_params: The parameters for DP-SGD training.
    gradient_computer: The gradient computer for DP-SGD training.
    state: The state of the model.
    data: The data for the model.

  Returns:
    (loss, aux), grads
  """
  (
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables,
  ) = state
  # TODO: access it and update it by name.
  rng = non_trainable_variables[0]
  rng_grads, rng_next = jax.random.split(rng)
  x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

  inputs = {'x': x, 'y': y, 'sample_weight': sample_weight}

  # TODO: use rng argument for dropout
  def loss_fn(params, network_state, unused_rng, inputs):
    loss, aux = compute_loss_and_updates_fn(
        params,
        non_trainable_variables,
        metrics_variables,
        inputs['x'],
        inputs['y'],
        inputs['sample_weight'],
        training=True,
        optimizer_variables=optimizer_variables,
    )

    (
        unscaled_loss,
        y_pred,
        new_non_trainable_variables,
        new_metrics_variables,
    ) = aux

    jp_metrics = jp_typing.Metrics(
        per_example={'y_pred': y_pred},
        scalars_avg={
            'loss': loss,
            'unscaled_loss': unscaled_loss,
            'non_trainable_variables': new_non_trainable_variables,
            'metrics_variables': new_metrics_variables,
        },
    )
    return loss, (network_state, jp_metrics)

  (loss, (_, jp_metrics)), grads = gradient_computer.loss_and_clipped_gradients(
      loss_fn=loss_fn,
      params=trainable_variables,
      network_state={},
      rng_per_local_microbatch=jax.random.PRNGKey(0),  # not used RNG
      inputs=inputs,
  )
  unscaled_loss = jp_metrics.scalars_avg['unscaled_loss']
  y_pred = jp_metrics.per_example['y_pred']
  non_trainable_variables = [rng_next] + non_trainable_variables[1:]
  metrics_variables = jp_metrics.scalars_avg['metrics_variables']

  grads, _, _ = gradient_computer.add_noise_to_grads(
      grads, rng_grads, dp_params.batch_size, None  # full batch size
  )
  aux = (unscaled_loss, y_pred, non_trainable_variables, metrics_variables)

  return (loss, aux), grads


# This is copy-paste from
# https://github.com/keras-team/keras/blob/6b4a4dfaa26c14d3071a489e43453917f7b42e30/keras/src/backend/jax/trainer.py#L88
def _update_metrics_variables(  # pylint: disable=too-many-positional-arguments
    self, metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
):
  """Updates the metrics variables."""
  with backend.StatelessScope(
      state_mapping=list(zip(self.metrics_variables, metrics_variables))
  ) as scope:
    self._loss_tracker.update_state(  # pylint: disable=protected-access
        unscaled_loss, sample_weight=tree.flatten(x)[0].shape[0]
    )
    logs = self.compute_metrics(x, y, y_pred, sample_weight)

  new_metrics_variables = []
  for ref_v in self.metrics_variables:
    new_v = scope.get_current_value(ref_v)
    if new_v is None:
      new_v = ref_v.value
    new_metrics_variables.append(new_v)
  return logs, new_metrics_variables


def _get_param(
    method_signature: inspect.Signature,
    param_name: str,
    *args,
    **kwargs,
):
  """Returns the value of the parameter in the method call.

  This function is used to get the value of the parameter in the method
  call. It checks if the parameter is passed in *args or **kwargs and
  returns the value of the parameter if it is. Otherwise, it returns the default
  value of the parameter if it is explicitly present in the signature or throws
  exception otherwise.

  Args:
    method_signature: The signature of the method.
    param_name: The name of the parameter to get the value of.
    *args: The positional arguments passed to the method.
    **kwargs: The keyword arguments passed to the method.
  """
  parameters = method_signature.parameters
  param_index = None
  try:
    param_index = list(parameters.keys()).index(param_name)
  except ValueError:
    # The parameter name is not present in the signature, but it might be passed
    # in **kwargs.
    pass
  # Check if the parameter is passed in *args
  if param_index is not None and param_index < len(args):
    return args[param_index]
  # Check if the parameter is passed in **kwargs
  if param_name in kwargs:
    return kwargs[param_name]
  return parameters[param_name].default if param_name in parameters else None


def _get_non_trainable_weight(weight_name: str, model: keras.Model):
  """Returns the non-trainable weight with the given name."""
  return next(w for w in model.non_trainable_weights if w.name == weight_name)


def _calculate_optimizer_steps_to_perform_in_fit(
    train_size: int,
    batch_size: int,
    epochs: int,
    initial_epoch: int,
    steps_per_epoch: int,
) -> int:
  """Returns the number of optimizer steps that will be performed by fit."""
  epochs_to_perform = epochs - initial_epoch
  steps_per_epoch = steps_per_epoch or (train_size // batch_size)
  return steps_per_epoch * epochs_to_perform


def _get_random_int64() -> np.int64:
  int64_info = np.iinfo(np.int64)
  return np.random.randint(
      low=int64_info.min, high=int64_info.max, dtype=np.int64
  )
