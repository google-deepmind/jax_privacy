.. Copyright 2026 DeepMind Technologies Limited.
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. _keras_api:

###########
 Keras API
###########

`keras` module provides API that you can use to train your model with
differential privacy in Keras. Currently, only DP-SGD algorithm is
supported. It works the following way: first you create a keras model,
then you wrap it with `make_private` function passing essential
parameters for DP training. `make_private` preserves the same interface
that a usual Keras model has, so use it as you would use a regular Keras
model. For example, call `fit` to train it and get a DP model. Refer to
the API Reference section to see the meaning of each DP parameter. The
example below shows that.

.. _keras_api_example:

***************
 Example Usage
***************

This section demonstrates how to integrate the Keras API into a typical
Keras training workflow.

The example below uses standard fixed-size batches and sets
``sampling_method=SamplingMethod.FIXED_BATCH_SIZE`` so the privacy accountant
matches the actual training loop.

If you instead want the wrapper to resample random-access array inputs with
Poisson sampling inside ``fit()``, enable ``poisson_sampling_in_fit=True``. In
that mode the wrapper uses Poisson accounting automatically.

For dataset or generator inputs, the wrapper cannot infer the sampling
semantics automatically, so ``sampling_method`` must be set explicitly when
``poisson_sampling_in_fit`` is disabled. Generator-like inputs whose length
cannot be inferred also need an explicit ``steps_per_epoch`` so the wrapper can
bound the privacy budget before training starts.

When ``gradient_accumulation_steps > 1``, ``train_steps`` counts optimizer
updates rather than physical minibatches. In practice, this means you should
divide the total number of minibatches your training loop will execute by
``gradient_accumulation_steps`` and round down.

``validation_split`` is not supported for DP Keras training. Create the
training/validation split explicitly so ``train_size`` matches the exact number
of training examples seen by the privacy accountant.

.. literalinclude:: ../examples/keras_api_example.py
   :language: python
   :linenos:
   :caption: examples/keras_api_example.py
   :start-after: [START example]
   :end-before: [END example]

***************
 API Reference
***************

.. automodule:: jax_privacy.keras_api
   :no-members:
   :no-undoc-members:

.. autoclass:: jax_privacy.keras_api.DPKerasConfig
   :members:
   :show-inheritance:

.. autofunction:: jax_privacy.keras_api.make_private
