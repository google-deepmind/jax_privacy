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

.. literalinclude:: ../../examples/keras_api_example.py
   :language: python
   :linenos:
   :caption: examples/keras_api_example.py
   :start-after: [START example]
   :end-before: [END example]

***************
 API Reference
***************

.. automodule:: jax_privacy.keras.keras_api
   :no-members:
   :no-undoc-members:

.. autoclass:: jax_privacy.keras.keras_api.DPKerasConfig
   :members:
   :show-inheritance:

.. autofunction:: jax_privacy.keras.keras_api.make_private
