################
 JAX & Flax API
################

`accounting` and `dp_sgd` modules provide the API for raw JAX and Flax
Linen. Flax NNX is not supported yet.

`accounting` module contains logic for computing the hyperparameters to
ensure the privacy budget is not exceeded. `dp_sgd` module contains
logic for calculating the gradients and adding noise to them.

The main steps of using the API are:

#. Choose the accounting algorithm (RDP or PLD).
#. Create the accountant.
#. Use the accountant to compute the "unfixed" DP hyperparameter.
#. Create the gradient computer.
#. Create the loss function wrapper that will be called to calculate
   loss per single example.
#. Use the gradient computer in the traing step to calculate the clipped
   gradients and add noise to them.
#. Apply the clipped noisy gradients to the model.

**Comments about step 1:**

Using PLD accounting algorithm is recommended because it utilizes the
budget better. It takes a bit longer,but still reasonably fast.

**Comments about step 3:**

Two of the following hyperparameters has to be fixed beforehand:

#. Noise multiplier (noise stddev = noise_multiplier * l2_clip_norm).
#. Number of updates.
#. Physical batch size.

Then the third one can be inferred from the other fixed two. Computation
of the non-fixed hyperparameter is called "calibration". The
`calibration.py` provides convenient functions to do that. See the API
specification below for details. Of course, you also have to supply
other core DP hyperparameters, like epsilon and delta.

Usually batch size and number of updates are fixed and noise multiplier
is calibrated.

**Comments about step 4:**

`DpsgdGradientComputer` is the core of the API. The main params to
supply there is the maximum L2 norm to which the gradients are clipped
and noise multiplier.

**Comments about step 5:**

It is important to understand that the gradients have to be calculated
and clipped per single example. This is one of the main differences
between the usually SGD training and DP SGD training. Therefore you have
to create a wrapper around your loss function that will be used to
calculate the loss per single example (see the example below).

In JAX Privacy the loss function wrapper has to be of the specific
signature: it has to accept model parameters, network state, random
generator and input data (tuple of x and y) and return the loss and
tuple of new network state and the metrics. For simple cases, network
state and random generator are not used and can be ignored. Network
state is just a python dictionary, therefore you can pass an empty
dictionary. In the returned metrics object you can add metrics and
specify how to aggregate them: stack it, average it or sum it. For
example, you can stack logits to calculate accuracy later in the code.
There are such strict requirements to the loss function wrapper because
it is not clear how to aggregate a metric or a state over batch
dimension: just stack it or average it or sum it, etc. See the API
reference below for exact types and signatures. This part might be
simplified in the future versions of JAX Privacy.

To make it more clear, here is the example using raw JAX that
illustrates the aforementioned steps. There is also a Jupyter notebook
that shows the usage of this API with Flax Linen:
:doc:`flax_linen_example`.

***************
 Example Usage
***************

.. literalinclude:: ../../examples/jax_api_example.py
   :language: python
   :linenos:
   :caption: examples/jax_api_example.py
   :start-after: [START example]
   :end-before: [END example]

***************
 API Reference
***************

The main API functions can be found in the API reference below.

Here is the list of available budget accounting algorithms (configs):

.. autoclass:: jax_privacy.accounting.accountants.RdpAccountantConfig
   :members:
   :show-inheritance:

.. autoclass:: jax_privacy.accounting.accountants.PldAccountantConfig
   :members:
   :show-inheritance:

PLD utilizes the budget better allowing to add less noise, but takes
more time, however still reasonably fast, therefore it is recommended to
use it.

Once you have chosen the accountant, you can create DP-SGD accountant
for training passing the accountant config from the previous step as a
constructor argument:

.. autoclass:: jax_privacy.accounting.analysis.DpsgdTrainingAccountant
   :members:
   :show-inheritance:

Then you can use this accountant to compute one of the DP
hyperparameters you have not fixed:

.. autofunction:: jax_privacy.accounting.calibrate.calibrate_noise_multiplier

.. autofunction:: jax_privacy.accounting.calibrate.calibrate_num_updates

.. autofunction:: jax_privacy.accounting.calibrate.calibrate_batch_size

Then you can create `DpSgdGradientComputer` to calculate the clipped
gradients and add noise to them.

.. autoclass:: jax_privacy.dp_sgd.gradients.GradientComputer
   :members:
   :show-inheritance:

.. autoclass:: jax_privacy.dp_sgd.gradients.DpsgdGradientComputer
   :members:
   :show-inheritance:
