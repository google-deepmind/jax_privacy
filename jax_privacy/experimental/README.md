<!-- Copyright 2026 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

This directory houses prototype APIs and features for the JAX Privacy library
that are under active development and evaluation. These APIs are considered
experimental, meaning their interfaces might change, and they may eventually
be integrated into the core jax_privacy library or potentially deprecated.

## execution_plan.py

The `execution_plan` module defines `DPExecutionPlan` objects that bundle batch
selection, clipping, and noise addition components together along with a
privacy accounting event. It currently supports BandMF execution plans, which
include standard DP-SGD as the special case `num_bands=1`.

### Example: Standard DP-SGD (BandMF with num_bands=1)

```python
import jax.numpy as jnp
import dp_accounting
from jax_privacy import batch_selection
from jax_privacy import clipping
from jax_privacy.experimental import execution_plan

# A simple per-example loss. The clipped_grad helper aggregates per-example
# gradients into a clipped sum (or mean, if normalize_by is set).
loss_fn = lambda params, batch: jnp.mean((batch - params) ** 2)
clipped_grad = clipping.clipped_grad(
    loss_fn, l2_clip_norm=1.0, normalize_by=128
)

config = execution_plan.BandMFExecutionPlanConfig.default(
    iterations=1000,
    num_bands=1,
    sampling_prob=128 / 60000,
    epsilon=2.0,
    delta=1e-6,
    partition_type=batch_selection.PartitionType.INDEPENDENT,
    accountant=dp_accounting.pld.PLDAccountant(
        dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
    ),
)

plan = config.make(clipped_grad)

# plan.batch_selection_strategy produces global batches of indices.
# plan.noise_addition_transform adds calibrated Gaussian noise.
# plan.dp_event can be composed with dp_accounting for privacy accounting.
```

## Disclaimer

APIs within the experimental/ directory are subject to change without notice.
Use them at your own risk while they are being developed and tested.
Feedback on these experimental features is welcome!

<!-- BEGIN INTERNAL ONLY -->
Note: The contents of this directory are intended to be pushed to open source.
<!-- END INTERNAL ONLY -->
