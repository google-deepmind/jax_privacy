This directory houses prototype APIs and features for the JAX Privacy library
that are under active development and evaluation. These APIs are considered
experimental, meaning their interfaces might change, and they may eventually
be integrated into the core jax_privacy library or potentially deprecated.

## batch_selection.py

Responsible for generating global batches of indices. See definitions below
for how this relates to minibatches and microbatches.

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

config = execution_plan.BandMFExecutionPlanConfig(
    iterations=1000,
    num_bands=1,
    sampling_prob=128 / 60000,
    epsilon=2.0,
    delta=1e-6,
    partition_type=batch_selection.PartitionType.INDEPENDENT,
    neighboring_relation=dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE,
    accountant=dp_accounting.pld.PLDAccountant(
        dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
    ),
)

plan = config.make(clipped_grad)

# plan.batch_selection_strategy produces global batches of indices.
# plan.noise_addition_transform adds calibrated Gaussian noise.
# plan.dp_event can be composed with dp_accounting for privacy accounting.
```

### Definitions: Global Batch Size vs. Minibatch Size vs. Microbatch Size

**(Physical) Microbatch Size** (default = None): The minibatch will be split
up into smaller microbatches of this size, which will be sequentially
fed into the loss and gradient function using jax.lax.scan. Can reduce
memory at increased sequential computation. Can be especially useful
when the size of the batch inputs is small relative to the intermediate
model activations, as in standard language modeling tasks.
This should generally be set as large as possible.

**(Physical) Minibatch Size** (default = None): This is the number of
elements that are grouped together into a single PyTree of input arrays
before computing and clipping the per-example gradients.

**Global Batch Size**: This is the overall batch size used to compute the
gradient for a single update step. It is what matters for privacy
calculations, and should be chosen to balance utility/compute
trade-offs. The global batch may never be materialized in memory, it
will be processed in chunks of size minibatch_size and further broken
down into chunks of size microbatch_size.

All three of microbatch size, minibatch size, and global batch size, may
vary from iteration to iteration depending on the batch selection strategy.
There are several ways one might try to deal with this enumerated below.

### Approach 1: Pay for recompilation.

Our function to compute the value and clipped gradients is compiled with
respect to a fixed minibatch size and microbatch size. If different
values are encountered, we must recompile the function for the new static
shapes. Often the compilation cost is significantly longer than the
training step time, although this cost is typically amortized over many
training iterations. Recompilation by itself is not a viable approach
unless coupled with a strategy to ensure that the number of different
shapes is relatively small compared to the number of training iterations.

### Approach 2: Pay for padding

While the "physical" minibatch size must be fixed to avoid recompilation
costs, it may consist of "real" and "padding" batch elements, where the
padding batch elements will contribute 0s to the aggregate gradient.
Our function for processing minibatch gradients can ensure that the
contributions from padding elements are zeroed out.

The global batch size can be any multiple of the minibatch size without
paying recompilation costs. Thus, the "cost" of this approach is the
unnecessary compute used to process the padding batch elements, which can
be up to O(minibatch_size). This can be significant depending on how large
the global batch size is relative to the minibatch size.

### Approach 3: Pay for padding + early stopping

We can reduce this cost further from the minibatch size to the microbatch
size, by incorporating dynamic early-stopping once a microbatch with all
padding elements are encountered. Thus, even though the shapes of the
inputs to our compiled function are static, the amount of compute the
function does is dynamic and depends on which batch elements are labeled
as padding examples. This offers a strict improvement over Approach 2, at
the expense of a more complex implementation.

### Approach 4: Truncation + new privacy analysis

In all three approaches above, the global batch size can be any multiple
of the microbatch size without paying for recompilation costs. If we also
require (or desire) a fixed global batch size, then there may be some
chance that the batch size we need is larger than the fixed batch size we
are constrained by, which is problematic. This can be handled by randomly
removing batch elements to reduce to the desired global batch size. This
random dropping of elements requires careful care when doing the privacy
analysis. The cost of this approach is the slack in the privacy analysis
needed to handle truncation, the gap between expected and physical batch
sizes to ensure this occurs with low probability, and relatedly the
processing of padding examples when the sampled batch size is less than
the physical global batch size.

### Approach 5: Hybrid recompilation + padding

Approaches 1, 3, and 4 all have their merits, and the best approach in
terms of compute utilization may vary from setting to setting. These
approaches represent extremes, and a hybrid approach that operates
between them can potentially offer better compute utilization. For
instance, if we allow up to K recompilations for small K, and doing so
greatly reduces the amount of padding batch elements we have to process,
this can be a worthwhile trade-off. Choosing the value of K and the
minibatch/microbatch sizes needed to minimize this cost is the main
challenge to solve here.

## Disclaimer

APIs within the experimental/ directory are subject to change without notice.
Use them at your own risk while they are being developed and tested.
Feedback on these experimental features is welcome!

<!-- BEGIN INTERNAL ONLY -->
Note: The contents of this directory are intended to be pushed to open source.
<!-- END INTERNAL ONLY -->
