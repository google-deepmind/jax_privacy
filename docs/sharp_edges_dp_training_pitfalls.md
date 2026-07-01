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

# Common Pitfalls in DP Training

*Design decisions JAX Privacy makes to prevent them*

Building a correct differentially private training pipeline is hard. A single
misplaced operation — dividing by the wrong constant, using the wrong batch
sampling strategy, or miscalibrating noise — can silently break the formal
privacy guarantee while producing models that *look* fine.

JAX Privacy is designed to make correctness the path of least resistance.
The design decisions described on this page are based on years of experience
building and integrating DP training pipelines — both our own implementations
and integrations with internal and external libraries. Each one addresses a
specific pitfall we have encountered in practice.

---

## Three Levels of Assurance

JAX Privacy provides building blocks at three levels of abstraction, each
with a different level of DP assurance:

1.  **End-to-end training loops** ([Keras API](keras_api.rst),
    [`training`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.training.html)):
    These consume a `DPExecutionPlan` and write the entire training loop for
    you — batch selection, gradient computation, noise addition, parameter
    updates, and privacy accounting. The resulting training satisfies the
    stated DP guarantee unconditionally. You do not need to reason about
    how the components interact; the library handles it.

2.  **[`DPExecutionPlan`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.execution_plan.DPExecutionPlan.html)**:
    This bundles batch selection, clipped gradient computation, noise
    addition, and the corresponding `DpEvent` into a single cohesive object.
    When you use the plan's components as documented to write your own
    training loop, the resulting loop inherits the stated DP guarantee
    *by construction*.

3.  **Low-level building blocks** ([`clipped_grad`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.clipping.clipped_grad.html),
    [`noise_addition`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.noise_addition.html),
    [`batch_selection`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.html),
    [`accounting`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.accounting.html)):
    These give you maximum flexibility. Each individual component is designed
    so that you cannot configure it in a way that breaks its own formal
    guarantees — if you can, that is a bug. Importantly, these formal
    guarantees are not DP guarantees: individual components do not satisfy
    DP by themselves. They are properties like sensitivity bounds and
    per-example isolation that serve as the building blocks for *proving*
    DP of the higher-level compositions. This is why they are carefully
    documented — to enable rigorous reasoning about the end-to-end
    guarantee when components are composed. The risk at this level is in
    *composition*: wiring components together incorrectly (e.g., calibrating
    noise to the wrong sensitivity, or using an accounting method that does
    not match the batch selection strategy). It is not until you couple
    components together that you can reason about a complete DP mechanism,
    and getting that coupling right is your responsibility.

```{tip}
**Design principle:** You should not be able to configure any individual
JAX Privacy utility in a way that breaks its stated guarantees. If you can,
that is a bug. We enforce this at the API level, even when it sacrifices
flexibility.
```

---

## Common Pitfalls at a Glance

| Pitfall | How JAX Privacy Handles It |
| :--------- | :------------------------- |
| [Division by batch size](#division-by-batch-size) | Computes a sum, not a mean; optional `normalize_by` reflected in sensitivity |
| [Sensitivity alignment](#sensitivity-alignment) | Returned callable exposes `.sensitivity()` — calibrate noise against it |
| [Accounting and batch selection mismatch](#accounting-and-batch-selection-mismatch) | `DPExecutionPlan` couples these by construction |
| [Public vs. private metadata](#public-vs-private-metadata) | Dataclass fields are public config; sensitive values are method arguments |
| [Neighboring relation clarity](#neighboring-relation-clarity) | Explicit `NeighboringRelation` enum; `.sensitivity()` parameterized by it |
| [Zero-sized batches and non-finite gradients](#zero-sized-batches-and-non-finite-gradients) | Robust edge-case handling that preserves DP, not just utility |
| [Gradient accumulation](#gradient-accumulation) | No manual accumulation needed — on-device microbatching instead |
| [Flat, auditable design](#flat-auditable-design) | No inter-component dependencies; each module stands alone |
| [Framework integration](#framework-integration) | High-level training loops recommended; loss-function-level API for advanced users |
| [Auxiliary information leakage](#auxiliary-information) | Per-example returns; aggregation is the caller's responsibility |
| [Randomness and RNG injection](#randomness-and-rng-injection) | Explicit RNG parameters; supports cryptographically secure sources |
| [Floating point robustness](#floating-point-robustness) | Opt-in discrete Gaussian mechanism with integer-domain clipping |
| [Foot-gun APIs (injectable `vmap`)](#the-vmap-design-decision) | Deliberately restricted (see below) |
| [Batch normalization and cross-example operations](#batch-normalization-and-cross-example-operations) | `vmap` isolates examples automatically; batch norm becomes per-example norm |
| [Opaque code](#opaque-code) | Everything open source; designed for auditability |

---

(division-by-batch-size)=
### Division by Batch Size

**The pitfall.** Under Poisson sampling, the batch size is a *random variable*
that depends on which examples were selected — making it technically sensitive
information. In standard (non-private) training, the loss function typically
computes a mean over the batch: `loss = sum(losses) / batch_size`. But in DP
training, dividing by the actual (random) batch size changes the sensitivity
of the query and can leak information about the batch composition.

**How JAX Privacy handles it.** The `clipped_grad` function computes a *sum*
of clipped per-example gradients, not a mean. This avoids any dependence on
the random batch size. If you want to recover a mean gradient (e.g., for
compatibility with standard optimizers), you can pass a `normalize_by` value
— typically the *expected* batch size, which is a public, deterministic
quantity:

```python
# User writes a per-example loss -- no batch-size division needed.
def loss_fn(params, x, y):
    logits = model.apply(params, x)
    return cross_entropy(logits, y)

# JAX Privacy computes a sum of clipped gradients, optionally normalized.
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    normalize_by=expected_batch_size,  # Public, deterministic quantity.
)
```

Crucially, the `normalize_by` value is reflected in the sensitivity reported
by the returned callable's `.sensitivity()` method, so downstream noise
calibration stays correct automatically.

---

(sensitivity-alignment)=
### Sensitivity Alignment

**The pitfall.** There are many ways to configure gradient clipping: global
clipping, per-layer clipping, rescaling to unit norm, integer grid rounding
for the discrete Gaussian mechanism, and more. Each configuration produces a
different sensitivity for the clipped gradient function. If you calibrate your
noise multiplier based on an *assumed* sensitivity that does not match the
*actual* sensitivity of your clipping configuration, the privacy guarantee is
invalidated.

**How JAX Privacy handles it.** The callable returned by `clipped_grad` (and
`clipped_fun`) exposes a `.sensitivity()` method that reports the exact L2
sensitivity of the clipped output, accounting for all configuration options:

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    rescale_to_unit_norm=True,
    normalize_by=expected_batch_size,
)

# Query the sensitivity -- it accounts for clipping, rescaling, and normalization.
sensitivity = grad_fn.sensitivity()
```

The `DPExecutionPlan` uses this method internally to calibrate noise, ensuring
that the noise multiplier and the clipping configuration are always aligned.
If you are using the lower-level building blocks directly, we recommend always
calibrating noise against the `.sensitivity()` of the clipped gradient
callable rather than computing the sensitivity by hand.

---

(accounting-and-batch-selection-mismatch)=
### Accounting and Batch Selection Mismatch

**The pitfall.** Privacy accounting makes assumptions about how batches are
formed (e.g., Poisson sampling with a specific probability). If the actual
batch selection strategy does not match these assumptions, the computed privacy
guarantee is meaningless. This is one of the most common and most dangerous
mistakes in DP training, because the training loop runs without errors and the
resulting model looks normal.

**How JAX Privacy handles it.** The `DPExecutionPlan` bundles the batch
selection strategy, clipped gradient computation, noise addition, and the
corresponding `DpEvent` into a single object. The `DpEvent` is derived from
the same parameters that configure the batch selection and noise addition,
ensuring they are consistent by construction:

```python
config = jax_privacy.BandMFConfig.default(
    num_bands=1,
    iterations=1000,
    expected_participations=100,
).calibrate(epsilon=1.0, delta=1e-5)

plan = config.make()

# These are guaranteed to be consistent:
plan.batch_selection_strategy  # Poisson sampling with matching probability
plan.noise_addition_transform  # Noise calibrated to matching sensitivity
plan.dp_event                  # DpEvent derived from the same parameters
```

---

(public-vs-private-metadata)=
### Public vs. Private Metadata

**The pitfall.** DP training pipelines handle a mix of public quantities
(hyperparameters, expected batch size, number of iterations) and sensitive
quantities (the actual dataset size, which examples were selected, the
random seed). If these are not clearly separated, it is easy to accidentally
pass sensitive information to a component that treats it as public — for
example, logging the actual batch size or including the dataset size in a
configuration object that gets serialized.

**How JAX Privacy handles it.** JAX Privacy uses a deliberate structural
separation. In the batch selection API, for example, the
`BatchSelectionStrategy` is a frozen dataclass whose *fields* are all public
configuration (sampling probability, number of iterations, cycle length).
Sensitive, data-dependent values — the number of examples and the random
number generator — are only consumed as *arguments* to the `batch_iterator`
method:

```python
# Fields are public configuration -- safe to log, serialize, pass around.
strategy = jax_privacy.CyclicPoissonSampling(
    sampling_prob=0.01,
    iterations=1000,
    cycle_length=1,
)

# Sensitive values are only consumed at call time.
for batch_indices in strategy.batch_iterator(
    num_examples=len(dataset),  # Sensitive under add-or-remove-one DP.
    rng=rng,                    # Privacy-critical randomness.
):
    ...
```

This means the configuration object can be freely passed around, logged, and
serialized without revealing any sensitive information. The sensitive values
are only introduced at the point where they are needed.

```{note}
**Note on neighboring relations:** Under some neighboring relations (e.g.,
zero-out or replace-one), the number of examples is public information. In
these cases, batch selection strategies may define it as a field on the
dataclass itself. The API design makes this distinction explicit rather than
leaving it to implicit convention.
```

---

(neighboring-relation-clarity)=
### Neighboring Relation Clarity

**The pitfall.** Different DP neighboring relations (add-or-remove-one,
replace-one, zero-out) imply different sensitivity bounds and different
assumptions about what information is public. If the neighboring relation is
not explicitly stated and programmatically enforced, it is easy for different
parts of the pipeline to make incompatible assumptions — for example, the
clipping code assumes replace-one (sensitivity = 2C) while the accounting
code assumes add-or-remove-one (sensitivity = C), silently doubling the
actual privacy loss.

A subtler version of this problem arises from *parameterization choices*. In
research code, it is common to parameterize a mechanism in terms of the batch
size or the expected batch size. But doing so makes an implicit assumption
that you know the number of records in the dataset — and that automatically
rules out add-or-remove-one as a viable neighboring relation, because under
add-or-remove-one, the dataset size is sensitive. This kind of implicit
assumption is easy to miss in code but would be caught immediately in a
careful paper.

**How JAX Privacy handles it.** JAX Privacy supports three neighboring
relations — `ADD_OR_REMOVE_ONE`, `REPLACE_ONE`, and `REPLACE_SPECIAL`
(zero-out) — and makes the choice explicit and first-class throughout the
API. We generally recommend `ADD_OR_REMOVE_ONE`, but JAX Privacy supports
mechanisms in the literature that work under different relations; in
particular, some variants of matrix factorization assume a zero-out model.

The neighboring relation is an explicit parameter at every level:

- The `.sensitivity()` method on `clipped_grad`'s returned callable is
  parameterized by a `NeighboringRelation` enum, so you always know exactly
  which neighboring relation your sensitivity bound corresponds to:

  ```python
  from dp_accounting import NeighboringRelation

  grad_fn = jax_privacy.clipped_grad(loss_fn, l2_clip_norm=1.0)

  # Explicit -- no ambiguity about what "sensitivity" means.
  s_add_remove = grad_fn.sensitivity(NeighboringRelation.ADD_OR_REMOVE_ONE)
  s_replace = grad_fn.sensitivity(NeighboringRelation.REPLACE_ONE)
  # s_replace == 2 * s_add_remove
  ```

- The `DPExecutionPlan` stores the `neighboring_relation` as a field, and
  uses it consistently across batch selection, noise calibration, and
  privacy accounting.

JAX Privacy also takes the philosophy of parameterizing mechanisms directly —
the way you would in a paper — rather than in terms of derived quantities
like batch size that carry implicit assumptions about what is public. This
connects to the [public vs. private metadata](#public-vs-private-metadata)
separation: by parameterizing in terms of fundamental quantities (sampling
probability, number of iterations) rather than derived ones (batch size),
the API avoids baking in assumptions about the neighboring relation.

---

(zero-sized-batches-and-non-finite-gradients)=
### Zero-Sized Batches and Non-Finite Gradients

**The pitfall.** These are two edge cases that break DP, not just utility:

- **Zero-sized batches.** With Poisson sampling, it is possible (though
  unlikely) for a batch to have zero examples. If a training step *fails* on a
  zero-sized batch but *succeeds* on a size-one batch, the success or failure
  of the step itself leaks information about whether a particular example was
  in the dataset — violating DP.

- **Non-finite gradients.** NaN or infinity values in per-example gradients
  (e.g., from numerical instability) can propagate through aggregation and
  corrupt the clipped sum. If one example produces NaN and another does not,
  the presence or absence of NaN in the output leaks per-example information.

Both cases require the training step to produce a well-defined, bounded output
*regardless* of the input, to preserve the formal DP guarantee.

**How JAX Privacy handles it.** The `clipped_grad` and `clipped_fun` functions
handle both cases correctly, regardless of how you structure your batches:

- **Zero-sized batches work directly.** `clipped_grad` produces the correct
  result even when passed a batch with zero examples — no special handling
  or padding is required. You can also pad batches to a fixed size and use
  the `is_padding_example` argument to mark padding examples, whose
  contributions are zeroed out before aggregation. Either approach works;
  JAX Privacy produces the correct result in both cases.

- **Non-finite gradients are handled by default.** When `nan_safe=True` (the
  default), per-example outputs with non-finite L2 norms are zeroed out
  before aggregation. This ensures that numerical instability in any single
  example cannot corrupt the aggregate or leak information.

See also [Variable Batch Sizes](sharp_edges_variable_batch_sizes) for
strategies to handle variable batch sizes efficiently.

---

(gradient-accumulation)=
### Gradient Accumulation

**The pitfall.** Gradient accumulation — processing the batch in smaller
chunks, accumulating the gradients, and applying a single optimizer update —
is a common technique for handling large batches in non-private training.
In DP training, it is not necessarily *wrong*, but it is tricky to get right.
For example, if you add IID noise to each chunk independently, you end up
with more total noise than necessary. If you add noise only after
accumulation, you need to ensure the accumulation preserves the sensitivity
bound. And if the noise across chunks is not truly independent (e.g., due to
PRNG reuse), the noise may be correlated or even identical across chunks,
which can silently break the privacy guarantee. Getting the interaction
between accumulation, clipping, and noise calibration correct is subtle and
error-prone.

**How JAX Privacy handles it.** JAX Privacy sidesteps this problem entirely.
The `clipped_grad` function works out of the box with not just large, but
*extremely* large batch sizes by incorporating *on-device microbatching* via
the `microbatch_size` parameter. Microbatching processes the batch in
sequential chunks using `jax.lax.scan`, but the clipping, aggregation, and
noise addition are all handled within a single function — no manual gradient
accumulation is needed.

To put "extremely large" in context: in a transformer setting, a batch
consists of B sequences of L tokens stored as `int32` values, requiring
`B × L × 4` bytes. With a sequence length of L = 1024, each sequence
occupies just 4 KB — meaning you can fit over 260,000 sequences in 1 GB of
memory for the input data alone. When training across multiple machines, this
scales proportionally. JAX Privacy handles all of this with a single
argument:

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=32,  # Process 32 examples at a time for memory savings.
)
# Works correctly with arbitrarily large batches -- no manual gradient
# accumulation, no brittle code, no error-prone noise calibration.
```

This means you never need to write manual gradient accumulation code — which
is brittle, error-prone, and difficult to audit for DP correctness. In
scenarios where microbatching alone is not sufficient, you can still implement
manual gradient accumulation — JAX Privacy will not prevent you from doing
so — but you will need to be careful about how you implement it.

---

(flat-auditable-design)=
### Flat, Auditable Design

**The pitfall.** When privacy-critical logic (clipping, noise addition,
sensitivity tracking) is spread across deeply nested abstraction layers, it
becomes difficult to audit. A bug in any layer can silently break the privacy
guarantee, and the more layers there are, the harder it is to verify
correctness.

**How JAX Privacy handles it.** JAX Privacy has a completely flat design: no
component depends on any other component. The clipping module, noise addition
module, batch selection module, and accounting module each stand alone and can
be understood, tested, and audited in isolation.

The coupling between components only happens at the higher-level API layer
(e.g., `DPExecutionPlan`), where the joint formal guarantees are explicitly
stated. This means that auditing *or contributing to* any individual
component does not require understanding the rest of the library — and
auditing the composition requires understanding only the thin integration
layer, not the internals of each component.

---

(framework-integration)=
### Framework Integration

**The pitfall.** Training frameworks are typically designed without DP in
mind. When you try to retrofit DP into such a framework, you are forced to
separate components that naturally need to be coupled in order to reason about
formal guarantees — batch selection ends up in the data pipeline, gradient
clipping in the optimizer, noise addition in a callback, and accounting in a
separate module. The result is spaghetti code where the privacy-critical
logic is spread across many files and abstraction layers, making it extremely
difficult to catch bugs.

We have experienced this firsthand. In one internal codebase built on a
training framework that was never designed for DP, it took *weeks* just to
understand how the different pieces fit together — not because the individual
components were complex, but because the framework's abstractions forced them
apart in unnatural ways. This is not an isolated experience; we have seen the
same pattern repeatedly with other framework integrations.

The framework also forces shortcuts that limit what you can do. For example,
a framework may assume that noise is added independently at each step, which
immediately rules out the correlated noise mechanisms that JAX Privacy
supports (e.g., matrix factorization). You end up constantly fighting against
the framework's assumptions to make the privacy mechanism work, and the
resulting code is brittle and hard to verify.

**How JAX Privacy handles it.** JAX Privacy operates at the pure JAX level.
The core API transforms *loss functions*, not training loops:

- `clipped_grad` transforms a loss function into a clipped-gradient function.
- `clipped_fun` transforms an arbitrary function into a clipped-output
  function.

This means you can write your training loop the way you would describe it in
a paper — and anyone can come in and say "yes, this is correct" or "no, this
is not correct," because the structure matches the mathematical description
rather than being distorted by framework abstractions.

We recommend using JAX Privacy's high-level training loops (e.g., the
[Keras API](keras_api.rst) or the `DPExecutionPlan`-based loop shown in our
[examples](examples_guide)) for the strongest guarantees. If the benefits of
a specific framework outweigh the benefits of JAX Privacy's built-in training
loops, you can still use the lower-level building blocks — JAX Privacy will
not prevent you from doing so. But based on our experience, the
framework-agnostic approach is less error-prone.

---

(auxiliary-information)=
### Auxiliary Information

**The pitfall.** DP training pipelines typically need diagnostic information —
training loss, test loss, gradient norms, and other statistics — to monitor
convergence and debug issues. Returning these quantities exactly breaks DP,
because they depend on the training data. Yet many practitioners surface them
anyway, reasoning that the diagnostics will not be released publicly. The
danger is that this becomes an undocumented, unaudited exception to the
privacy guarantee.

A previous version of JAX Privacy allowed users to define custom metrics
aggregation functions. This not only complicated the API surface, but it
also obscured the fact that these aggregated metrics are not private — users
could easily overlook the need to privatize them.

**How JAX Privacy handles it.** The `clipped_grad` function returns auxiliary
outputs (loss values, gradient norms, and user-defined auxiliary data) on a
*per-example* basis. It does not aggregate them.

This design has two benefits:

1. **Power users can privatize.** Because the auxiliary outputs are
   per-example, a sophisticated user who wants end-to-end DP — including for
   the diagnostics — can apply an appropriate DP mechanism (e.g., a DP mean,
   median, or histogram) to the per-example values.

2. **Non-private aggregation is a deliberate choice.** If a user decides to
   take a non-private mean of the per-example losses for logging, that is
   their explicit decision. JAX Privacy does not do it for them, and the
   per-example nature of the output makes it clear that aggregation is the
   caller's responsibility.

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    return_values=True,        # Per-example loss values
    return_grad_norms=True,    # Per-example gradient norms
)

clipped_grads, aux = grad_fn(params, batch)
aux.values       # shape: (batch_size,) -- per-example, not aggregated
aux.grad_norms   # shape: (batch_size,) -- per-example, not aggregated
```

```{warning}
**Limitation:** JAX Privacy does not *prevent* non-private aggregation of
auxiliary outputs. A careless user can still compute exact means or complete
histograms over per-example values without adding noise. The API makes this
a *visible, deliberate* choice rather than a hidden default, but it does
not enforce private aggregation.
```

---

(randomness-and-rng-injection)=
### Randomness and RNG Injection

**The pitfall.** Privacy-critical randomness — batch selection, noise
generation — must come from a high-quality source. If the random number
generator is improperly seeded, reused across components, or not
cryptographically secure in settings that require it, the privacy guarantee
can be weakened or broken entirely.

**How JAX Privacy handles it.** Wherever privacy-critical randomness is
needed, JAX Privacy consumes the RNG directly as an explicit parameter. For
example, the `batch_iterator` method accepts an `rng` argument:

```python
# Option 1: Pass None to use NumPy's default RNG (convenient for research).
for batch in strategy.batch_iterator(num_examples, rng=None):
    ...

# Option 2: Pass a seed for reproducibility.
for batch in strategy.batch_iterator(num_examples, rng=42):
    ...

# Option 3: Inject a cryptographically secure source for production hardening.
import secrets
secure_rng = np.random.Generator(np.random.SFC64(secrets.randbits(128)))
for batch in strategy.batch_iterator(num_examples, rng=secure_rng):
    ...
```

This design provides three levels of assurance:

- **Convenient defaults** for research and experimentation (pass `None` or a
  seed).
- **Reproducibility** for debugging (pass a fixed seed).
- **Dependency injection** of cryptographically secure RNG sources for
  production deployments that require it.

---

(floating-point-robustness)=
### Floating Point Robustness

**The pitfall.** Standard DP-SGD implementations operate in floating point
arithmetic, which introduces rounding errors. These rounding errors are
deterministic and data-dependent, meaning that in principle, an adversary
could exploit floating point non-associativity or rounding patterns to extract
information about individual training examples — even in the presence of
noise. While these attacks are largely theoretical today, they represent a
real gap between the mathematical DP guarantee (which assumes exact real
arithmetic) and what the implementation actually provides.

**How JAX Privacy handles it.** JAX Privacy's default code path uses standard
floating point arithmetic, which is sufficient for most research and
production use cases. For settings that require hardened guarantees, JAX
Privacy provides an opt-in code path that uses the **discrete Gaussian
mechanism** combined with **integer-domain clipping** via the `grid_scale`
parameter:

```python
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    grid_scale=10**9,  # Quantize gradients to an integer grid.
)
```

This code path:

1. Clips per-example gradients and quantizes them to an integer grid with
   `grid_scale` steps per `l2_clip_norm`.
2. Aggregates the quantized gradients using exact integer arithmetic.
3. Adds noise from the discrete Gaussian distribution, which is defined over
   the integers and avoids floating point rounding in the noise itself.

The result is a DP guarantee that holds *exactly* — not up to floating point
approximation. Combined with
[cryptographically secure RNG injection](#randomness-and-rng-injection), this
provides a fully hardened implementation. See the
[secure noise example](https://github.com/google-deepmind/jax_privacy/blob/main/examples/secure_noise_example.py)
for a complete working demonstration.

```{warning}
**Limitation:** The discrete Gaussian mechanism and cryptographically secure
RNG are opt-in, not the default. The default code path uses standard
floating point arithmetic and standard PRNGs (NumPy for batch selection,
JAX for noise addition), which means it is *not* robust to
floating-point-based attacks. For most research and production settings
this is acceptable, but users who need hardened guarantees must explicitly
opt in.
```

---

(the-vmap-design-decision)=
### The `vmap` Design Decision

**The pitfall.** JAX's `vmap` is central to how JAX Privacy computes
per-example gradients: it vectorizes the gradient computation across the batch
dimension. A natural API design would parameterize `clipped_grad` with a
user-injectable `vmap` function, allowing users to plug in custom variants
(e.g., `shard_map` for distributed settings).

**Why this is dangerous.** A user could plug in a function that *satisfies
the signature* of `vmap` but does not actually compute per-example results
independently. For example, a function that applies the computation to the
first example and replicates the result across the batch dimension. This
would silently break the per-example clipping guarantee: the first user's
gradient would be used for every "per-example" gradient, exposing that
individual to a much higher privacy risk while appearing to work correctly.

**How JAX Privacy handles it.** We deliberately chose *not* to parameterize
the `vmap` function. The `clipped_grad` and `clipped_fun` functions always
use `jax.vmap` internally (with an optional `spmd_axis_name` for distributed
settings). This means you cannot accidentally break per-example isolation by
plugging in a broken vectorization function.

This is a concrete example of our design principle: *correctness over
flexibility*. The small loss in configurability is worth the guarantee that
the API cannot be misconfigured in a way that silently breaks DP.

---

(batch-normalization-and-cross-example-operations)=
### Batch Normalization and Cross-Example Operations

**The pitfall.** Some computations include operations that aggregate
statistics *across* examples in a batch — most notably batch normalization,
which computes running means and variances over the batch dimension. These
cross-example operations are fundamentally incompatible with per-example DP,
because a single example's contribution affects the statistics used by every
other example.

**How JAX Privacy handles it.** Because JAX Privacy's clipping is built on
`vmap`, each example's forward and backward pass runs in complete isolation.
If your loss function includes batch normalization or other cross-example
operations, JAX Privacy will not raise an error — it will work and satisfy
the stated DP properties. But `vmap` naturally transforms "batch
normalization" into *per-example normalization*: the statistics are computed
over the spatial/channel dimensions of a single example, not across the
batch.

This means:

- **DP is not broken.** The per-example isolation is preserved by
  construction. You cannot accidentally introduce cross-example information
  leakage through the loss function.
- **Semantics change.** The operation is no longer doing what "batch
  normalization" typically means. Whether per-example normalization is
  desirable for your task is a modeling decision, not a privacy one.
- **No special handling needed.** JAX Privacy does not need to enumerate
  which operations or layer types are compatible with DP. Any JAX-traceable
  loss function that runs under `vmap` is automatically compatible.

This is a direct benefit of the `vmap`-based design: the DP guarantee holds
for *any* JAX-traceable loss function, because per-example isolation is
enforced at the computation level rather than at the layer level.

---

(opaque-code)=
### Opaque Code

**The pitfall.** If a DP library's implementation is opaque — closed-source,
poorly documented, or excessively complex — it is impossible for the broader
DP research community to verify that the implementation provides the claimed
guarantees. This is not a hypothetical concern: bugs in DP implementations
have been discovered in widely-used libraries, sometimes years after release.

Excessive complexity often arises from deep framework integration, where the
DP logic must conform to the framework's abstractions rather than following
the natural structure of the DP mechanism. The resulting code can be so
difficult to follow that even experienced DP researchers struggle to verify
its correctness.

**How JAX Privacy handles it.** JAX Privacy is fully open source under the
Apache 2.0 license. The codebase is designed for auditability:

- **Flat architecture.** No component depends on any other component, so each
  module can be audited independently.
- **Formal guarantees in docstrings.** Public APIs document their sensitivity
  guarantees and assumptions explicitly.
- **Exposed internals.** The `DPExecutionPlan` exposes the `DpEvent` so that
  users can independently verify the privacy accounting.
- **Minimal framework coupling.** The privacy-critical code does not depend on
  any training framework, eliminating an entire category of auditability
  concerns.
