# Sharp Edges

## jax.vmap + jax.sharding

When using JAX privacy to train or fine-tune large models with data or model parallelism, one must be careful when imposing sharding annotations on inputs, outputs, and intermediates. Consider the following toy example for demonstration purposes, where we are using pure data parallelism on a simple mean estimation task. The input data consists of a batch of 1024 scalars, and the parameter is a single scalar. The loss function imposes a sharding constraint on the data that it is distributed across these 8 devices. This function (and it’s grad) works great when passing in batches of the expected shape, as demonstrated below:

```python
import jax

jax.config.update('jax_num_cpu_devices', 8)
P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((8,), ('data',), devices=jax.devices())
sharding = jax.sharding.NamedSharding(mesh, P('data'))

def loss(param, data):
  data = jax.lax.with_sharding_constraint(data, sharding)
  return 0.5 * jnp.mean((data - param)**2)

loss_and_grad = jax.grad(loss)

param = 4.0
data = jax.random.normal(jax.random.key(0), (1024,))
loss_and_grad(param, data)
```
```
Array(4.004355, dtype=float32, weak_type=True)
```

However, when we use jax privacy instead, this breaks, as seen below:

```python
import jax_privacy

loss_and_clipped_grad = jax_privacy.clipped_grad(loss, l2_clip_norm=1.0)
loss_and_clipped_grad(param, data)
```
```
ValueError: One of pjit outputs with pytree key path result was given the sharding of NamedSharding(mesh=Mesh('data': 8, axis_types=(Auto,)), spec=PartitionSpec(UNCONSTRAINED, 'data'), memory_kind=unpinned_host), which implies that the global size of its dimension 1 should be divisible by 8, but it is equal to 1 (full shape: (1024, 1))
```

The problem is that `clipped_grad` forms size-1 batches and passes them into
 the loss / grad function using jax.vmap. Since the function expects batches of
 size 1024 (or at least something divisible by 8), this fails. The simplest fix
 is to rewrite loss to remove the sharding annotation (possibly adding it back
 in elsewhere):

```python
def loss(param, data):
  return 0.5 * jnp.mean((data - param)**2)
```

```python
loss_and_clipped_grad = jax_privacy.clipped_grad(loss, l2_clip_norm=jnp.inf, normalize_by=1024)

loss_and_clipped_grad(params, data)
```
```
Array(4.004355, dtype=float32)
```

```python
def loss_and_clipped_grad_wsc(params, data):
  data = jax.lax.with_sharding_constraint(data, sharding)
  return loss(param, data)

loss_and_clipped_grad_wsc(params, data)
```
```
Array(4.004355, dtype=float32)
```

This simple approach may not always be feasible, e.g., if the sharding
constraint is applied to an intermediate value computed by loss rather than an
input/output. One thus be very careful with how the code is designed and how
sharding constraints are incorporated into the program. When trying to integrate
jax privacy into an existing large-scale training framework, this can be
particularly challenging in some cases. The “right” solution to this thorny
issue is not obvious and we do not prescribe a specific approach here, but just
provide a warning to users who might encounter similar issues.
