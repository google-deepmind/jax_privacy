# Troubleshooting

If you don't find a solution for your problem here, look for similar
[issues on our GitHub](https://github.com/google-deepmind/jax_privacy/issues).

This documentation is in active development and we are still gathering most
common pitfalls and issues that might arise during the use of our library. If
you have one, you are very welcome to contribute!

## Loss does not go down

Check whether it goes down with non-DP training. If it does not go down for
non-DP either, then try usual ML remediation techniques like changing learning
rate, using different optimizers. Such a problem might often happen if you use
16 bit mixed precision because more error is accumulated. In this case
accumulating less gradients might help.

If it happens only for DP training, then it worth noting that DP training in
general converges slower than usual training. In this case, try to do more
iterations, increase batch size, both physical and effective, increase learning
rate.
