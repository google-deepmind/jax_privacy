JAX Privacy is an open-source library for differentially private (DP) training
of machine learning models. Originally developed to support research on DP image
classification within DeepMind, it has subsequently been extended to other
models, data modalities, use cases, and contributors. JAX Privacy is currently
being developed and maintained to support the following goals:

*   Provide a production-focused API for differentially-private training of ML
    models in JAX and Keras.
*   Enable reproducibility of DP training research done at Google.
*   Provide a platform enabling external researchers to easily experiment with
    settings relevant to Google's DP training ecosystem and problem set.

The library is still in development and we are actively working on improving its
usability and functionality. If you have any feedback or feature requests,
please don't hesitate to [contact us](support).

DP training is an active research area with many recent developments. It does
not come for free therefore expect:

*   Some accuracy decrease compared to non-DP model verions, in most of the
    cases it will be negligible.
*   Larger training time.
*   More hyperparameters to tune.

If you are unfamiliar with the following topics, we recommend reading the
provided literature:

*   Differential Privacy: an approach to bound the leakage of information about
    individuals during data processing, including training of ML models. Start
    with
    [this Medium blog post](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3),
    dive deeper if you want [1](https://programming-dp.com/),
    [2](https://desfontain.es/blog/differential-privacy-awesomeness.html),
    [3](https://en.wikipedia.org/wiki/Differential_privacy).
*   [JAX](https://docs.jax.dev/en/latest/quickstart.html): numpy-like library
    for high-performance numerical computation and gradient calculations,
    successor of TensorFlow.

Note that the documentation is written with the assumption that you are familiar
with DP (especially DP for machine learning) and JAX, in particular with
terminology used there.

Additionally, depending on APIs you will use, you might need to get familiar
with one of the following libraries:

*   [Flax NNX](https://flax-linen.readthedocs.io/en/latest/quick_start.html):
    PyTorch-like high-level library for building neural networks in JAX.
*   [Flax Linen](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/index.html):
    predecessor of Flax NNX, more JAX-like than Pytorch-like.
*   [Keras](https://keras.io): even more high-level library for building and
    training ML models using different backends including JAX and PyTorch.

You can navigate the documentation in the following way:

1.  Start with short [Overview](overview) of the library main components to get
    better high-level understanding.
1.  Install the library following instructions in [Installation](installation)
    section.
1.  If you want to use the library for your own use-case, choose the framework
    you want to use (e.g. [Keras](keras_api.rst) or [JAX & Flax](jax_api)) and
    study its API capabilities with the provided examples.
1.  If you are interested in doing some specific task (e.g. LLM fine-tuning),
    navigate to the corresponding example in the "Examples" section.
1.  If you want to reproduce paper results, navigate to
    [Paper Results Reproduction](paper_reproductions).
1.  If you run into issues, [Troubleshooting](troubleshooting) might help.
1.  If you want to understand the library's design, navigate to
    [Library Design](library_design). It can be useful for library contributors.
1.  If you want to contribute to the library, navigate to
    [Contribution Guide](contribution_guide).
1.  If you need help, navigate to [Support](support).
