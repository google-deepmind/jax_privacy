# JAX-Privacy: Algorithms for Privacy-Preserving Machine Learning in JAX

| [**Library**](#library) | [**Installation**](#installation) |
[**Reproducing Results**](#reproducing-results) | [**Citing**](#citing) |
[**Contact**](#contact)

This repository contains:

*   A production-focused API for differentially-private (DP) training of ML
    models in JAX and Keras.
*   A library of core components for implementing differentially private machine
    learning algorithms in JAX.
*   A JAX-based machine learning DP pipeline using components from the library
    to experiment with image classification models.

This code is open-sourced with the main objective of transparency and
reproducibility for research purposes, and includes production-focused APIs for
differentially private machine learning. Some rough edges should be expected,
especially in the research components.

## New: Production-Focused JAX Privacy Library <a id="library"></a>

We are excited to introduce a more production-focused JAX Privacy API designed
to simplify the development of differentially-private (DP) machine learning
models, including Large Language Models (LLMs).

**Key Features:**

*   **Algorithm Support**: Currently supports the DP-SGD (Differentially Private
    Stochastic Gradient Descent) algorithm. We are actively working on
    incorporating more DP algorithms (e.g. DP-FTRL) in the near future.
*   **Framework Integration**: The library provides APIs tailored for different
    JAX-based development experiences:
    *   **Keras**: A high-level API, excellent for common tasks like fine-tuning
        LLMs. See
        [Keras API simple example](https://github.com/google-deepmind/jax_privacy/blob/main/examples/keras_api_example.py)
        and
        [Gemma fine-tuning notebook](https://github.com/google-deepmind/jax_privacy/blob/main/examples/dp_sgd_keras_gemma3_lora_finetuning_samsum.ipynb)
        to get started.
    *   **Flax Linen**: Offers greater flexibility for custom model
        architectures and training loops, at the cost of some additional
        boilerplate. See
        [MNIST notebook](https://github.com/google-deepmind/jax_privacy/blob/main/examples/dp_sgd_flax_linen_mnist.ipynb)
        to get started.
    *   **Raw JAX**: Provides the most low-level control. Recommended for
        researchers who want to test out new ideas, people who want to use a
        framework not listed above (e.g. equinox), or people who want a more
        numpy-like experience.
        <!-- TODO - b/398715962: add "External Contributions & Design" section, link to readthedocs -->

This new JAX Privacy API aims to provide a more streamlined and robust
experience for building DP ML models, complementing the existing
research-focused components.

We believe this new API will significantly lower the barrier to implementing DP
in your machine learning projects.

## Installation<a id="installation"></a>

**Note:** to ensure that your installation is compatible with your local
accelerators such as a GPU, we recommend to first follow the corresponding
instructions to install [JAX](https://github.com/jax-ml/jax#installation).

### Option 1: Static Installation

This option is preferred for the purpose of re-using functionalities of our
library without modifying them. The library package can be installed by running
the following command-line:

```
pip install git+https://github.com/google-deepmind/jax_privacy
```

This will not install the training pipeline.

### Option 2: Local Installation <a id="install-option2"></a>

This option is preferred to either build on top of our codebase, or to reproduce
our results using the training pipeline.

*   The first step is to clone the repository:

```
git clone https://github.com/google-deepmind/jax_privacy
```

*   Then the code can be installed. We recommend local installation so
    modifications to the code are reflected in imports of the package:

```
cd jax_privacy
pip install -e .
```

## Reproducing Results<a id="reproducing-results"></a>

### Unlocking High-Accuracy Differentially Private Image Classification through Scale

*   Instructions:
    [experiments/image_classification](experiments/image_classification).
*   arXiv link: https://arxiv.org/abs/2204.13650.
*   Bibtex reference:
    [link](https://github.com/google-deepmind/jax_privacy/blob/main/bibtex/de2022unlocking.bib).

### Unlocking Accuracy and Fairness in Differentially Private Image Classification

*   Instructions:
    [experiments/image_classification](experiments/image_classification).
*   arXiv link: https://arxiv.org/abs/2308.10888.
*   Bibtex reference:
    [link](https://github.com/google-deepmind/jax_privacy/blob/main/bibtex/berrada2023unlocking.bib).

## How to Cite This Repository <a id="citing"></a>

If you use code from this repository, please cite the following reference:

```
@software{jax-privacy2022github,
 author = {Balle, Borja and Berrada, Leonard and Charles, Zachary and
Choquette-Choo, Christopher A and De, Soham and Doroshenko, Vadym and Dvijotham,
Dj and Galen, Andrew and Ganesh, Arun and Ghalebikesabi, Sahra and Hayes, Jamie
and Kairouz, Peter and McKenna, Ryan and McMahan, Brendan and Pappu, Aneesh and
Ponomareva, Natalia and Pravilov, Mikhail and Rush, Keith and Smith, Samuel L
and Stanforth, Robert},
  title = {{JAX}-{P}rivacy: Algorithms for Privacy-Preserving Machine Learning in JAX},
  url = {http://github.com/google-deepmind/jax_privacy},
  version = {0.4.0},
  year = {2025},
}
```

## Contact <a id="contact"></a>

If you have any questions or feedback, you can contact us via email:
jax-privacy-open-source@google.com.

## Acknowledgements

-   [NFNet codebase](https://github.com/deepmind/deepmind-research/tree/master/nfnets)
-   [DeepMind JAX Ecosystem](https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt)

## License

All code is made available under the Apache 2.0 License. Model parameters are
made available under the Creative Commons Attribution 4.0 International (CC BY
4.0) License.

See https://creativecommons.org/licenses/by/4.0/legalcode for more details.

## Disclaimer

This is not an official Google product.
