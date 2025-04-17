# JAX-Privacy: Algorithms for Privacy-Preserving Machine Learning in JAX

[**Installation**](#installation)
| [**Reproducing Results**](#reproducing-results)
| [**Citing**](#citing)

This repository contains:

* A library of core components for implementing differentially private (DP)
machine learning algorithms in JAX.
* A JAX-based machine learning DP pipeline using components from the library to
experiment with image classification models.

This research code is open-sourced with the main objective of
transparency and reproducibility, so (some) rough edges should be expected.

## Installation<a id="installation"></a>

**Note:** to ensure that your installation is compatible with your local
accelerators such as a GPU, we recommend to first follow the corresponding
instructions to install [TensorFlow](https://github.com/tensorflow/tensorflow#install)
and [JAX](https://github.com/jax-ml/jax#installation).

### Option 1: Static Installation

This option is preferred for the purpose of re-using functionalities of our
library without modifying them.
The library package can be installed by running the following command-line:

```
pip install git+https://github.com/google-deepmind/jax_privacy
```

This will not install the training pipeline.

### Option 2: Local Installation <a id="install-option2"></a>

This option is preferred to either build on top of our codebase, or to reproduce
our results using the training pipeline.

* The first step is to clone the repository:

```
git clone https://github.com/google-deepmind/jax_privacy
```

* Then the code can be installed. We recommend local installation so
modifications to the code are reflected in imports of the package:

```
cd jax_privacy
pip install -e .
```

## Reproducing Results<a id="reproducing-results"></a>

### Unlocking High-Accuracy Differentially Private Image Classification through Scale

* Instructions: [experiments/image_classification](experiments/image_classification).
* arXiv link: https://arxiv.org/abs/2204.13650.
* Bibtex reference: [link](https://github.com/google-deepmind/jax_privacy/blob/main/bibtex/de2022unlocking.bib).

### Unlocking Accuracy and Fairness in Differentially Private Image Classification

* Instructions: [experiments/image_classification](experiments/image_classification).
* arXiv link: https://arxiv.org/abs/2308.10888.
* Bibtex reference: [link](https://github.com/google-deepmind/jax_privacy/blob/main/bibtex/berrada2023unlocking.bib).

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

## Acknowledgements

- [NFNet codebase](
https://github.com/deepmind/deepmind-research/tree/master/nfnets)
- [DeepMind JAX Ecosystem](https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt)

## License

All code is made available under the Apache 2.0 License.
Model parameters are made available under the Creative Commons Attribution 4.0
International (CC BY 4.0) License.

See https://creativecommons.org/licenses/by/4.0/legalcode for more details.

## Disclaimer

This is not an official Google product.
