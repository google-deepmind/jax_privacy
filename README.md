# JAX-Privacy: Algorithms for Privacy-Preserving Machine Learning in JAX

[**Installation**](#installation)
| [**Reproducing Results**](#reproducing-results)
| [**Citing**](#citing)

This repository contains the JAX implementation of algorithms that we develop
in our research on privacy-preserving machine learning.
This research code is open-sourced with the main objective of
transparency and reproducibility, so (some) rough edges should be expected.

## Installation<a id="installation"></a>

### Option 1: Static Installation

The code can be installed by running the following command-line:

```
pip install git+https://github.com/deepmind/jax_privacy
```

### Option 2: Downloading and Modifying the Code

If you wish to modify the code after the package is installed, you can clone
the code:

```
git clone https://github.com/deepmind/jax_privacy
```

Then the code can be installed so that local modifications to the code are
reflected in imports of the package:

```
cd jax_privacy
python setup.py develop
```

## Reproducing Results<a id="reproducing-results"></a>

The instructions are detailed in [experiments/image_classification](experiments/image_classification).

## How to Cite<a id="citing"></a>

### This Repository

```
@software{jax-privacy2022github,
  author = {Balle, Borja and Berrada, Leonard and De, Soham and Hayes, Jamie and Smith, Samuel L and Stanforth, Robert},
  title = {{JAX}-{P}rivacy: Algorithms for Privacy-Preserving Machine Learning in JAX},
  url = {http://github.com/deepmind/jax_privacy},
  version = {0.1.0},
  year = {2022},
}
```


### Our Research

```
@article{de2022differentially,
  title={{Unlocking High-Accuracy Differentially Private Image Classification through Scale}},
  author={De, Soham and Berrada, Leonard and Hayes, Jamie and Smith, Samuel L and Balle, Borja},
  journal={arXiv},
  year={2022}
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
