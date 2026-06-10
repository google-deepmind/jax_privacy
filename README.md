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

# JAX-Privacy: Algorithms for Privacy-Preserving Machine Learning in JAX

| [**Docs**](https://jax-privacy.readthedocs.io/) | [**Citing**](#citing) |
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

For installation instructions, examples, and full API documentation, please
visit the [JAX Privacy documentation](https://jax-privacy.readthedocs.io/).

## How to Cite This Repository <a id="citing"></a>

If you use code from this repository, please cite the following reference:

```
@software{jax-privacy2022github,
 author = {Balle, Borja and Berrada, Leonard and Charles, Zachary and
Choquette-Choo, Christopher A and De, Soham and Doroshenko, Vadym and Dvijotham,
Dj and Galen, Andrew and Ganesh, Arun and Ghalebikesabi, Sahra and Hayes, Jamie
and Kairouz, Peter and McKenna, Ryan and McMahan, Brendan and Pappu, Aneesh and
Ponomareva, Natalia and Pravilov, Mikhail and Rush, Keith and Smith, Samuel L
and Stanforth, Robert and Mishra, Chaitanya},
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

