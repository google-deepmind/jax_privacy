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

# Installation

**Note:** to ensure that your installation is compatible with your local
accelerators such as a GPU, we recommend to first follow the corresponding
instructions to install [JAX](https://github.com/jax-ml/jax#installation).

## Option 1: Static Installation

This option is preferred for the purpose of re-using functionalities of our
library without modifying them. The library package can be installed by running
the following command-line:

```
pip install git+https://github.com/google-deepmind/jax_privacy
```

This will not install the training pipeline.

## Option 2: Local Installation

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
