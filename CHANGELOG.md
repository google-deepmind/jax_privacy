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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on https://keepachangelog.com/en/1.1.0/

## [1.0.0] - 2025-07-10

### Added

- First official stable release of JAX-Privacy — a library for privacy-preserving machine learning in JAX.[(#24)](https://github.com/google-deepmind/jax_privacy/pull/24)
- Full support for DP-SGD training with JAX, including integration with:
  - Raw JAX training pipelines
  - Flax Linen API
  - Keras API for high-level model fine-tuning workflows.
- End-to-end examples and Jupyter notebooks demonstrating usage of the library.
- Core implementations of DP algorithms, including:
  - Differentially Private Stochastic Gradient Descent (DP-SGD)
  - Utilities for DP accounting (computing noise scales given privacy parameters) integrated into training pipelines.

### Changed

- Library APIs designed for a more production-ready and modular experience compared with earlier research code.

### Fixed

- Stability and packaging improvements to support publication on PyPI as version 1.0.0.

<!-- disableFinding(LINK_UNUSED_ID) -->
[unreleased]: https://github.com/google-deepmind/jax_privacy/compare/v1.0.0...HEAD.
<!-- enableFinding(LINK_UNUSED_ID) -->
[1.0.0]: https://github.com/google-deepmind/jax_privacy/compare/v0.2...v1.0.0.

