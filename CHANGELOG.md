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

[unreleased]: https://github.com/google-deepmind/jax_privacy/compare/v1.0.0...HEAD.
[1.0.0]: https://github.com/google-deepmind/jax_privacy/compare/v0.2...v1.0.0.

