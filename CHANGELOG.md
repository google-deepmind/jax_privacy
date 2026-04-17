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

## [2.0.0] - 2026-04-16

Since version 1.0.0, the **jax-privacy** library has undergone a major
structural transformation, transitioning from a framework-coupled design to a
modular, extensible architecture. This includes a complete overhaul of batch
selection strategies, the introduction of experimental Monte Carlo accounting,
and expanded support for matrix factorization mechanisms.

### 1. API Changes

*   **Decoupling and Deprecation**: The monolithic `dp_sgd`,
    `training`, and `experiments` modules have been removed. The library now
    prioritizes decoupled modules like `clipping.py`, `noise_addition.py`, and
    `batch_selection.py`, which can be integrated into custom training loops
    without inheriting framework-specific dependencies.

### 2. Experimental Features

*   **DPExecutionPlan**: Introduced an experimental `DPExecutionPlan` that
    encapsulates the "three pillars" of a differentially private mechanism:
    batch selection, clipping, and noise addition. This allows for
    framework-agnostic execution of DP-SGD and other mechanisms.
*   **Monte Carlo Accounting**: A new experimental framework for privacy
    accounting using Monte Carlo simulations was added. This includes utilities
    for generating privacy loss distribution (PLD) samples and calculating
    overall deltas for complex mechanisms that are difficult to analyze
    analytically.

### 3. New Features, Components, and Mechanisms

*   **Expanded Batch Selection Strategies**: Batch Selection API: Reorganized
    the batch selection API to support diverse strategies, including fixed-size,
    cyclic Poisson, balls-in-bins, and b-min-sep sampling. It also includes a
    UserSelectionStrategy for user-level DP.
*   **Auditing Enhancements**:
    *   **One-Run Auditing**: Added support for "Auditing f-Differential Privacy
        in One Run," allowing empirical privacy estimation from a single
        training run.
    *   **Threshold Strategies**: Refactored `CanaryScoreAuditor` to support
        multiple threshold selection strategies, including `Split` and
        `MultiSplit` for epsilon auditing.
*   **Matrix Factorization and BandMF**: Expanded support for private matrix
    factorization with implementations for `Banded`, `Dense`, and `Toeplitz`
    matrices, including specific accountants for BandMF with Cyclic Poisson
    sampling.

### 4. Examples and Documentation

*   **Advanced Training Examples**:
    *   **Secure Noise**: Demonstrates the use of cryptographic-strength
        randomness for noise addition and batch selection.
    *   **Transformers and LLMs**: Added examples for training Transformers with
        DP-SGD and fine-tuning **Gemma3** using LoRA.
    *   **Synthetic Data**: Provided a notebook for using DP-SGD to generate
        synthetic data for Gemma3.
*   **Enhanced Documentation**:
    *   Added a "Library Design" document and a "Sharp Edges" guide covering
        pitfalls like mixed-precision training.
    *   Improved discoverability of batch selection strategies and their
        partitioning logic.

### 5. Bug Fixes

*   **Auxiliary Output Alignment**: Fixed a bug where auxiliary outputs had an
    incorrect extra dimension when `keep_batch_dim=True` was used; the
    implementation now correctly squeezes this artificial batch dimension.
*   **Mixed Precision Support**: Added an optional `dtype` argument to noise
    addition and clipping functions to ensure consistency during low-precision
    or mixed-precision training.
*   **Type Safety and Consistency**: `clip_pytree` was refactored to guarantee
    that the output dtype matches the input dtype, preventing issues where
    `jax.lax.cond` could produce inconsistent return types.

### 6. Internal Refactorings and Code Cleanups

*   **Optax Migration**: Internal microbatching utilities were migrated to the
    public `optax.microbatching` API to reduce redundancy, and the
    experimental `microbatching.py` file was subsequently removed.
*   **Unified Noise Addition**: Refactored the noise addition pipeline to
    support customizable intermediate sharding strategies and factored out
    general-purpose sharding utilities into dedicated files, improving the
    maintainability of distributed noise generation.

## [1.0.0] - 2025-07-10

### Added

-   First official stable release of JAX-Privacy — a library for
    privacy-preserving machine learning in
    JAX.[(#24)](https://github.com/google-deepmind/jax_privacy/pull/24)
-   Full support for DP-SGD training with JAX, including integration with:
    -   Raw JAX training pipelines
    -   Flax Linen API
    -   Keras API for high-level model fine-tuning workflows.
-   End-to-end examples and Jupyter notebooks demonstrating usage of the
    library.
-   Core implementations of DP algorithms, including:
    -   Differentially Private Stochastic Gradient Descent (DP-SGD)
    -   Utilities for DP accounting (computing noise scales given privacy
        parameters) integrated into training pipelines.

### Changed

-   Library APIs designed for a more production-ready and modular experience
    compared with earlier research code.

### Fixed

-   Stability and packaging improvements to support publication on PyPI as
    version 1.0.0.

<!-- disableFinding(LINK_UNUSED_ID) -->

[unreleased]: https://github.com/google-deepmind/jax_privacy/compare/v1.0.0...HEAD.

<!-- enableFinding(LINK_UNUSED_ID) -->

[1.0.0]: https://github.com/google-deepmind/jax_privacy/compare/v0.2...v1.0.0.
