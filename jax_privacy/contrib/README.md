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

# Contributed Mechanism Components (`contrib/`)

This directory houses community-contributed mechanism components for JAX
Privacy. Its primary purpose is to establish a clean boundary between
contributor-written code and core maintainer-written code, providing a home for
novel algorithmic components in differentially private machine learning.

## Scope & Design Philosophy

*   **Modular Differentiating Components (No E2E Reimplementations):**
    `contrib/` should **not** house full end-to-end training loops or monolithic
    mechanism pipelines. Instead, contributions should focus strictly on the
    **key modular components** that differentiate a paper or method from
    standard DP-SGD (or other primitives already supported by JAX
    Privacy)—for example, a custom per-sample clipping rule, an adaptive noise
    schedule, a specialized privacy accountant, a custom optimizer, etc.
*   **Self-Contained Paper Implementations:** Ideally, each file in `contrib/`
    corresponds to a single paper in DP ML and is fully self-contained
    (e.g., `contrib/<feature_name>.py`).
*   **Engage with Core Abstractions:** The best contributions engage directly
    with JAX Privacy's core library abstractions (such as
    `AugmentedGradientTransformation`, custom clipping interfaces, privacy
    accountants, and execution plans).
*   **Close Gaps Upstream:** If existing core abstractions are not sufficiently
    flexible to support a clean and lightweight integration of your mechanism,
    please **open a GitHub Issue** before implementing complex workarounds. We
    welcome discussions on evolving upstream interfaces to better accommodate
    novel DP techniques.

## Consumption & Packaging Policy

*   **No Direct Imports by End Users:** End users should **not** depend on or
    import from `contrib` directly (e.g., `from jax_privacy.contrib import ...`
    is discouraged and not part of the supported public API).
*   **Top-Level Omission:** `contrib` is deliberately not exposed in the
    top-level `jax_privacy/__init__.py` namespace.
*   **Targeted Re-Exports:** Maintainers curate and provide targeted re-exports
    or integrations of accepted `contrib` components through the standard
    public modules (e.g., `jax_privacy.clipping`, `jax_privacy.noise_addition`,
    `jax_privacy.optimizers`).

## Roles & Maintenance Expectations

*   **Implementation Faithfulness & Testing:** Contributors own their
    implementation's fidelity to the underlying paper, correctness across
    edge cases, and unit test coverage. Maintainers ensure continuous
    integration (CI) build compatibility.
*   **DP-Correctness & Review:**
    *   **Safe Pluggable Interfaces:** Components implementing safe interfaces
        (such as `AugmentedGradientTransformation`), where DP guarantees are
        enforced by the surrounding execution framework, undergo standard code
        quality and testing review without requiring formal DP proofs.
    *   **Formal DP Guarantees:** Components that define or modify formal DP
        guarantees (e.g., privacy accounting, clipping bounds, noise
        calibration) require thorough mathematical review to ensure privacy
        invariants hold.
*   **Lifecycle & Graduation:** Widely adopted, robustly tested components with
    active maintainer sponsorship may graduate into the core `jax_privacy`
    library. Conversely, unmaintained or broken code may be deprecated and
    removed.

## Contributing Guidelines

1.  **File Organization:**
    *   Place implementation modules directly under `contrib/` (e.g.,
        `contrib/<feature_name>.py`).
    *   Include comprehensive unit tests in a corresponding test file (e.g.,
        `tests/contrib/<feature_name>_test.py`).
2.  **Documentation & Style:**
    *   Include a module-level docstring with full bibliographic citations to
        the relevant paper (author names, year, and clickable paper/arXiv
        links).
    *   Document public functions and classes with complete type annotations,
        `Args`, `Returns`, and docstring examples.
    *   All files must include the standard Apache 2.0 license header.
    *   Follow Google Python style (`pyink`, `pylint`) and adhere to JAX's
        functional, immutable design patterns.
