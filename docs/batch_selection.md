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

# Batch Selection Strategies

<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_ID) -->

This document overviews the research behind different batch selection
(subsampling) strategies for differentially private training and summarizes
their support in `jax_privacy` and `dp_accounting`.

## Background

Differentially private training algorithms like DP-SGD
([Abadi+ 2016](#abadi-2016))
rely on two key ingredients: (1) gradient clipping to bound each example's
contribution, and (2) additive noise to mask individual examples.
The **batch selection strategy** determines which subset of examples
participates in each training step, and the analysis of this selection
strategy is critical for quantifying the overall privacy guarantee.

### Noise addition mechanisms

jax_privacy supports two noise addition paradigms:

*   **DP-SGD** (independent noise): Gaussian noise is added independently at
    each step.
    See `gaussian_privatizer()` in
    [`noise_addition.py`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/noise_addition.py).
*   **DP-MF** (correlated noise via Matrix Factorization): Noise is correlated
    across steps using a lower-triangular matrix, which can reduce total error.
    See
    [Pillutla+ 2025](#pillutla-2025)
    for an introduction and survey, and the
    [`matrix_factorization`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/matrix_factorization)
    module and `matrix_factorization_privatizer()` in
    [`noise_addition.py`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/noise_addition.py).

### API tiers and batch selection

jax_privacy provides [three API tiers](api-tiers) with different levels of
built-in DP assurance; see the [Overview](overview) for the full description.
In the context of batch selection:

*   **Tier 1 (end-to-end training loops)** and
    **Tier 2 (`DPExecutionPlan` via `BandMFConfig`)**: Batch selection is
    configured automatically and coupled with noise addition and accounting by
    construction. Currently supports `CyclicPoissonSampling` variants
    (including truncated Poisson and fixed-order multi-epoch). See
    [`dp_sgd_transformer.py`](https://github.com/google-deepmind/jax_privacy/tree/main/examples/dp_sgd_transformer.py)
    and
    [`dp_logistic_regression.py`](https://github.com/google-deepmind/jax_privacy/tree/main/examples/dp_logistic_regression.py)
    for examples.

*   **Tier 3 (Core API)**: Users compose
    [`batch_selection`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/batch_selection.py),
    [`clipping`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/clipping.py),
    [`noise_addition`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/noise_addition.py),
    and
    [`accounting`](https://github.com/google-deepmind/jax_privacy/tree/main/jax_privacy/accounting.py)
    modules directly. **Any** `BatchSelectionStrategy` can be used with **any**
    noise addition scheme and **any** `DpEvent` — including Monte Carlo
    accounting via `jax_privacy.experimental.monte_carlo` for combinations
    where PLD/RDP accounting is not available. See
    [`jax_api_example.py`](https://github.com/google-deepmind/jax_privacy/tree/main/examples/jax_api_example.py)
    and
    [`balls_in_bins_accounting.py`](https://github.com/google-deepmind/jax_privacy/tree/main/examples/balls_in_bins_accounting.py)
    for examples.

---

## Summary table

The table below lists all batch selection strategies, both those implemented in
jax_privacy and those discussed in the literature. Support levels are:

*   ✅ **Supported**: Implemented in jax_privacy with accounting support.
*   📄 **Analyzed**: Privacy analysis exists in the literature but accounting is
    not yet implemented in jax_privacy.
*   ❓ **Not analyzed**: No known privacy analysis for this combination.
*   N/A: Not applicable (e.g., DP-SGD is a special case of DP-MF with the
    identity matrix, so these entries are simply not relevant rather than
    invalid).

Privacy accounting is done via the
[dp_accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting)
package by consuming `DpEvent` objects. There are two main accountants: **PLD**
and **RDP**. PLD is generally tighter and also supports additional methods
beyond standard $(\varepsilon, \delta)$-DP:

*   `get_true_positive_rates(false_positive_rates)`: Upper bounds on TPR at
    given FPR values, using the full PLD (not just an $(\varepsilon, \delta)$ summary). Important
    for membership inference analysis, especially at very low FPR.
*   `get_gdp_parameter_estimate()`: Estimates the $\mu$-GDP parameter from the PLD.

These methods are **PLD-only** — they are not available via the RDP accountant.
See the [Accounting support matrix](#accounting-support-matrix) below for
details on which `DpEvent` types support each accountant.

For supported combinations, the accounting approach is noted as **PLD**,
**RDP**, or **MC** (Monte Carlo, experimental).

<!-- mdformat off -->

| Strategy | jax_privacy class | DP-SGD | DP-MF | Accounting | Key papers |
|---|---|---|---|---|---|
| Poisson sampling | [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html) (cycle_length=1) | ✅ | ✅ (Tier 2 via `BandMFConfig`) | PLD via `dpsgd_event()` / `amplified_bandmf_event()` | [Abadi+ 2016](#abadi-2016) |
| Cyclic Poisson (BandMF) | [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html) (cycle_length > 1) | N/A | ✅ (Tier 2 via `BandMFConfig`) | PLD via `amplified_bandmf_event()` | [Choquette-Choo+ 2023](#choquette-choo-2023), [McKenna 2024](#mckenna-2024) |
| Truncated Poisson | [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html) (truncated_batch_size set) | ✅ | ✅ (Tier 2 via `BandMFConfig`) | PLD via `truncated_dpsgd_event()` / `truncated_amplified_bandmf_event()` | [Chua+ 2024a](#chua-2024a), [Ganesh 2025](#ganesh-2025) |
| Fixed-order + multi-epoch | [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html) (sampling_prob=1) | N/A | ✅ (Tier 2 via `BandMFConfig`) | PLD via `amplified_bandmf_event()` | [Choquette-Choo+ 2022](#choquette-choo-2022) |
| Fixed-size (with replacement) | [`FixedBatchSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.FixedBatchSampling.html) (replace=True) | ✅ | ❓ | DpEvent created but no accountant support | [Balle+ 2018](#balle-2018) |
| Fixed-size (without replacement) | [`FixedBatchSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.FixedBatchSampling.html) (replace=False) | ✅ | ❓ | RDP via `fixed_dpsgd_event()` | [Balle+ 2018](#balle-2018), [Mironov+ 2019](#mironov-2019) |
| Balls-in-bins | [`BallsInBinsSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.BallsInBinsSampling.html) | 📄 | ✅ (MC) | MC via `balls_in_bins_accounting.py` example | [Choquette-Choo+ 2024](#choquette-choo-2024), [Chua+ 2024b](#chua-2024b) |
| Random allocation ($k$-out-of-$t$), a.k.a. Balanced Iteration Subsampling (BIS) | [`RandomAllocationSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.RandomAllocationSampling.html) | 📄 | 📄 | None implemented | [Liew+ 2022](#liew-2022), [Feldman+ 2025](#feldman-2025), [Dong+ 2025](#dong-2025), [Dong+ 2026b](#dong-2026b) |
| $b$-min-sep ($b > 1$) | [`BMinSepSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.BMinSepSampling.html) | 📄 | ✅ (MC) | MC via [`experimental.monte_carlo`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/_autosummary_output/jax_privacy.experimental.monte_carlo.sample_generation.generate_sample.html#jax_privacy.experimental.monte_carlo.sample_generation.generate_sample) | [Dong+ 2026a](#dong-2026a) |
| User-level wrapper | [`UserSelectionStrategy`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.UserSelectionStrategy.html) | ✅ | ✅ | Inherits from base strategy | — |
| Shuffling | [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html) (partition_type=EQUAL_SPLIT) | 📄 | ❓ | No accounting support | [Chua+ 2024a](#chua-2024a) |
| Multi-owner $b$-min-sep | `MultiOwnerMinSepSampling` | 📄 | 📄 | None implemented | [Ganesh+ 2025](#ganesh-2025-multi-user), [Dong+ 2026a](#dong-2026a) |

<!-- mdformat on -->

### Key observations

1.  **Poisson + Cyclic Poisson via `CyclicPoissonSampling`** is the most
    versatile strategy class. With different parameter settings it covers
    standard Poisson (cycle_length=1), BandMF-style cyclic partitioning
    (cycle_length > 1), fixed-order multi-epoch (sampling_prob=1), and truncated
    variants (truncated_batch_size set). All of these have PLD accounting,
    meaning `get_true_positive_rates` and `get_gdp_parameter_estimate` are
    available.

2.  **Balls-in-bins** is a special case of `CyclicPoissonSampling` with
    `partition_type=INDEPENDENT` and `sampling_prob=1` (included as an alias
    for better discoverability). It has an example of Monte Carlo accounting
    with DP-MF, but no PLD/RDP accounting. The near-exact privacy
    amplification framework of
    [Choquette-Choo+ 2024](#choquette-choo-2024)
    provides the theoretical foundation.

3.  **$b$-min-sep** is implemented in jax_privacy with Monte Carlo accounting
    support via `jax_privacy.experimental.monte_carlo`
    ([Dong+ 2026a](#dong-2026a)).

4.  **Random allocation** is implemented in jax_privacy but has no accounting
    functions in `accounting.py`. Privacy analysis is available in the
    literature
    ([Feldman+ 2025](#feldman-2025);
    [Feldman+ 2026](#feldman-2026);
    [Schuchardt+ 2026](#schuchardt-2026)),
    and dp_accounting support is
    [being developed](https://github.com/google/differential-privacy/pull/414).

5.  **Shuffling** is the most common batch selection strategy in practice but
    has a *strictly worse* privacy guarantee than Poisson subsampling
    ([Chua+ 2024a](#chua-2024a)).
    It can be parameterized via `CyclicPoissonSampling` with
    `partition_type=EQUAL_SPLIT`, but has no accounting support. Use Poisson
    or truncated Poisson sampling instead.

---

(accounting-support-matrix)=
## Accounting support matrix

The table below maps
[DpEvent types](https://github.com/google/differential-privacy/blob/main/python/dp_accounting/dp_accounting/dp_event.py)
to accountant support and available privacy metrics.

<!-- mdformat off -->

| DpEvent type | PLD | RDP | Neighboring relation | `get_epsilon` | `get_true_positive_rates` | `get_gdp_parameter_estimate` |
|---|---|---|---|---|---|---|
| `PoissonSampledDpEvent(GaussianDpEvent)` | ✅ | ✅ | ADD_OR_REMOVE_ONE, REPLACE_SPECIAL | ✅ (both) | ✅ (PLD) | ✅ (PLD) |
| `TruncatedSubsampledGaussianDpEvent` | ✅ | ❌ | REPLACE_ONE, REPLACE_SPECIAL | ✅ (PLD) | ✅ (PLD) | ✅ (PLD) |
| `SampledWithoutReplacementDpEvent` | ❌ | ✅ | REPLACE_ONE only | ✅ (RDP) | ❌ | ❌ |
| `SampledWithReplacementDpEvent` | ❌ | ❌ | — | ❌ | ❌ | ❌ |
| `MixtureOfGaussiansDpEvent` | ✅ | ❌ | ADD_OR_REMOVE_ONE, REPLACE_SPECIAL | ✅ (PLD) | ✅ (PLD) | ✅ (PLD) |

<!-- mdformat on -->

---

## Research appendix

(abadi-2016)=
### Abadi+ 2016: Deep Learning with Differential Privacy

*   **Paper**: [Abadi et al., 2016](https://arxiv.org/abs/1607.00133)
*   **Strategy**: Poisson subsampling — each example is included independently
    with probability $q = B/N$ (batch_size / dataset_size). **Algorithm 1**
    defines DP-SGD with Poisson subsampling. **Section 3** introduces the
    Moments Accountant for tracking cumulative privacy loss under subsampled
    mechanisms.
*   **MF compatibility**: Not discussed (independent noise, $C = I$).
*   **dp_accounting support**: `PoissonSampledDpEvent(GaussianDpEvent)` in PLD
    and RDP accountants. Full support for TPR/GDP via PLD.
*   **jax_privacy support**: ✅
    [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html)`(cycle_length=1)`
    + `dpsgd_event()`.

(balle-2018)=
### Balle+ 2018: Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences

*   **Paper**: [Balle, Barthe, Gaboardi, Hsu, Sato, 2018](https://arxiv.org/abs/1807.01647)
*   **Strategy**: Unified framework for analyzing privacy amplification across
    subsampling methods. **Theorem 9** covers subsampling without replacement.
    **Theorem 10** covers subsampling with replacement. **Section 3** develops
    the analytical tools (couplings, divergences, privacy profiles) and
    **Section 4** derives the main amplification bounds.
*   **MF compatibility**: Not discussed.
*   **dp_accounting support**: `SampledWithoutReplacementDpEvent` — RDP only,
    REPLACE_ONE only. No PLD support, no TPR/GDP.
*   **jax_privacy support**: ✅
    [`FixedBatchSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.FixedBatchSampling.html)`(replace=False)`
    + `fixed_dpsgd_event()`. Note: RDP-only accounting.

(mironov-2019)=
### Mironov+ 2019: Rényi Differential Privacy of the Sampled Gaussian Mechanism

*   **Paper**: [Mironov, Talwar, Zhang, 2019](https://arxiv.org/abs/1908.10530)
*   **Strategy**: Characterizes RDP for the Sampled Gaussian Mechanism
    (composition of Poisson subsampling + Gaussian noise). **Theorem 1**
    reduces the multi-dimensional case to 1D Gaussian mixtures. **Theorem 3**
    gives a closed-form bound: $\varepsilon = 2q^2\alpha/\sigma^2$ under
    conditions on $q$, $\sigma$, and $\alpha$. **Section 3.3** provides a
    numerically stable procedure for exact RDP computation.
*   **MF compatibility**: Not discussed.
*   **dp_accounting support**: Core reference for the RDP accountant
    implementation of `PoissonSampledDpEvent(GaussianDpEvent)`.
*   **jax_privacy support**: ✅ Referenced in `fixed_dpsgd_event()`.

(liew-2022)=
### Liew+ 2022: Privacy Amplification via Shuffled Check-Ins

*   **Paper**: [Liew, Hasegawa, Takahashi, 2022](https://arxiv.org/abs/2206.03151)
*   **Strategy**: Shuffled check-in protocol where clients independently decide
    to participate with probability $\gamma$. **Section 3** defines the protocol
    (§3.1), provides the RDP analysis (§3.2, Theorem 2), and introduces a
    numerical method for tracking privacy of Gaussian mechanisms under shuffling
    (§3.3). **Section 4** derives a lower bound for the Gaussian mechanism.
*   **MF compatibility**: Not discussed.
*   **dp_accounting support**: Not implemented.
*   **jax_privacy support**: ✅ Batch selection via
    [`RandomAllocationSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.RandomAllocationSampling.html)`(total_participations=1)`.
    Accounting not implemented.

(choquette-choo-2022)=
### Choquette-Choo+ 2022: Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning

*   **Paper**: [Choquette-Choo et al., 2022](https://arxiv.org/abs/2211.06530)
*   **Strategy**: Introduces $(k, b)$-participation schema where each example
    participates at most $k$ times with spacing $b$ between participations.
    **Section 2** formalizes $(k, b)$-participation and defines sensitivity for
    multi-participation adaptive streams. **Section 3** covers optimal matrix
    factorization for multiple epochs. **Section 4** introduces an FFT-based
    mechanism with reduced computational cost.
*   **MF compatibility**: ✅ Core paper for combining fixed-order sampling with
    banded Toeplitz matrices.
*   **dp_accounting support**: Via `amplified_bandmf_event()`.
*   **jax_privacy support**: ✅
    [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html)`(sampling_prob=1)`.

(choquette-choo-2023)=
### Choquette-Choo+ 2023: (Amplified) Banded Matrix Factorization: A Unified Approach to Private Training

*   **Paper**: [Choquette-Choo et al., 2023](https://arxiv.org/abs/2306.08153)
*   **Strategy**: Cyclic Poisson sampling with cyclic partitioning.
    **Section 3** introduces the $b$-min-sep participation schema for FL.
    **Section 4** covers optimization of banded matrices. **Section 5**
    introduces cyclic Poisson subsampling (Algorithm 2) and proves the privacy
    amplification result (**Theorem 1**, informal; formal in Theorem 3):
    $b$-banded BandMF achieves the same privacy as DP-SGD with $n/b$ iterations
    and sampling probability $B \cdot b / m$.
*   **MF compatibility**: ✅ Central paper. Introduces $b$-banded Toeplitz
    matrices and proves they achieve the same amplification as DP-SGD.
*   **dp_accounting support**: `PoissonSampledDpEvent(GaussianDpEvent)` via
    reduction to per-group independent Poisson DP-SGD.
*   **jax_privacy support**: ✅
    [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html)`(cycle_length > 1)`
    + `amplified_bandmf_event()` +
    [`BandMFConfig`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.execution_plan.BandMFConfig.html)
    (Tier 2).

(mckenna-2024)=
### McKenna 2024: Scaling up the Banded Matrix Factorization Mechanism for Differentially Private ML

*   **Paper**: [McKenna, 2024](https://arxiv.org/abs/2405.15913)
*   **Strategy**: Scales up DP-BandMF ([Choquette-Choo+ 2023](#choquette-choo-2023))
    along two axes: (1) *strategy optimization*: restricting to banded Toeplitz
    strategies enables $O(n \cdot b)$ time / $O(n)$ space (**Proposition 3.1**),
    scaling from $n \approx 10^4$ to $n > 10^7$ with $< 2\%$ RMSE degradation;
    (2) *distributed noise generation* (**§3.3**): shards correlated noise
    across accelerators, enabling support for large models with negligible
    overhead. Does not change the batch selection strategy (inherits cyclic
    Poisson sampling) or the privacy analysis (**Proposition 2.1** reduces
    $b$-banded BandMF to DP-SGD with $n/b$ iterations).
*   **MF compatibility**: ✅ Central focus. Scalable optimization of banded
    Toeplitz strategy matrices and distributed noise generation for DP-BandMF.
*   **dp_accounting support**: Same as
    [Choquette-Choo+ 2023](#choquette-choo-2023):
    `PoissonSampledDpEvent(GaussianDpEvent)` via reduction to amplified DP-SGD.
*   **jax_privacy support**: ✅
    [`BandMFConfig.default()`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.execution_plan.BandMFConfig.html)
    uses `toeplitz.optimize_banded_toeplitz()`, and `sharding_utils.py`
    implements the distributed noise generation.

(beltran-2024)=
### Beltran+ 2024: Efficient and Scalable Implementation of Differentially Private Deep Learning without Shortcuts

*   **Paper**: [Beltran et al., 2024](https://arxiv.org/abs/2406.17298)
*   **Strategy**: Addresses efficient implementation of Poisson subsampling in
    JAX without "shortcuts" (like shuffling). **Section 3** proposes "masked
    DP-SGD" that avoids recompilation by rounding up the batch size and masking
    extra gradients. **Lemma 1** proves that decomposing Poisson subsampling
    into a Binomial draw + WOR is equivalent to standard Poisson subsampling.
    Referenced for implementation correctness in `CyclicPoissonSampling`.
*   **MF compatibility**: N/A (implementation detail).
*   **dp_accounting support**: N/A.
*   **jax_privacy support**: Used internally.

(choquette-choo-2024)=
### Choquette-Choo+ 2024: Near Exact Privacy Amplification for Matrix Mechanisms

*   **Paper**: [Choquette-Choo et al., 2024](https://arxiv.org/abs/2410.06266)
*   **Strategy**: Balls-in-bins sampling — each example is independently
    assigned to a "bin" uniformly at random. **Section 2** introduces Monte
    Carlo accounting via PLD and the Estimate-Verify-Release framework.
    **Section 3** defines balls-in-bins batching (Definition 3.1) and provides
    the dominating pair via **Lemma 3.2** (dimension reduction). **Section 4**
    describes how to do Monte Carlo accounting for correlated noise with
    balls-in-bins. Enables joint optimization of correlation matrix $C$ with
    amplification for arbitrary lower-triangular non-negative $C$.
*   **MF compatibility**: ✅ Core paper for combining balls-in-bins with
    arbitrary matrix mechanisms (not limited to banded matrices).
*   **dp_accounting support**: Near-exact privacy parameters computable, but
    not integrated into dp_accounting. Monte Carlo accounting available via
    `jax_privacy.experimental.monte_carlo`.
*   **jax_privacy support**: ✅ Batch selection via
    [`BallsInBinsSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.BallsInBinsSampling.html).
    Accounting via Monte Carlo (see
    [`balls_in_bins_accounting.py`](https://github.com/google-deepmind/jax_privacy/tree/main/examples/balls_in_bins_accounting.py)).

(chua-2024a)=
### Chua+ 2024a: Scalable DP-SGD: Shuffling vs. Poisson Subsampling

*   **Paper**: [Chua et al., 2024](https://arxiv.org/abs/2411.04205)
*   **Strategy**: Establishes the privacy gap between shuffling and Poisson
    subsampling. **Section 2** defines Adaptive Batch Linear Queries (ABLQ)
    and dominating pairs. **Section 3** provides privacy analysis for all
    batch samplers: truncated Poisson (Theorem 3.3), persistent shuffling
    lower bound (Theorem 3.5), and dynamic shuffling lower bound (Theorem
    3.8). Introduces truncated Poisson subsampling as an XLA-compatible
    alternative (scalable implementation in Appendix A).
*   **MF compatibility**: Discusses independent noise only, but truncated
    Poisson has been extended to MF via
    [Ganesh 2025](#ganesh-2025).
*   **dp_accounting support**: `TruncatedSubsampledGaussianDpEvent` in PLD
    accountant.
*   **jax_privacy support**: ✅
    [`CyclicPoissonSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.CyclicPoissonSampling.html)`(truncated_batch_size=B)`
    + `truncated_dpsgd_event()`.
*   **Additional strategies from this paper not in jax_privacy**: Shuffling
    (not implemented; strictly worse privacy than Poisson).

(chua-2024b)=
### Chua+ 2024b: Balls-and-Bins Sampling for DP-SGD

*   **Paper**: [Chua et al., 2024](https://arxiv.org/abs/2412.16802)
*   **Strategy**: Balls-and-bins as a "best of both worlds." **Section 3**
    defines the sampling scheme. **Section 4** provides the privacy analysis via
    Monte Carlo accounting (estimating hockey-stick divergence). Shows this
    achieves privacy amplification comparable to Poisson subsampling while being
    practical like shuffling.
*   **MF compatibility**: Focused on DP-SGD (independent noise). MF extension
    is via
    [Choquette-Choo+ 2024](#choquette-choo-2024).
*   **dp_accounting support**: Not integrated.
*   **jax_privacy support**: ✅ Batch selection via
    [`BallsInBinsSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.BallsInBinsSampling.html).
    Accounting not directly implemented for DP-SGD.

(feldman-2025)=
### Feldman+ 2025: Privacy Amplification by Random Allocation

*   **Paper**: [Feldman & Shenfeld, 2025](https://arxiv.org/abs/2502.08202)
*   **Strategy**: $k$-out-of-$t$ random allocation — user's data is used in $k$
    steps chosen uniformly at random from $t$ total steps. **Section 3** reduces
    to a single non-adaptive randomizer. **Section 4.1** proves the main
    result (Theorem 4.1): random allocation privacy is upper-bounded by
    Poisson subsampling with rate $(1+o(1))k/t$. **Section 4.3** decomposes
    Poisson into a mixture of random allocations (Theorem 4.6). **Section
    4.4** provides direct RDP bounds (Theorem 4.8, closed-form for the
    remove direction).
*   **MF compatibility**: Only mentioned when referring to
    [Choquette-Choo+ 2024](#choquette-choo-2024).
*   **dp_accounting support**: Not integrated; analysis techniques provided.
*   **jax_privacy support**: ✅ Batch selection via
    [`RandomAllocationSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.RandomAllocationSampling.html).
    Accounting not implemented.

(dong-2025)=
### Dong+ 2025: Leveraging Randomness in Model and Data Partitioning for Privacy Amplification

*   **Paper**: [Dong, Chen, Ozgur, 2025](https://arxiv.org/abs/2503.03043)
*   **Strategy**: Concurrently introduced random allocation, which they call
    Balanced Iteration Subsampling (BIS): each sample participates in exactly
    $k$ iterations out of $T$. **Section 3.1** provides the core RDP bound
    (Theorem 3.1) for the dominating pair. **Section 3.2** applies this to
    model parallelism. **Section 3.3** defines BIS and shows it achieves
    privacy amplification similar to or stronger than Poisson (Theorem 3.8).
*   **MF compatibility**: Not discussed.
*   **dp_accounting support**: Not integrated.
*   **jax_privacy support**: Referenced in `RandomAllocationSampling` docstring.

(ganesh-2025-multi-user)=
### Ganesh+ 2025: It's My Data Too: Private ML for Datasets with Multi-User Training Examples

*   **Paper**: [Ganesh, McKenna, McMahan, Smith, Wu, 2025](https://arxiv.org/abs/2503.03622)
*   **Strategy**: Multi-attribution model where each example may be associated
    with multiple users. **Section 2** defines fixed-graph DP for multi-
    attribution. **Section 3** discusses batch selection strategies for multi-
    attribution DP-SGD/DP-MF: Poisson sampling with group privacy (Section
    3.2), $(k, b)$-min-sep with BandMF, and cyclic Poisson. **Section 4**
    proposes greedy algorithms for contribution bounding.
*   **MF compatibility**: Discussed in context of multi-owner DP-SGD.
*   **dp_accounting support**: For DP-SGD with Poisson sampling, the analysis
    can be reduced to group privacy which is supported by the
    `MixtureOfGaussiansDpEvent`, but currently jax_privacy does not provide a
    utility to do this easily. Unamplified DP-MF using $b$-min-separation is also
    immediately supported, as the mechanism behaves as a single application of
    the Gaussian mechanism.
*   **jax_privacy support**: ✅ Batch selection via
    `MultiOwnerMinSepSampling`, `MultiOwnerGraph`, and
    `greedy_contribution_bound()` (see `_multi_owner.py`).
    Accounting not implemented.

(pillutla-2025)=
### Pillutla+ 2025: Correlated Noise Mechanisms for Differentially Private Learning

*   **Paper**: [Pillutla et al., 2025](https://arxiv.org/abs/2506.08201)
*   **Strategy**: Comprehensive monograph on correlated noise mechanisms
    (matrix mechanisms / factorization mechanisms / DP-FTRL). Not a batch
    selection paper per se, but foundational for the noise addition side of
    DP-MF. Covers $(k, b)$-participation schemas, Poisson subsampling, cyclic
    participation, balls-in-bins, $b$-min-sep, and other batch selection methods
    as they relate to correlated noise mechanisms.
*   **MF compatibility**: ✅ Core reference for the theory behind
    `matrix_factorization_privatizer()`.
*   **dp_accounting support**: General framework; specific DpEvent types depend
    on the batch selection strategy used.
*   **jax_privacy support**: ✅ Referenced in `noise_addition.py`.

(ganesh-2025)=
### Ganesh 2025: Tighter Privacy Analysis for Truncated Poisson Sampling

*   **Paper**: [Ganesh, 2025](https://arxiv.org/abs/2508.15089)
*   **Strategy**: Truncated Poisson — Poisson sampling with probability $p$,
    truncated to maximum batch size $B$, with random subselection if $|S| > B$.
    **Section 2** defines the mechanism formally. **Section 3** provides the
    tighter privacy analysis (tighter than absorbing truncation probability
    into $\delta$). Generalizes to adaptive vector queries via post-processing and
    composition.
*   **MF compatibility**: ✅ Analysis extends to DP-MF via
    `truncated_amplified_bandmf_event()`.
*   **dp_accounting support**: `TruncatedSubsampledGaussianDpEvent` in PLD.
    Full TPR/GDP support.
*   **jax_privacy support**: ✅ Integrated in
    `truncated_amplified_bandmf_event()` and
    [`BandMFConfig`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.execution_plan.BandMFConfig.html).

(schuchardt-2026)=
### Schuchardt+ 2026: Sampling-Free Privacy Accounting for Matrix Mechanisms under Random Allocation

*   **Paper**: [Schuchardt & Kalinin, 2026](https://arxiv.org/abs/2601.21636)
*   **Strategy**: Random allocation with matrix mechanisms. Develops
    sampling-free (deterministic) privacy bounds via Rényi divergence
    (Section 3) and Conditional Composition (Section 4). **Lemma 3.2**
    computes the Rényi divergence for random-allocation matrix mechanisms;
    **Theorem 3.1** converts this to $(\varepsilon, \delta)$-DP. **Theorem 4.4** bounds the
    non-dominance probability for conditional composition. **Algorithm 3**
    completes the Rényi bound computation via dynamic programming; applicable
    to arbitrary banded and non-banded matrices.
*   **MF compatibility**: ✅ Central focus.
*   **dp_accounting support**: Not integrated.
*   **jax_privacy support**: Referenced in
    [`RandomAllocationSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.RandomAllocationSampling.html)
    docstring.
    Integration could be useful for random allocation + DP-MF users, though
    the tighter method is exponential in the number of bands.

(dong-2026a)=
### Dong+ 2026a: Privacy Amplification for BandMF via b-Min-Sep Subsampling

*   **Paper**: [Dong & Ganesh, 2026](https://arxiv.org/abs/2602.09338)
*   **Strategy**: $b$-min-sep subsampling — ensures any given example does not
    participate more than once within any window of $b$ iterations. Generalizes
    Poisson and balls-in-bins. **Section 4** defines $b$-min-sep subsampling.
    **Section 5** presents Monte Carlo accounting exploiting Markovian structure
    via dynamic programming. **Section 6** discusses multi-attribution extension
    with **Algorithm 3**. **Appendix B** covers extensions to non-banded
    matrices.
*   **MF compatibility**: ✅ Primary motivation is enabling privacy
    amplification for BandMF. Closes a substantial portion of the gap to
    theoretical lower bounds.
*   **dp_accounting support**: Not integrated; Monte Carlo accounting only.
*   **jax_privacy support**: ✅ Batch selection via
    [`BMinSepSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.BMinSepSampling.html).
    Accounting via Monte Carlo (as noted in docstring: "we only know how to
    analyze b-min-sep via Monte Carlo accounting").

(feldman-2026)=
### Feldman+ 2026: Efficient Privacy Loss Accounting for Subsampling and Random Allocation

*   **Paper**: [Feldman & Shenfeld, 2026](https://arxiv.org/abs/2602.17284)
*   **Strategy**: Demonstrates that the PLD of random allocation applied to any
    DP algorithm can be computed efficiently. Introduces **PLD realization** as
    a new accounting tool (Section 3). **Lemma 2.7** shows domination is
    preserved under random allocation transformation. **Theorem 4.4**
    characterizes the PLD of random allocation as convolution of exp-PLDs.
    **Theorem 4.6** provides the efficient computation algorithm with runtime
    $O(\log^3(t) \cdot \text{IQR}^2/\alpha^2)$. Shows privacy-utility tradeoff
    for random allocation is at least as good as Poisson subsampling.
*   **MF compatibility**: Discussed.
*   **dp_accounting support**: Not integrated; provides PLD-based techniques.
*   **jax_privacy support**: Referenced in
    [`RandomAllocationSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.RandomAllocationSampling.html)
    docstring.

(dong-2026b)=
### Dong+ 2026b: Less Random, More Private: What is the Optimal Subsampling Scheme for DP-SGD?

*   **Paper**: [Dong & Özgür, 2026](https://arxiv.org/abs/2605.07072)
*   **Strategy**: Proposes random allocation (which they call Balanced
    Iteration Subsampling / BIS, see also
    [Feldman+ 2025](#feldman-2025),
    [Dong+ 2025](#dong-2025))
    as an optimal alternative to Poisson subsampling. **Algorithm 1** formally
    defines BIS. **Section 3** proves BIS is optimal at both $\sigma \to 0$ and $\sigma \to \infty$
    extremes via a layered analysis: **Proposition 3.2** (low-noise privacy
    loss) shows participation variance dominates, **Proposition 3.3**
    (high-noise) shows uniform marginals dominate. **Section 4** introduces the
    near-exact Monte Carlo accountant with an $O(Tk)$ dynamic program
    (**Lemma 4.1** provides the $O(T)$ screening bound). Shows up to 9.6%
    reduction in required noise multiplier vs. Poisson.
*   **MF compatibility**: Discussed.
*   **dp_accounting support**: Not integrated.
*   **jax_privacy support**: ✅ Batch selection via
    [`RandomAllocationSampling`](https://jax-privacy.readthedocs.io/en/latest/_autosummary_output/jax_privacy.batch_selection.RandomAllocationSampling.html).
    Accounting not implemented.

---

## Future work

The following combinations are analyzed in the academic literature but not yet
fully supported in jax_privacy, and represent good candidates for future
integration:

1.  **Random allocation ($k$-out-of-$t$) + PLD accounting**:
    [Feldman+ 2025](#feldman-2025),
    [Schuchardt+ 2026](#schuchardt-2026),
    and
    [Feldman+ 2026](#feldman-2026)
    provide efficient PLD computation techniques for $k=1$, and a loose reduction
    from $k > 1$ to $k = 1$, that could be integrated into dp_accounting,
    enabling `get_true_positive_rates` and `get_gdp_parameter_estimate` for
    `RandomAllocationSampling`.
    [Dong+ 2026b](#dong-2026b)
    provides an exact Monte Carlo accounting scheme for $k \ge 1$ that could be added
    to jax_privacy's Monte Carlo estimation libraries.

2.  **Balls-in-bins + PLD accounting**:
    [Choquette-Choo+ 2024](#choquette-choo-2024)
    provides near-exact privacy parameters for arbitrary matrix mechanisms.
    Integrating this into dp_accounting would upgrade `BallsInBinsSampling`
    from Monte Carlo to full PLD support.

3.  **$b$-min-sep + PLD accounting**:
    [Dong+ 2026a](#dong-2026a)
    provides Monte Carlo analysis. Developing near-exact PLD-based analysis
    remains an open research question and would enable TPR/GDP methods for
    this strategy.
