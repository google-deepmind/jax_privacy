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

# DP-FTRL and Matrix Mechanisms

This sub-package contains all the code necessary to define, optimize,
instantiate, and use correlated noise ("DP-FTRL") mechanisms.

TODO: b/329444015 - Provide a better overview of the package, and how it
integrates into the rest of `jax_privacy`.

## Notation and symbols

Important symbols and names are summarized in the table below.

The full name (left column) is generally used for kwargs in public API function,
whereas the single-letter or abbreviated names are often used in implementations
and internal functions to add readability of complex equations, and match the
mathematical notation used in the original papers and this README.

| Full Name / API Name | Symbol or short name | Description                                                                                                                                                                     |
|----------------------|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `workload`           | A                    | Workload matrix, for example `np.tri` (the lower-triangular matrix of ones) for prefix sums                                                                                    |
| `strategy_matrix`    | C                    | Strategy matrix, also sometimes referred to as the encoder matrix.                                                                                                               |
| `noising_matrix`     | $$C^{-1}$$ or `C_inv`| In a matrix factorization $$A = BC$$, this is the matrix $$C^{-1}$$; the noise added on iteration $$i$$ is given (conceptually by `(C_inv @ Z)[i, :]`, though the matrix multiplication and indexing are usually done implicitly in production implementations for efficiency. |
| `decoder_matrix`     | B                    | Decoder Matrix $$B$$ in the factorization $$A = B C$$                                                                                                                            |
| `n`                  | n                    | Number of iterations the mechanism supports, equal to the number of rows and columns in $$A$$.                                                                                  |
| `max_participations` | k                    | The number of epochs or maximum number of participations for one datum/user.                                                                                                    |
| `min_sep`            | b                    | Can be used for both num_bands (see Banded matrices, below) and min_sep (see minsep participation below).                                                                       |
| `sensitivity`        | sens                 | L2 sensitivity of $$C$$, with respect to some [participation pattern](#participation-schemas).                                                                                  |
| `error`              | error                | Variance (squared error) of a mechanism $$C^{-1}$$ under unit noise and regardless of sensitivity.                                                                             |
| `loss`               | loss                 | Total loss of the mechanism, including sensitivity. loss(C, B) = sens(C) ** 2 * error(B)                                                                                         |

## Types of loss and error.

By default, we operated on the squared losses/errors. The `per_query_error`
refers to the vector of errors introduced for each query (row) in the workload
matrix $$A$$. We often summarize this with an appropriate reduction from the
`per_query_error` to a scalar; the most common choices are `max_error` (the
maximum per-query error) and `mean_error` (the mean per-query error).
Particularly in papers, it is common to report `sqrt(mean_error)` which is also
referred to as `rmse` (root-mean-squared-error).

## Sensitivity and participation schemas {#participation-schemas}

The (L2) sensitivity of a matrix mechanism is a function of $$C$$, see
[MultiEpoch23] and [Banded23].

All definitions herein are with respect to an abstract notion of a privacy unit.
By appropriate translations, in particular, proper bundling of data (e.g.,
FedAvg style), per-example notions can be readily converted to per-user notions.
For simplicity, we may often use `example`, `user`, and `datum` interchangeably
with this understanding.

The sensitivity of a mechanism depends heavily on which iterations each example
participates in (that is, on which iterations a given example was trained on);
we compute sensitivity under a variety of different participation schemas:

| Name in code  | Full name            | Description                     |
| ------------- | -------------------- | ------------------------------- |
| (implicit)    | single-participation or single-epoch      | When each example participates at most once in all of training. If not specified, this is our default. See [SingleEpoch'22][SingleEpoch22].          |
| `min_sep`     | $$b$$ min-separation | Examples are seen at least $$b$$ steps apart. If an example participates in iterations $$i$$ and $$j$$, then $$\|i - j\| \ge b$$. Note that $$(k, b)$$-participation (below) is a special case. See [Banded'23][Banded23].          |
| `fixed_epoch` | $$(k, b)$$ participation        | There are exactly $$n = k \cdot b$$ iterations, and client participations are separated by multiples of exactly $$b$$ if a client participates on iterations $$i$$ and $$j$$, then $$\|i - j\|\ \text{mod}\  b = 0$$. See [MultiEpoch'23][MultiEpoch23].   |

## Parameterized matrix classes

Suitably parameterizing the $$C$$ and $$C^{-1}$$ matrices can both make the
computation of sensitivity and error more efficient, as well as enabling more
efficient online noise generation when the mechanism is deployed (for example,
in DP-FTRL). We consider a number of parameterized classes:

### General banded matrices

Implemented in [`banded.py`](banded.py), introduced in [Banded'23][Banded23].

We say a (general) matrix $$X$$ is $$b$$-banded if $$i,j \in [n], |i -j| \ge b$$
implies $$X_{[i,j]} = 0$$. While this is off-by-one from the typical definition
of bandwidth ($$X$$ has bandwidth $$b-1$$), our definition will be useful as it
will be natural to match $$b$$-banded matrices with $$b$$-min-separation
participation. Further, for $$b$$-banded lower-triangular matrices (which will
play a central role), $$b$$ intuitively gives the number of bands in the matrix.

### (Banded) Toeplitz Matrices

Implemented in [`toeplitz.py`](toeplitz.py). Structure $$C \in R^{n \times n}$$
lower-triangular Toeplitz matrix. Parameters n, the size of $$C$$ coef, the
non-zero Toeplitz coefficients of $$C$$. See [BandMF24].

TODO: b/329444015 - Markdown example where n = 4 and coef = [1, 0.5]) for
example.

### Buffered Linear Toeplitz (BLT) Matrices

Implemented in [`buffered_toeplitz.py`](buffered_toeplitz.py); BLT mechanisms
were introduced in [BLT24] and extended to multiple-participations in
[UseBLTs24].

### LTI Toeplitz Matrices

These matrices are parametrized by a single constant $$\nu$$ which defines the
recursive relationship for each new entry in a column. Thus, these matrices are
not optimized, though the constant $$\nu$$ can be swept and optimized w.r.t.
some error definition. These matrices are optimal w.r.t. a LTI error definition.
see the [LTI24].

## References {#refs}

1.  **[SingleEpoch22]** *Improved Differential Privacy for SGD via Optimal
    Private Linear Operators on Adaptive Streams.* NeurIPS 2022.

1.  **[MultiEpoch23]** *Multi-Epoch Matrix Factorization Mechanisms for Private
    Machine Learning.* Arxiv Nov 2022, ICML 2023.

1.  **[Banded23]** *(Amplified) Banded Matrix Factorization: A unified approach
    to private training.* NeurIPS 2023.

1.  **[FHU23]** *Constant matters: Fine-grained Complexity of Differentially Private Continual Observation*. ICML 2023.

1.  **[LTI24]** *Correlated Noise Provably Beats Independent Noise for
    Differentially Private Learning.*. ICLR 2024.

1.  **[BLT24]** *Efficient and Near-Optimal Noise Generation for Streaming
    Differential Privacy.* FOCS 2024.

1.  **[BandedSqrt24]** *Banded Square Root Matrix Factorization for Differentially Private Model Training*. NeurIPS 2024.

1.  **[BandMF24]** *Scaling up the Banded Matrix Factorization Mechanism for
    Differentially Private ML.* Arxiv 2024, ICLR 2025.

1.  **[UseBLTs24]** *A Hassle-free Algorithm for Private Learning in Practice:
    Don't Use Tree Aggregation, Use BLTs.* EMNLP Industry Track 2024

[SingleEpoch22]: https://arxiv.org/abs/2202.08312
[FHU23]: https://arxiv.org/abs/2202.11205
[MultiEpoch23]: https://arxiv.org/abs/2211.06530
[Banded23]: https://arxiv.org/abs/2306.08153
[LTI24]: https://arxiv.org/abs/2310.06771
[BLT24]: https://arxiv.org/abs/2404.16706
[BandedSqrt24]: https://arxiv.org/abs/2405.13763
[BandMF24]: https://arxiv.org/abs/2405.15913
[UseBLTs24]: https://arxiv.org/abs/2408.08868
