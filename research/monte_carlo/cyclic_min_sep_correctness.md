<!--
Copyright 2026 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Correctness of the cyclic minimum-separation privacy loss

This note is self-contained. It assumes only familiarity with Monte Carlo
privacy accounting — the privacy loss of an output is the logarithm of an
expected likelihood ratio, and each iteration contributes a log-likelihood
ratio — and derives the combinatorics from scratch. All quantities are kept in
real (non-log) space; a single logarithm is taken at the very end.

## Setup and notation

Consider a training run of $$T$$ iterations, indexed $$0, 1, \dots, T-1$$. A
sensitive example participates in exactly $$K$$ of these iterations, subject to
a minimum separation $$b$$: any two iterations in which it participates are at
least $$b$$ apart. Fix an observed output $$y$$ of the mechanism.

For each iteration $$j$$, let $$\mathrm{llr}_j$$ be the *per-iteration
log-likelihood ratio* of $$y$$ — the log-ratio of the density of $$y$$ when the
example participates only at iteration $$j$$ to its density when it does not
participate at all. Its exponential

$$w_j = \exp(\mathrm{llr}_j) \ge 0$$

is simply the per-iteration *likelihood ratio*, the multiplicative evidence
that $$y$$ provides for participation at iteration $$j$$. We keep
$$\mathrm{llr}_j$$ abstract and do not inline its explicit formula — for the
Gaussian mechanisms of interest it has a closed form, but that form plays no
role in what follows — and work with the ratio $$w_j$$ because the derivation
below is entirely a product of these per-iteration ratios.

**Why a product suffices.** For a fixed participation set
$$A \subseteq \{0, \dots, T-1\}$$, the likelihood ratio of the whole observation
$$y$$ is the product $$\prod_{j \in A} w_j$$. This follows from the banded
structure that the minimum separation enforces: the mechanism is a
linear-Gaussian matrix mechanism whose sensitivity matrix is banded with
bandwidth at most $$b$$, so participations $$\ge b$$ apart perturb disjoint
blocks of the output. Their Gaussian log-likelihood ratios are then independent
and additive — equivalently, the ratios $$w_j$$ multiply. This decomposition is
the standard starting point of Monte Carlo privacy accounting for banded matrix
mechanisms, and we take it as given.

The participation set $$A$$ is random, and the privacy loss is the logarithm of
the expected likelihood ratio,

$$L(y) = \log \mathbb{E}_{A}\Big[\prod_{j \in A} w_j\Big].$$

**The cyclic law.** Arrange the $$T$$ iterations as the vertices of a cycle, so
position $$T-1$$ is adjacent to position $$0$$. Write a candidate participation
set as $$S = \{s_1 < s_2 < \dots < s_K\}$$, where each $$s_i$$ is a selected
position, and define its $$K$$ cyclic gaps — the distances between consecutive
selected positions around the cycle — by

$$g_i = s_{i+1}-s_i\ (1\le i<K),\quad g_K = T-s_K+s_1,$$

where $$g_K$$ wraps from the last selected position back to the first. The gaps
are positive and satisfy $$\sum_{i=1}^{K} g_i = T$$. Call $$S$$ *valid* if every
gap meets the minimum separation, i.e. $$g_i \ge b$$ for all $$i$$. The
mechanism draws $$A$$ uniformly at random from the set of all valid subsets;
write $$N$$ for the number of such subsets. (The name *random rotation* refers
to the uniformly random cyclic offset applied when sampling, which gives every
position the same marginal selection probability $$K/T$$.)

## Derivation sketch

Because $$A$$ is uniform over the $$N$$ valid subsets, the expectation is a
plain average, and

$$\exp L(y) = \frac{1}{N} \sum_{S \text{ valid}} \prod_{j \in S} w_j,$$

where the *weight* of a subset is the product of $$w_j$$ over its elements. This
splits the computation into two independent sub-problems:

- the **denominator** $$N$$ — a pure count of the valid subsets;
- the **numerator** $$\sum_{S \text{ valid}} \prod_{j \in S} w_j$$ — the total
  weight of all valid subsets, computed efficiently by conditioning on the
  smallest selected position.

Solving these (next two subsections) gives a closed form for the denominator and
a tractable sum for the numerator, which combine into

$$\exp L(y) = \frac{1}{N} \sum_{F=0}^{T-1} w_F\, Q(F),$$

where $$N$$ and the per-position quantity $$Q(F)$$ are derived below, and
$$L(y)$$ is its logarithm.

### Denominator: counting the valid subsets

Distinguish one selected position as an *anchor* and walk around the cycle from
it, recording the gaps $$g_1, \dots, g_K$$. These are $$K$$ positive integers
with each $$g_i \ge b$$ and $$\sum_i g_i = T$$ — that is, a composition of $$T$$
into $$K$$ ordered parts, each at least $$b$$. Substituting $$g_i = b + h_i$$
with $$h_i \ge 0$$ turns this into a composition of $$T - Kb$$ into $$K$$
nonnegative parts $$h_i$$, of which there are

$$\binom{(T-Kb)+(K-1)}{K-1} = \binom{T-K(b-1)-1}{K-1}.$$

There are $$T$$ positions for the anchor, and each valid subset is generated
exactly $$K$$ times (once per element taken as the anchor), so dividing out this
overcount gives

$$N = \frac{T}{K}\binom{T-K(b-1)-1}{K-1}.$$

As a check: $$T=5, K=2, b=2$$ gives $$N = \tfrac{5}{2}\binom{2}{1} = 5$$, the
five pairs on a $$5$$-cycle at cyclic distance at least $$2$$; and $$K=1$$ gives
$$N = T$$.

### Numerator: the total weight of valid subsets

Every valid subset has a unique smallest element
$$F = \min S \in \{0, \dots, T-1\}$$. Grouping the valid subsets by the value of
$$F$$ partitions the numerator with nothing counted twice:

$$\sum_{S \text{ valid}} \prod_{j \in S} w_j = \sum_{F=0}^{T-1} w_F\, Q(F),$$

where $$w_F$$ is the weight of the smallest selected position $$F$$ itself, and
$$Q(F)$$ is the total weight of all admissible ways to choose the remaining
$$K-1$$ positions given that the smallest one is $$F$$.

**Constraints on the completion.** Given that $$F$$ is the minimum, those
$$K-1$$ positions $$s_2 < \dots < s_K$$ are constrained by exactly the $$K$$
cyclic gaps:

- the first gap $$s_2 - F \ge b$$ forces $$s_2 \ge F+b$$, so every remaining
  position lies in $$\{F+b, \dots, T-1\}$$;
- the interior gaps make $$s_2, \dots, s_K$$ pairwise at least $$b$$ apart;
- the wrap gap $$T - s_K + F \ge b$$ forces the largest position to satisfy
  $$s_K \le T-b+F$$.

Hence $$Q(F)$$ is the total weight of the size-$$(K-1)$$ subsets of
$$\{F+b, \dots, T-1\}$$ whose elements are pairwise at least $$b$$ apart and
whose largest element is at most $$T-b+F$$.

**A reusable count.** For positions $$p \le u$$, let $$G(p, u)$$ denote the
total weight of all size-$$(K-1)$$ subsets of $$\{p, p+1, \dots, u\}$$ whose
elements are pairwise at least $$b$$ apart (each subset contributes the
product of its
weights; when $$K-1 = 0$$ the only subset is empty, with weight $$G = 1$$). This
quantity obeys a simple recurrence and is computable in linear time by sweeping
the lower end $$p$$ from $$u$$ downwards. With it,

$$Q(F) = G\big(F+b,\ \min(T-1,\ T-b+F)\big),$$

the upper limit being $$T-1$$ (no position exceeds $$T-1$$) capped by the wrap
constraint $$T-b+F$$. Two regimes arise:

- if $$F \ge b-1$$, then $$T-b+F \ge T-1$$: the wrap cap is not binding and
  $$Q(F) = G(F+b,\ T-1)$$ — the same computation for all such $$F$$;
- if $$F < b-1$$, then $$T-b+F < T-1$$: the wrap cap binds and
  $$Q(F) = G(F+b,\ T-b+F)$$, using the lowered upper limit.

Only the $$b-1$$ smallest values $$F \in \{0, \dots, b-2\}$$ require the lowered
limit, so evaluating the numerator costs $$O(b)$$ passes of the recurrence.

### Corner case: a single participation

When $$K = 1$$ the example participates once, its single cyclic gap equals
$$T \ge b$$ automatically, and every position is valid. Then $$A$$ is uniform
over all $$T$$ positions, the denominator is $$N = T$$, and

$$\exp L(y) = \frac{1}{T}\sum_{i=0}^{T-1} w_i.$$
