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

# How to Contribute

We welcome contributions, particularly bug fixes and documentation
improvements. For larger changes, please open an issue first to discuss the
approach.

## Coordination & Claiming Issues

To avoid duplicate effort and ensure your work can be merged, please follow
these steps:

*   **Check for existing PRs and Issues:** Before starting work, search the
    issue tracker and active PRs to see if the feature/bug is already being
    addressed.

*   **Claim the issue:** If an issue exists, comment on it expressing your
    intent to work on it (e.g., "I'd like to work on this"). A maintainer will
    then assign it to you.

*   **Wait for Assignment:** Do not start a large-scale implementation until a
    maintainer has acknowledged your comment. This prevents multiple people from
    working on the same fix simultaneously.

*   **Stale Assignments:** If an assigned issue hasn't seen progress or
    communication for 14 days, the assignment may be cleared to allow others to
    contribute.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Style guide

Code in this library generally follows the
[Google Style Guide](https://google.github.io/styleguide/pyguide.html). We aim
to keep APIs, names, and design patterns in line with the broader JAX ecosystem
as much as possible, with immutability and functional purity being key guiding
principles we adhere to across our library. Below are some more detailed
conventions depending on what is being contributed.

1.  **Public Facing Functions**: Public facing functions are those that are
    exposed to the **users** of JAX Privacy (usually surfaced via
    `__init__.py`). Public facing functions and classes should **always** have
    full docstrings, type annotations, and example usages in the form of
    [doctests](https://docs.python.org/3/library/doctest.html). Doctests provide
    useful documentation that stays up-to-date with code changes and serve as a
    useful litmus test on the simplicity and usability of the API surface. See
    the [Python documentation](#python-documentation) section below for more
    details on writing Python docstrings.

1.  **Internal functions used across files**: For maintainability of the core
    library, it is sometimes beneficial to define a function in one file and
    have it be used by another file within the `jax_privacy` package. These
    functions are not intended to be consumed by JAX Privacy users (although
    they may be encountered by developers / contributors). These functions
    should generally have descriptive names and type annotations. Internal
    functions should have a one-line docstring explaining what they do. A full
    docstring is encouraged if the function has non-obvious side effects,
    complex arguments, or implements a multi-step algorithm that isn't clear
    from the code alone.

1.  **File-local private functions**: These functions should always have a
    leading underscore (`_`). This signals to developers that the function is
    not part of the public API and is subject to change without notice. These
    functions should have 1-line docstrings; type annotations are optional and
    context-dependent. Example usages are not needed as they can be found in the
    corresponding `_test.py` file.

1.  **Nested functions**: Functions defined within other functions should
    generally be as simple as possible; we prefer to keep the boilerplate
    minimal on these (no docstrings + type annotations). Inline comments can be
    used, but should be used sparingly.

### Naming conventions

*   **Function Naming Conventions (`fn` vs. `fun`)**: When adding or refactoring
    code in `jax_privacy`, adhere to the following function abbreviation
    guidelines:

    -   **Higher-Order Transformation Inputs (`fun`):** Use `fun` as the
        positional parameter name for functions passed into top-level JAX-style
        function transformations (e.g., `clipped_grad(fun, ...)`,
        `clipped_fun(fun, ...)`). *Rationale:* Aligns directly with core
        upstream JAX primitive signatures (`jax.grad(fun)`, `jax.vmap(fun)`,
        `jax.jit(fun)`).

    -   **Compound Identifiers, Protocols, Types, and Arguments (`fn`):** Use
        `fn` as the suffix or abbreviation for all compound function names, type
        names, and domain-specific arguments.

        -   **Protocols & Types:** `LossFn`, `CallbackFn`, `AccountantFn`,
            `_NoiseStructureFn`.
        -   **Arguments & Attributes:** `loss_fn`, `accountant_fn`,
            `reduction_fn`, `extract_preconditioner_from_state_fn`.
        -   **Internal Callables:** `grad_fn`, `update_fn`, `scan_fn`, `map_fn`,
            `partition_fn`.

(documentation-guide)=
## Documentation guide

(markdown-documentation)=
### Markdown documentation

Standalone documentation files (`.md` files in `docs/` and the repository root)
are rendered on ReadTheDocs using Sphinx with the **MyST** parser
([`myst-parser`](https://myst-parser.readthedocs.io/) / `myst_nb`), rather than
plain GitHub-Flavored Markdown. When contributing Markdown documentation:

*   **Markdown engine (MyST):** MyST extends CommonMark with Sphinx features.
    Standard CommonMark syntax (tables, bold, lists, code blocks) works as
    expected.
*   **Mathematical notation in Markdown (`.md` files):** Use dollar-delimited
    math (`$math$` for inline math and `$$math$$` for display block equations).
    Do **not** use LaTeX `\(...\)` or `\[...\]` delimiters in Markdown files.
    Actively format mathematical parameters and variables in math mode (e.g.,
    `$(\epsilon, \delta)$-DP`, `$\epsilon$`, `$\delta$`, `$\mu$-GDP`,
    `$\epsilon' = \epsilon + \text{tiny}$`, `$C$`, `$\eta$`,
    `$B \times L \times 4$`, `$L = 1024$`). *(Note: In Python docstrings, use
    the `:math:` rST role instead of `$`).*

*   **Callouts and admonitions:** For documentation files rendered with Sphinx /
    MyST on ReadTheDocs, **always use native MyST directive callouts** (such as
    ```` ```{note} ````, ```` ```{tip} ````, ```` ```{important} ````,
    ```` ```{warning} ````, or ```` ```{caution} ````):

    ````markdown
    ```{note}
    Callout text goes here.
    ```
    ````

    Do **not** use GitHub-style blockquote alerts (e.g., `> [!NOTE]`,
    `> [!WARNING]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!CAUTION]`), as the MyST
    Sphinx parser does not recognize them as directives and will render them as
    plain blockquotes with literal `[!NOTE]` text.

*   **Linking to other documentation pages:** Prefer the standard and common
    Markdown link format `[Document Title](doc_name)` (e.g.,
    `[Overview](overview)` or `[Keras API](keras_api)`). MyST Sphinx
    automatically resolves relative Markdown document links.

*   **Heading anchors (only when referenced):** Add an explicit MyST target
    label directly above a heading **only if that specific section is actually
    referenced or linked to** (e.g. from an in-document index or another
    document). Do not add target labels to every heading by default. For
    page-level links, use standard Markdown links `[Title](page_name)`.

    ```markdown
    (my-section-id)=
    ## My Section Title

    See [My Section Title](#my-section-id) for details.
    ```

#### Symbol cross-references using Sphinx roles in Markdown

In Markdown documentation files (`.md`), MyST supports full Sphinx and
Intersphinx symbol cross-referencing via `` {role}`target` `` syntax. **We
strongly recommend using these cross-reference roles instead of bare unlinked
code** (e.g., instead of `` `clipped_grad` ``) for all public functions,
classes, methods, and modules. This generates clickable, verified hyperlinks to
API reference documentation in ReadTheDocs HTML:

*   **Functions & Methods:** `` {func}`~jax_privacy.clipping.clipped_grad` `` or
    `` {meth}`~jax_privacy.training.DPTrainer.train_step` ``

*   **Classes:** `` {class}`~jax_privacy.execution_plan.DPExecutionPlan` ``

*   **Modules:** `` {mod}`jax_privacy.matrix_factorization` ``

*   **Intersphinx (JAX/Optax):** `` {func}`jax.grad` `` or
    `` {class}`optax.GradientTransformation` ``

Prefixing the target with a tilde (`~`) displays only the short name of the
symbol (e.g. `clipped_grad`) in the rendered text while linking to the fully
qualified target page.

*(Note: In Markdown files, use curly braces `` {role}`target` ``; in Python
docstrings, use colons `` :role:`target` ``).*

(python-documentation)=
### Python documentation

Python docstrings are rendered on ReadTheDocs using Sphinx and Napoleon, which
parse docstrings as **reStructuredText (rST)** rather than Markdown. Do not use
Markdown syntax (e.g. pipe tables or Markdown code blocks) inside Python
docstrings. See the
[Sphinx rST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html),
the
[Docutils rST Quick Reference](https://docutils.sourceforge.io/docs/user/rst/quickref.html),
and the
[Sphinx Napoleon Example Google-Style Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

#### Cross-references and code snippets in docstrings (using backticks)

*   **Single backticks with Sphinx roles** for **symbol cross-references**: Use
    explicit Sphinx roles (`:func:`, `:class:`, `:meth:`, `:attr:`) to
    cross-reference Python symbols in `jax_privacy`, `jax`, and `optax`. This
    creates clickable hyperlinks in the rendered documentation. For example:

    ```python
    """See :class:`~jax_privacy.clipping.BoundedSensitivityCallable`."""
    """Similar to :func:`jax.grad`."""
    ```

    Always use **fully qualified names** (including the package path) in
    cross-references. Unqualified names (e.g. ``:func:`clipped_grad` ``) resolve
    inconsistently and should be avoided. Use the `~` prefix to display only the
    short name: ``:func:`~jax_privacy.clipping.clipped_grad` `` renders as just
    `clipped_grad()`.

    **Cross-referencing guidance:**

    -   **`jax_privacy` symbols:** Unless there is a good reason not to,
        *always* cross-reference `jax_privacy` symbols and display the short
        name.

    -   **Symbols outside of `jax_privacy`:** Avoid cross-referencing very
        common symbols like `jax.Array`, but *do cross-reference* any symbol
        where the user might need to fully understand the underlying
        functionality to understand the documentation.

*   **Double backticks** (` ``foo`` `) for **literals and parameters**: Use
    double backticks for parameter names, argument names, variables, boolean
    flags (`True`/`False`/`None`), literal numbers, and short code expressions.
    In rST, double backticks render as inline code (monospace).

(using-math-in-python-docstrings)=
#### Using math in Python docstrings

Given the mathematical foundations of `jax_privacy`, it is often useful to
include equations in documentation. In Python docstrings, equations are written
using Sphinx rST math syntax:

*   **Inline math:** Use the `:math:` role with backticks, e.g., ``:math:`A =
    \pi r^2` `` or ``:math:`\sigma > 0` ``.

*   **Block / display math:** Use the `.. math::` directive, preceded and
    followed by a blank line and indented by 4 spaces:

    ```python
    def my_math_function():
      r"""Calculates gravitational force between two masses.

      For inline math, use the :math: role with backticks:
      Calculates :math:`A = \pi r^2`.

      For block/display math, use the .. math:: directive.
      Leave a blank line and indent the equation:

      .. math::

          F = G \frac{m_1 m_2}{r^2}
      """
      pass
    ```

```{important}
**Always use raw docstrings (`r"""..."""`)** when including LaTeX equations so
Python does not interpret backslashes as escape characters (e.g., treating
`\theta` as a tab or `\sigma` as an escape).

In Python docstrings (`.py` files), do **not** use dollar signs (`$x$` or
`$$...$$`) for math. Dollar-delimited math is supported in Markdown (`.md`)
files under `docs/` via the MyST parser, but Python docstrings are parsed as
reStructuredText (rST), which requires the `:math:` role and `.. math::`
directive.
```

#### Hyperlinks and paper citations in Python docstrings

*   **rST hyperlink format:** Avoid bare long URLs in docstrings. Format
    external links as clickable hyperlinks with concise anchor text using rST
    syntax: `` `Anchor Text <https://example.com>`_ `` (do not use Markdown
    `[Text](url)` format, which does not render in rST).

*   **arXiv and paper references:** When referencing papers in docstrings, use
    the **author names and year** as the clickable link text:

    ```python
    """See `Denisov et al. (2022) <https://arxiv.org/abs/2202.08312>`_."""
    ```

## Linting and testing

We use `flake8`, `pylint`, and `pyrefly` for linting and type checking.
Please run the following commands locally before submitting a pull request:

```bash
$ flake8 jax_privacy/**.py
$ pylint jax_privacy/**.py
$ pyrefly check
```
