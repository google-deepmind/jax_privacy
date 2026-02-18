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

We welcome small patches related to bug fixes, but we do not plan to accept
major changes to this repository at this time.

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
as much as possible, with immutability and functional purity being a key guiding
principles we adhere to across our library. Below are some more detailed
conventions depending on what is being contributed.

1.  **Public Facing Functions**: Public facing functions are those that are
    exposed to the **users** of JAX Privacy (usually surfaced via
    \_\_init\_\_.py). Public facing functions and classes should **always** have
    full docstrings, type annotations, and example usages in the form of
    [doctests](https://docs.python.org/3/library/doctest.html). Doctests
    provides useful documentation that stays up-to-date with code changes, and
    is a useful litmus test on the simplicity and usability of the API surface.

1.  **Internal functions used across files**: For maintainability of the core
    library, it is sometimes beneficial to define a function in one file and
    have it be used by another file within the jax_privacy package. These
    functions are not intended to be consumed by JAX Privacy users (although may
    be encountered by developers / contributors). These functions should
    generally have descriptive names, type annotations. Internal functions
    should have a one-line docstring explaining what they do. A full docstring
    is encouraged if the function has non-obvious side effects, complex
    arguments, or implements a multi-step algorithm that isn't clear from the
    code alone.

1.  **File-local private functions**: These function should always have a
    leading "_". This signals to developers that the function is not part of the
    public API and is subject to change without notice. These functions should
    have 1-line docstrings; type annotations are optional and context-dependent.
    Example usages are not needed as they can be found in the corresponding
    _test.py file.

1.  **Nested functions**: Functions defined within other functions should
    generally be as simple as possible; we prefer to keep the boilerplate
    minimal on these (no docstrings + type annotations), inline comments can be
    used, but should be used sparingly.

## Linting and testing

We use `flake8`, `pylint` and `pytype` for linting and type checking. Please run
the following commands locally before submitting a pull request:

```bash
$ flake8 jax_privacy/**.py
$ pylint jax_privacy/**.py
$ pytype jax_privacy/**.py
```
