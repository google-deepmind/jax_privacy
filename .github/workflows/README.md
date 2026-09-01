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

# GitHub Actions Workflows

All third-party GitHub Actions used in these workflow files are pinned to full
40-character commit SHAs (with version comments) to protect against supply chain
attacks and comply with security policies (e.g.,
[Zizmor](https://docs.zizmor.sh/audits/#unpinned-uses)).

## Updating Action Versions

Future maintainers can update pinned actions using automated tools or manually.

### Option 1: Using Ratchet (Recommended)

[Ratchet](https://github.com/sethvargo/ratchet) is specifically designed to
manage and bump pinned GitHub Actions:

```bash
# Upgrade all pinned actions to their latest major/minor releases (e.g. v1 -> v3)
ratchet upgrade .github/workflows/*.yml

# Update pinned actions to the latest commit within their current tag constraint
ratchet update .github/workflows/*.yml

# Unpin all actions back to named tags
ratchet unpin .github/workflows/*.yml

# Pin named tags to their commit SHAs
ratchet pin .github/workflows/*.yml
```

### Option 2: Using Zizmor (Pinning & Remediation)

`zizmor --fix=all` resolves named tags (e.g. `@v4`) to their immutable commit
SHAs. It does not automatically bump major versions (e.g. `v4` → `v5`), but it
will remediate unpinned tags and known-vulnerable actions.

To upgrade versions using Zizmor:

1.  Change the action tag in the workflow file to the desired new release tag
    (e.g. `@v5`).
2.  Run `zizmor --fix=all` with an authenticated GitHub token to resolve the SHA
    and add the version comment:

```bash
zizmor --fix=all --gh-token=$(gh auth token) .github/workflows/
```

The zizmor documentation suggests setting up either Dependabot or Renovate to
automate this process. (https://docs.zizmor.sh/audits/#remediation_35).

### Option 3: Manual Updating

To manually find the commit SHA corresponding to a specific release tag, use
`git ls-remote`:

```bash
git ls-remote https://github.com/actions/checkout.git refs/tags/v4.2.2
```

Then update the workflow step format:

```yaml
- uses: actions/checkout@<FULL_COMMIT_SHA> # v4.2.2
```

## Security Guidelines

*   **Permissions:** Maintain least-privilege access by declaring explicit
    `permissions:` blocks (e.g., `permissions: { contents: read }`) at the top
    or job level.
*   **Credential Persistence:** Ensure `actions/checkout` steps include `with:
    persist-credentials: false` unless git push credentials are explicitly
    needed.
*   **Suppressions:** Any false-positive suppressions must be documented inline
    above the offending line using `# zizmor: ignore[<rule-name>] - {reason}`.
