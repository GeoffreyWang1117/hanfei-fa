# Releasing

One tag publishes both PyPI distributions built from this repository:
`merkle-weight-verify` (primary) and `hanfei-fa` (identical copy under the
HanFei series name). `scripts/build_dists.py` explains why the second name is
a full copy rather than an alias.

## One-time setup (PyPI Trusted Publishing)

1. On PyPI, for **each** of the two projects, open
   *Manage project → Publishing → Add a new publisher → GitHub* and enter:
   - Owner: `GeoffreyWang1117`
   - Repository: `hanfei-fa`
   - Workflow: `publish.yml`
   - Environment: `pypi`
2. In the GitHub repository settings, create the environment `pypi`
   (optionally with required reviewers), then set the repository variable:

   ```bash
   gh variable set PYPI_PUBLISH --body true --repo GeoffreyWang1117/hanfei-fa
   ```

Until the variable is set, tag pushes still build and check the
distributions but skip the upload.

## Each release

1. Bump `version` in `pyproject.toml` and `__version__` in
   `merkle_verify/__init__.py`; date the entry in `CHANGELOG.md`.
2. Make sure CI on `main` is green.
3. Tag and push:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

The workflow refuses to publish if the tag does not match both version strings.
