# Publishing to PyPI

This package uses GitHub Actions + PyPI Trusted Publishers (OIDC) for releases. No API tokens are stored in repository secrets; PyPI verifies the GitHub Actions run via short-lived OIDC tokens at publish time.

Workflow file: [`.github/workflows/publish.yml`](../.github/workflows/publish.yml).

## One-time setup

The workflow will not actually publish anything until two things exist on the PyPI side and the GitHub Environments are configured. **Do this once per maintainer; the configuration persists across releases.**

### 1. TestPyPI trusted publisher

Create a TestPyPI account if you don't have one: <https://test.pypi.org/account/register/>.

Go to <https://test.pypi.org/manage/account/publishing/> and click *Add a new pending publisher*. Fill in:

| Field | Value |
|---|---|
| PyPI Project Name | `dfxm-geo` |
| Owner | `borgi-s` |
| Repository name | `Geometrical_Optics_master` |
| Workflow name | `publish.yml` |
| Environment name | `testpypi` |

### 2. PyPI trusted publisher

Same form on <https://pypi.org/manage/account/publishing/>, but with `Environment name = pypi`.

### 3. GitHub Environments

In the repo: **Settings → Environments → New environment**.

- **`testpypi`** — no protection rules. Tag-driven runs deploy here automatically. Optionally add a deployment branch restriction to `main`.
- **`pypi`** — add a **required reviewer** (yourself) under *Deployment protection rules*. This means every PyPI publish requires you to manually approve the run in the Actions UI. Same branch restriction.

## Publishing a release

Once the one-time setup is in place:

```bash
# 1. Bump version in pyproject.toml. SemVer:
#    PATCH  e.g. 1.1.0 -> 1.1.1  (bug fixes only)
#    MINOR  e.g. 1.1.1 -> 1.2.0  (new features, backward-compatible)
#    MAJOR  e.g. 1.2.0 -> 2.0.0  (breaking changes)

# 2. Commit, push to main.
git add pyproject.toml
git commit -m "chore: bump version to 1.1.1"
git push

# 3. Tag.
git tag -a v1.1.1 -m "v1.1.1 — fix Foo, improve Bar"
git push origin v1.1.1
```

The push of the tag triggers the workflow:

1. **`build`** — installs `build` + `twine`, verifies the tag matches `pyproject.toml`'s `version` field, builds wheel + sdist, runs `twine check`, uploads the dist as an artifact.
2. **`publish-testpypi`** — downloads the artifact and uploads to TestPyPI. No approval needed.
3. **`publish-pypi`** — waits for manual approval (from the `pypi` environment's required reviewer), then uploads to PyPI.

After publish, verify:

```bash
pip install dfxm-geo
dfxm-bootstrap --help
```

## Manual / dry-run publishes

The workflow also accepts `workflow_dispatch` (manual trigger via the Actions UI). Inputs:

- `target = testpypi` — builds and publishes to TestPyPI only. Useful for a no-tag dry-run on `main` (e.g. after a metadata refactor) to confirm the build still produces a valid wheel.
- `target = pypi` — builds and publishes directly to PyPI, skipping TestPyPI. Only use this for republishes or recovery; the normal flow is a tag push.

## Troubleshooting

- **`Tag X does not match pyproject.toml version Y`** — the verify step bails. Either re-tag at the right version or bump `pyproject.toml` and re-tag.
- **`Forbidden: This project is not configured with Trusted Publisher`** — the one-time PyPI setup hasn't been done for this project + workflow combination. See sections 1-2 above.
- **`PyPI publish is waiting for approval`** — expected. Go to Actions → the running workflow → click *Review deployments* → approve.
- **`File already exists`** — the version was previously published. PyPI does not allow overwriting; bump `version` (PATCH bump is fine for re-issuing the same release).

## Why Trusted Publishers, not API tokens?

- **No secrets in the repo.** Compromising the repository doesn't compromise the PyPI account.
- **Scoped to the exact workflow.** A PyPI token, once leaked, could publish anything. The OIDC trust binding is `(owner, repo, workflow, environment)` — even a leaked OIDC token from a different workflow won't authorize a publish.
- **Recommended by PyPI** since 2023 and the only approach for new projects on PyPI's roadmap.

Reference: <https://docs.pypi.org/trusted-publishers/>
