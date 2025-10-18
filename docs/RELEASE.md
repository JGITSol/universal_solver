# Release Management

This project follows [Semantic Versioning](https://semver.org/) and keeps the authoritative version number in the top-level `VERSION` file.  The value is surfaced at runtime as `universal_solver.__version__`.

## Branching Strategy

- `base_proposal` is the default development branch.
- Feature branches follow the pattern `feature/<description>` or `fix/<description>`.
- Release branches use the format `release/<major>.<minor>.x` and are created from `base_proposal`.
- Tags are created from the release branch once validation is complete (e.g. `v0.2.0`).

## Release Checklist

1. Update `VERSION` with the new release number.
2. Update the changelog section in `docs/RELEASE_NOTES.md` (create if missing) or the GitHub Release draft.
3. Run the full validation locally:

   ```powershell
   pip install -r requirements-dev.txt
   pip install -e .
   pytest --cov=adv_resolver_math --cov-report=html
   pre-commit run --all-files
   ```

4. Ensure CI workflows are green for the release branch.
5. Merge the release branch into `base_proposal` (and `main` if maintained).
6. Create a signed tag `git tag -s v<major>.<minor>.<patch>` and push with `git push --tags`.
7. Publish release notes and distribute artifacts (PyPI, Docker, etc. when applicable).

## Post-Release

- Bump the `VERSION` file to the next development iteration (e.g. `0.2.1-dev.0`).
- Back-merge tagging changes into all active branches to keep history consistent.
- Update roadmap status and mark completed items.
