<!-- Thanks for contributing! -->

## Summary

<!-- One or two sentences on what changes and why -->

## Type

- [ ] Bug fix
- [ ] New feature (connector / extraction backend / pipeline stage)
- [ ] Tuning recipe for a new model (add to `docs/tuning/`)
- [ ] Documentation
- [ ] Refactor (no functional change)
- [ ] Methodology extension (RFC required — see CONTRIBUTING.md)

## Related Issue

Fixes #

## Test Plan

<!-- How you verified this change -->

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check src tests` passes
- [ ] If touching the dream pipeline: ran an end-to-end `dreamagent dream`
      against fixtures and confirmed the eval gate behavior

## Methodology Impact

<!-- If this changes the published DreamAgent methodology (the contract,
the gate logic, the rehearsal mix design, the extraction prompt rules), say
so explicitly. These changes update `docs/METHODOLOGY.md`. -->

- [ ] No methodology impact
- [ ] Methodology updated; `docs/METHODOLOGY.md` revised

## Checklist

- [ ] My code follows the project's style (`ruff` clean)
- [ ] I added tests where appropriate
- [ ] I updated docs for any user-visible behavior change
- [ ] I added a CHANGELOG entry if user-visible
- [ ] CI is green
