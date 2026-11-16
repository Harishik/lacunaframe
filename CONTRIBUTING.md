# Working on Lacunaframe

Start with the [architecture](docs/guides/ARCHITECTURE.md) and [experiment protocol](docs/guides/EXPERIMENTS.md). The local workflow needs Python 3.12, Node 20.19+, CPU and no API keys.

## Keep the scientific boundary intact

- Restoration receives damaged pixels, explicit missing flags and masks. Never pass evaluator targets into a solver.
- Preserve timestamps and observed pixels for temporal interpolation. Make fallbacks and unsupported capabilities explicit.
- Keep actual measurements, including regressions. A faster or newer model is not automatically a quality improvement.
- Label synthetic controls, encoded previews, unknown training overlap and metric scope. Do not compare unmatched publication headline numbers.
- Pin outside assets by revision and hash; preserve their licenses. Code's MIT license does not relicense DAVIS footage.

## Verify locally

```bash
.venv/bin/python -m pytest -q
npm run build
npm run lint
npm run typecheck
npx playwright install chromium
npm run test:browser
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/check_documentation.py
```

Playwright starts the local frontend/backend if absent and reuses healthy existing servers. Do not run two suites against the same temporary backend store concurrently. Benchmark with a task-specific `LACUNA_DATA`; benchmark runners update their study output directories.

The interactive guide's browser test checks local-file rendering, slider/charts, image loading, responsive overflow, reduced-motion behavior and automated accessibility. The documentation checker checks local file links and experiment provenance. Rebuild visuals with `scripts/build_documentation.py` after changing their source arrays or summaries.

The publishing script requires a private repo and disables GitHub Actions before pushing to avoid automatic costs. The workflow definition is manual-only. Do not enable paid jobs, deploy, publish, change visibility, or expose uploaded media without the owner's authorization.

## Make a review easy

Describe the user-visible result and why it is needed. Include the relevant local checks and any changed model/data protocol. Add meaningful tests for restoration, alignment, evaluation or state behavior; avoid tests that merely repeat an implementation. Preserve unrelated work and historical evidence.
