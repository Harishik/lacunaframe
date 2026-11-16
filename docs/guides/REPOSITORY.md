# Where everything lives

[Documentation map](../README.md) · [Architecture](ARCHITECTURE.md)

The layout separates application code, pinned inputs, measured experiments, and explanations. Existing natural/checkpoint paths remain stable so reports and scripts retain a clear connection. Earlier synthetic runs and interface screenshots are preserved in named archive folders.

```text
lacunaframe/
├── README.md                    visual front door
├── LICENSE                      MIT for original code
├── CHANGELOG.md                 release history
├── CONTRIBUTING.md              local verification and research rules
├── src/
│   ├── App.tsx                  page shell / navigation
│   ├── components/              library, viewer, inspector, evaluation, project
│   └── lib/                     typed API and workbench state
├── backend/
│   ├── app.py                   API, decoding, jobs, local storage and exports
│   ├── core.py                  damage, classical restoration and metrics
│   ├── learned.py               pinned RIFE 4.9 CPU adapter
│   ├── candidate.py             optional RIFE 4.26 research adapter
│   ├── benchmark*.py            synthetic and natural study runners
│   └── tests/                   pytest suite
├── scripts/                     setup, profiling, comparison and verification
├── tests/                       browser, accessibility and documentation checks
├── assets/
│   ├── video/                   three original pinned DAVIS mirror clips
│   ├── models/                  both pinned ONNX exports
│   ├── manifest.json            default checkpoint and video hashes/URLs
│   ├── candidate.json           candidate checkpoint hash/URL
│   └── font-licenses/           local typography notices
├── examples/
│   ├── natural-v02/             18 conditions / 90 method runs / 108 MP4s
│   ├── checkpoint-v03/          18 conditions / 36 method runs / 12 MP4s
│   ├── external-pack-v03/       lossless inputs, targets and scored prediction
│   ├── synthetic-v01/           all earlier diagnostic experiments
│   └── interface-history/       all earlier UI captures
├── public/
│   ├── media/                  small actual previews used by the app
│   └── reports/                frontend copies of measured summaries/licenses
├── docs/
│   ├── README.md                guided map / reading routes
│   ├── guides/                  setup, walkthrough, protocol, architecture, map
│   ├── experience/              standalone interactive HTML field guide
│   ├── visuals/                 reproducible GIFs, plots and comparison frames
│   ├── design/                  current browser screenshots
│   ├── verification/            captured local checks and limitations
│   ├── REPORT.md                measured findings
│   ├── RELATED_WORK.md          primary-source research comparisons
│   ├── RESEARCH.md              mathematical/implementation conventions
│   ├── DESIGN.md                interface rationale
│   └── RUNBOOK.md               full operations reference
├── .github/                     owner mapping and inactive verification template
├── Dockerfile                  optional non-root backend image
└── package*.json / tsconfig...   frontend dependencies and strict build settings
```

## What “all the files” includes

All authored source, dependency manifests/locks, model exports, pinned media, synthetic and natural studies, candidate comparisons, lossless exchange arrays, reports, CSVs, plots, screenshots, animations, benchmark logs and verification evidence are versioned. No Git LFS or external artifact bucket is needed to recover these files from this repository.

Installed `node_modules`, `.venv`, build caches, bytecode, local environment secrets, transient uploads/jobs and Playwright scratch files are excluded. They are reproducible installation/runtime state, not experiment evidence. Reusable screenshots and verification outputs are retained explicitly under `docs/` and `examples/`.

`docs/verification/repository-inventory.json` lists the versioned payload and hashes at the documented publication point, excluding Git metadata and the inventory itself. Later commits can regenerate it. A committed working tree can be checked using `git status` and `git ls-files`.
