# Local CPU research resumption — 2026-11-17

Inspected before dependency installation or experiment execution. Source revision:
`d9253fb755e21d23927708bdab7b91e2c345b32d` on `research/cpu-efficiency-audit`.
Read `README.md`, `RESEARCH_PROGRAM.md`, the prior-art ledger overview, the pilot
protocol, runner, and evidence tests. No paid compute is authorized or used.

## Observed laptop

| Item | Observation |
|---|---|
| Machine | MacBook Pro, MacBookPro16,1 |
| CPU | Intel Core i9-9880H, 2.30 GHz, x86_64 |
| Cores | 8 physical, 16 logical |
| RAM | 16 GiB |
| OS | macOS 26.7.1, build 25G241 |
| Available disk | Approximately 379 GiB on the workspace volume |
| Python | Existing Python 3.11.0; no repository `.venv` present |
| Power | AC connected, reported 65 W; battery 42%, not charging |
| Power policy | Low Power Mode enabled for AC and battery |
| Thermal snapshot | No recorded thermal/performance warning; CPU speed limit 100% |
| Memory snapshot | 16,623.25 MiB swap used; about 4.44 GiB occupied by compressor |
| Memory-pressure query | System-wide memory free percentage reported as 43% |

Sources: `system_profiler SPHardwareDataType SPPowerDataType`, `sysctl`,
`sw_vers`, `df -h`, `vm_stat`, `memory_pressure -Q`, and `pmset -g therm`.
Hardware serial numbers and UUIDs are deliberately not retained here.
Swap occupancy does not by itself prove current active swapping. These are
point-in-time observations, not sustained-load measurements.

## Readiness findings

1. The archived six-configuration pilot is from a different, virtualized x86
   host. Its reported 21 passing tests have not been rerun on this laptop.
2. `compression_pilot.py`, `profile_operators.py`, and `summarize_pilot.py` use
   the archived pilot directory. The documented reproduction commands must
   not be used for a new laptop trial until output isolation is implemented.
3. The pilot worker converts `ru_maxrss` using `/1024`, appropriate for Linux
   KiB. macOS returns bytes and needs `/1024**2` for MiB. Fix and verify this
   before reporting laptop peak RSS.
4. The environment collector reads Linux `/proc` and cgroup files. Add native
   CPU, power, and memory metadata for meaningful laptop trial provenance.
5. `backend/requirements.txt` omits `onnx`, although the research runner imports
   it; the existing lock includes it. Compatibility of that lock with this
   Intel macOS/Python combination is unverified. Resolve an isolated local
   environment and retain its actual lock; do not silently replace the old one.
6. Low Power Mode and the substantial existing memory footprint may affect
   timing. Record the chosen power policy, pre/post memory and thermal state,
   and concurrent workload. Do not change system settings or terminate other
   applications automatically. Do not rank new laptop latency against old-host
   latency as a method improvement.

## Next bounded work

1. Make the research runners support separate, fresh output directories,
   propagate those paths to workers/profilers/summaries, and retain the archived
   evidence tests. Correct and test platform-dependent RSS accounting.
2. Check Intel macOS wheel availability, create a project-local environment,
   and record dependency versions. Start with CPUExecutionProvider only,
   two intra-op threads, one inter-op thread, and one worker at a time.
3. Run one existing 320x180 FP32 case as an engineering smoke check in a new
   directory before any full pilot. This is neither independent evaluation
   nor baseline parity. Preserve failures and avoid training or large sweeps.
4. Advance research gates 2 and 3: inspect an official lightweight baseline
   (IFRNet-S is a candidate), pin source/checkpoint hashes and rights, and
   establish native/export output parity before comparing speed. Confirm
   checkpoint access and CPU support before committing to that candidate.
5. Freeze separate development/test scene identities from an official benchmark
   before tuning; record pretraining-overlap uncertainty. Existing DAVIS scenes
   remain diagnostics. No independent dataset split has been created yet.

No dependencies were installed, experiments executed, system settings changed,
or archived pilot outputs overwritten during this preflight. The research
program remains exploratory; no learned contribution or competitive win has
been established.
