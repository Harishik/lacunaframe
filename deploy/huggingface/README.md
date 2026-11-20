---
title: LACUNAFRAME
emoji: 🎞️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
license: mit
short_description: Remove video frames, reconstruct them on CPU, measure failures
---

# LACUNAFRAME: live demo

The Space card for a Hugging Face Docker Space. **Nothing has been deployed.** Publishing makes the demo public, so it is the owner's decision.

## Deploy (owner action)

1. Create a new Space with the **Docker** SDK on the free **CPU basic** hardware.
2. Push the repository contents to it, with this file as the Space's root `README.md` and `deploy/huggingface/Dockerfile` as the root `Dockerfile`.
3. The container builds the frontend, installs the pinned CPU stack, and serves app and API on port 7860.

## Before going public

- DAVIS clips and derived media are **CC BY-NC 4.0**. A free, non-commercial demo is compatible; keep the attribution page reachable.
- Uploads are processed in a temporary directory and expire after one hour. The app is single-user by design, so concurrent visitors share one CPU worker.
- Free Spaces sleep when idle, so the first request after sleep is slow.
