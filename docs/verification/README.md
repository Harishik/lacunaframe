# Verification evidence

The text/JSON files in this directory are captured from the actual local commands. Browser screenshots are in `../design/`.

- `pytest.txt`: core restoration, metric, upload/API, learned CPU, candidate ABI and external alignment tests.
- `frontend.txt`: production build, lint and strict TypeScript checks.
- `browser.txt`: Playwright flow, keyboard/mobile/offline and Axe accessibility checks.
- `artifacts.json`: source/checkpoint checksums, 120 fully decoded MP4s, 36 aligned condition reports and summary agreement.
- `docker-build.txt`, `docker-smoke.txt`: successful image build and the explicit container-start failure caused by the local disk limit. Container runtime verification remains incomplete; native API/CPU inference is verified by pytest and Playwright.

The first Docker build encountered this environment's private certificate authority and storage-layer duplication. The successful build uses the optional BuildKit certificate secret (`--secret id=pip_ca,src=/etc/ssl/certs/ca-certificates.crt`) without disabling TLS verification or storing the host trust bundle in the image. Copying application assets before dependency installation avoids repeated large dependency layers on the local VFS driver. Normal hosts with public trust roots can omit the secret.

The Python suite emits one upstream Starlette/httpx deprecation warning. It does not affect the tested results. Browser log timings and downloaded reports describe this run, while the saved natural study keeps its original timings.

## Private repository documentation edition

`documentation.json` checks local links, embedded summary equality, the actual 40-frame GIF and worst-frame selection. The browser suite now includes an offline interactive-guide test with zero HTTP requests, functional comparison and chart controls, reduced-motion behavior, mobile overflow checks and Axe WCAG A/AA analysis. `repository-inventory.json` records the versioned payload and hashes (excluding itself). Historical synthetic and UI artifacts have moved into labeled archive folders without removing evidence.

Publication is blocked by GitHub integration permissions; `github-publication.json` records the actual GraphQL/REST denials. No remote repository creation, push or Actions-setting mutation succeeded. `documentation-browser.txt` records the final targeted guide test after the publication-status wording update. The complete application suite immediately before it passed five tests.
