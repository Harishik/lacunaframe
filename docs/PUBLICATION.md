# Private repository publication status

The complete project is prepared locally for **Harishik/lacunaframe**, with a reviewed Git payload, a local commit and a portable Git bundle. **A remote repository has not been created or pushed.**

The connected GitHub account is `Harishik`. Both creation paths were rejected:

- GitHub CLI / GraphQL: `Resource not accessible by integration (createRepository)`.
- REST `POST /user/repos`: HTTP 403, `Resource not accessible by integration`.

This is an integration-permission restriction, not a request for new permission to publish from the owner. The owner already authorized private creation and upload. No token or credential is stored in the project.

## Complete publication

Either grant the connected GitHub integration repository-creation access, or create an **empty private** `Harishik/lacunaframe` repository and grant this integration access to it. Do not initialize a README/license on GitHub, since the prepared commit already contains them. No deployment or paid service is required.

Then run with an appropriately authorized GitHub login:

```bash
bash scripts/publish_private.sh
```

The script verifies the authenticated owner, creates only a private repository if absent, checks visibility, disables repository Actions, verifies that setting, checks the payload inventory, and pushes `main` without force. It refuses to replace an unrelated `origin` and verifies the remote commit afterward. It does not enable CI, deploy, or change a public repo to private implicitly.

## Recover from the portable bundle

The prepared `lacunaframe-private-ready.bundle` is stored beside the working repository, not duplicated inside it. It contains all committed source, assets, documentation, binary checkpoints and experiment evidence. It omits installed dependencies, temporary uploads and secrets.

```bash
git clone /path/to/lacunaframe-private-ready.bundle lacunaframe
cd lacunaframe
# Set up dependencies following docs/guides/QUICKSTART.md.
```

The payload inventory in `docs/verification/repository-inventory.json` can be checked with `.venv/bin/python scripts/repository_inventory.py --check` after setup.
