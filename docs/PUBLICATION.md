# Private repository publication

Target: **[Harishik/lacunaframe](https://github.com/Harishik/lacunaframe)**. The owner created this private repository; authenticated access confirms private visibility and push permission. The owner’s original README/MIT-license commit is preserved in the merged history. All authored project commits use Harishik Dev Singh Jamwal; the owner’s GitHub-created initial commit uses the Harishik account identity. `.mailmap` connects those two identities in local author summaries. No automated-agent author or co-author is added.

The release includes the complete source, pinned assets/checkpoints, measured experiment archives, lossless comparison arrays and interactive documentation. The machine-readable verification record is [github-publication.json](verification/github-publication.json).

## Publish subsequent changes

```bash
bash scripts/publish_private.sh
```

The script verifies the authenticated owner and private visibility, checks the payload inventory, refuses active GitHub workflow files and an unrelated `origin`, pushes `main` without force, and compares the resulting remote commit with local HEAD.

Repository Actions settings remain inaccessible to the integration (HTTP 403). The verification definition is therefore retained as the inactive **`.github/verification.workflow.yml`** template, outside `.github/workflows`. This upload does not dispatch a runner, activate a workflow or deploy a service. The template can be reviewed and activated separately only with the owner’s authorization.

## Verification trail

Initial repository creation through the integration was denied by GraphQL and REST. The owner then created the repository directly, resolving that blocker. Publication preserves both the original owner commit and the prepared local release history. Required upstream asset licenses/citations remain separate from sole project authorship.

The portable `lacunaframe-private-ready.bundle` beside the working repository can be used as a full offline Git backup. It contains source, checkpoints and experiments, excluding installed dependencies, private uploads and secrets.

```bash
git clone /path/to/lacunaframe-private-ready.bundle lacunaframe
cd lacunaframe
# Follow docs/guides/QUICKSTART.md for dependency setup.
```
