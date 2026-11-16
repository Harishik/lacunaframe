#!/usr/bin/env bash
# Publish only to the owner's private repository; never force-push or enable Actions.
set -euo pipefail
cd "$(dirname "$0")/.."
repository="${1:-Harishik/lacunaframe}"
if [[ "$repository" != 'Harishik/lacunaframe' ]]; then
  echo 'This prepared release targets Harishik/lacunaframe. Review the script before choosing another destination.' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Commit or preserve local changes before publishing.' >&2
  exit 1
fi
if [[ "$(gh api user --jq .login)" != 'Harishik' ]]; then
  echo 'Authenticate gh as the intended owner, Harishik, before publishing.' >&2
  exit 1
fi
if ! gh api "repos/$repository" --silent 2>/dev/null; then
  # A permission denial exits here; no fallback to public or another account.
  gh api --method POST user/repos -f name=lacunaframe -F private=true -F auto_init=false \
    -f description='Every missing frame leaves a trace. CPU video restoration and reproducible research.' --silent
fi
if [[ "$(gh api "repos/$repository" --jq .private)" != 'true' ]]; then
  echo 'Destination is not private. Refusing to push.' >&2
  exit 1
fi
# An inactive template is retained outside .github/workflows. No hosted job
# can start, even when the integration cannot access repository Actions settings.
while IFS= read -r file; do
  if [[ "$file" == .github/workflows/*.yml || "$file" == .github/workflows/*.yaml ]]; then
    echo 'Active workflow detected. Keep verification as an inactive template before publishing.' >&2
    exit 1
  fi
done < <(git ls-tree -r --name-only HEAD)
remote_url="https://github.com/$repository.git"
if git remote get-url origin >/dev/null 2>&1; then
  if [[ "$(git remote get-url origin)" != "$remote_url" ]]; then
    echo 'An unrelated origin exists. Refusing to replace it.' >&2
    exit 1
  fi
else
  git remote add origin "$remote_url"
fi
.venv/bin/python scripts/repository_inventory.py --check
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -u origin main
local_sha="$(git rev-parse HEAD)"
remote_sha="$(gh api "repos/$repository/commits/main" --jq .sha)"
if [[ "$local_sha" != "$remote_sha" ]]; then
  echo 'Remote commit does not match the prepared local commit.' >&2
  exit 1
fi
gh api "repos/$repository" --jq '{url: .html_url, private: .private, visibility: .visibility}'
