#!/bin/sh
# Publish only with the owner's authenticated account; never rewrite history.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
TARGET=rogeriojorge/vmex-benchmark-analytical
LOGIN=$(gh api user --jq .login)
[ "$LOGIN" = rogeriojorge ] || { echo "Wrong GitHub account: $LOGIN" >&2; exit 1; }
ID=$(gh api user --jq .id)
EMAIL="${ID}+rogeriojorge@users.noreply.github.com"
[ -f plan.md ] && [ -f README.md ]
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    git init -b main
else
    [ "$(git rev-parse --show-toplevel)" = "$ROOT" ] || { echo "Nested inside another repository" >&2; exit 1; }
fi
git config --local user.name rogeriojorge
git config --local user.email "$EMAIL"
export GIT_AUTHOR_NAME=rogeriojorge GIT_COMMITTER_NAME=rogeriojorge
export GIT_AUTHOR_EMAIL="$EMAIL" GIT_COMMITTER_EMAIL="$EMAIL"
# Stage explicitly. Local environments, credentials and downloaded repositories
# are never included. Read the staged diff before rerunning with PUBLISH=1.
git add README.md plan.md AGENT_PROMPT.md LICENSE NOTICE.md sources.json cases.json \
    requirements.txt pyproject.toml benchmark_matrix.json .gitignore .github benchmarks tests tools docs \
    inputs results figures HANDOFF_SHA256SUMS
if [ "${PUBLISH:-0}" != 1 ]; then
    git diff --cached --stat
    echo "Review git diff --cached. Then PUBLISH=1 sh tools/publish.sh authorizes commit and push."
    exit 0
fi
if ! git diff --cached --quiet; then
    git commit -m "${COMMIT_MESSAGE:-Add analytical references and staged VMEX benchmark plan}"
fi
# Validate the commit this invocation is publishing. Existing ancestors may
# include GitHub-created merge commits with their own committer identity.
COMMIT_IDENTITY=$(git show -s --format='%an|%cn|%ae|%ce' HEAD)
if [ "$COMMIT_IDENTITY" != "rogeriojorge|rogeriojorge|$EMAIL|$EMAIL" ]; then
    echo "Unexpected new commit author/committer; inspect it before publication." >&2
    exit 1
fi
if git show -s --format='%B' HEAD | grep -i 'Co-authored-by:'; then
    echo "Unexpected co-author trailer on the new commit; inspect locally before publication." >&2
    exit 1
fi
if gh repo view "$TARGET" --json name >/dev/null 2>&1; then
    [ "$(gh repo view "$TARGET" --json visibility --jq .visibility)" = PUBLIC ] || {
        echo "Existing target is not public; inspect it before changing visibility." >&2; exit 1; }
else
    gh repo create "$TARGET" --public --description "Analytical equilibrium, sensitivity and integration benchmarks for VMEX"
fi
if git remote get-url origin >/dev/null 2>&1; then
    case "$(git remote get-url origin)" in
      "https://github.com/$TARGET"|"https://github.com/$TARGET.git"|"git@github.com:$TARGET.git") ;;
      *) echo "Unexpected origin; refusing to replace it" >&2; exit 1;;
    esac
else
    git remote add origin "https://github.com/$TARGET.git"
fi
gh auth setup-git
git push -u origin HEAD:main
