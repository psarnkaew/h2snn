#!/bin/bash
# Clone every `kind=clone` row of vendor/VENDORS.tsv at its pinned commit (cluster-setup CS3).
#
#   bash scripts/setup/clone_vendors.sh [--vendor-dir DIR] [--only NAME]
#
# Idempotent: a clone already at its commit is skipped. A clone with modified tracked files is
# refused (never overwritten); untracked files such as our vendor/HGB/CLONED_COMMIT.txt are left
# alone. Rows of kind in-repo / no-clone / unresolved are not cloned (an unresolved row has no
# URL in the sources: it is reported, never guessed). Exit 1 if any row was refused or failed.
# Git only: safe on the login node. Never commit the clones (HGB has no licence).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${REPO_ROOT}/vendor/VENDORS.tsv"
VENDOR_DIR="${REPO_ROOT}/vendor"
ONLY=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --vendor-dir) VENDOR_DIR="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

failed=0
while IFS=$'\t' read -r name url commit licence used_by kind; do
    [ "$name" = "name" ] && continue
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    if [ "$kind" != "clone" ]; then
        echo "NOT CLONED  ${name} (${kind})"
        continue
    fi
    if ! [[ "$commit" =~ ^[0-9a-f]{40}$ ]]; then
        echo "REFUSED     ${name}: commit '${commit}' is not a full 40-hex hash" >&2; failed=1; continue
    fi
    dest="${VENDOR_DIR}/${name}"
    if [ -d "${dest}/.git" ]; then
        if [ -n "$(git -C "$dest" status --porcelain --untracked-files=no)" ]; then
            echo "REFUSED     ${name}: dirty clone (modified tracked files) at ${dest}" >&2; failed=1; continue
        fi
        if [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ]; then
            echo "PRESENT     ${name} ${commit}"
            continue
        fi
    else
        mkdir -p "$dest"
        git -C "$dest" init -q && git -C "$dest" remote add origin "$url" || { failed=1; continue; }
    fi
    if git -C "$dest" fetch -q --depth 1 origin "$commit" && git -C "$dest" checkout -q --detach FETCH_HEAD \
        && [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ]; then
        echo "CLONED      ${name} ${commit}"
    else
        echo "FAILED      ${name}: could not check out ${commit} from ${url}" >&2; failed=1
    fi
done < "$MANIFEST"
exit "$failed"
