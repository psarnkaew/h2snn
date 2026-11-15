#!/bin/bash
# Apply our recorded patches to vendor clones (Part M3b B1).
#
#   bash scripts/setup/apply_vendor_patches.sh [--vendor-dir DIR] [--patch-dir DIR] [--manifest FILE] [--only NAME]
#
# For every `<patch-dir>/<NAME>.patch` (default vendor/patches/) the clone `<vendor-dir>/<NAME>` must be at the
# commit pinned for NAME in vendor/VENDORS.tsv. Then, per patch:
#   - already applied (`git apply --reverse --check` succeeds) -> "ALREADY APPLIED", nothing changes;
#   - applies cleanly (`git apply --check` succeeds)           -> applied with `git apply`, "APPLIED";
#   - neither                                                   -> "CONFLICT", nothing changes, exit 1.
# A missing clone or a clone at another commit is refused (exit 1). Only the files named in a patch are
# ever written (git apply); nothing else in the clone or the repository is touched. Git only: safe on the
# login node. The patches carry the author's edits of the H2GB clone (decisions log 2026-11-15, CS7 review).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/vendor"
PATCH_DIR="${REPO_ROOT}/vendor/patches"
MANIFEST="${REPO_ROOT}/vendor/VENDORS.tsv"
ONLY=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --vendor-dir) VENDOR_DIR="$2"; shift 2 ;;
        --patch-dir) PATCH_DIR="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

failed=0
shopt -s nullglob
for patch in "${PATCH_DIR}"/*.patch; do
    name="$(basename "$patch" .patch)"
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    dest="${VENDOR_DIR}/${name}"
    pinned="$(awk -F'\t' -v n="$name" '$1 == n {print $3}' "$MANIFEST")"
    if [ ! -d "${dest}/.git" ]; then
        echo "REFUSED     ${name}: no clone at ${dest}" >&2; failed=1; continue
    fi
    if [ -z "$pinned" ] || [ "$(git -C "$dest" rev-parse HEAD)" != "$pinned" ]; then
        echo "REFUSED     ${name}: clone is not at the pinned commit '${pinned}' of ${MANIFEST}" >&2
        failed=1; continue
    fi
    if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then
        echo "ALREADY APPLIED ${name} ($(basename "$patch"))"
    elif git -C "$dest" apply --check "$patch" 2>/dev/null; then
        if git -C "$dest" apply "$patch"; then
            echo "APPLIED     ${name} ($(basename "$patch"))"
        else
            echo "FAILED      ${name}: git apply failed after a clean check" >&2; failed=1
        fi
    else
        echo "CONFLICT    ${name}: ${patch} neither applies nor is applied; nothing changed" >&2; failed=1
    fi
done
exit "$failed"
