#!/usr/bin/env bash
# Initialize the verl submodule and apply the HBA binary-parquet data-loader patch.
#
# Why a patch? The verl/ submodule is pinned to DDVD233/verl @ hba_public_release,
# which does NOT yet contain the embedded-binary-parquet loader (lazy decode of
# audio/video/image bytes + modality-batching sampler fast-path + omni model
# detection). Those changes are carried here as patches/0001-hba-binary-parquet-loader.patch
# and applied on top of the pinned submodule — so no write access to the fork is needed.
#
# Idempotent: safe to re-run. Usage:  bash scripts/setup_verl.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="${ROOT}/patches/0001-hba-binary-parquet-loader.patch"
cd "$ROOT"

echo "[setup_verl] Initializing verl submodule..."
git submodule update --init --recursive verl

if [ ! -f "$PATCH" ]; then
    echo "[setup_verl] ERROR: patch not found at $PATCH" >&2
    exit 1
fi

cd verl
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "[setup_verl] Patch already applied — nothing to do."
elif git apply --check "$PATCH" >/dev/null 2>&1; then
    git apply "$PATCH"
    echo "[setup_verl] Applied HBA binary-parquet loader patch."
else
    echo "[setup_verl] ERROR: patch does not apply cleanly to the current verl checkout." >&2
    echo "             The submodule commit may have changed; regenerate the patch." >&2
    exit 1
fi

echo "[setup_verl] Done. verl is ready for HBA binary-parquet training."
