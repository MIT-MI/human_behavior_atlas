#!/usr/bin/env bash
# Initialize the verl submodule and apply the HBA patches on top of it.
#
# Why patches? The verl/ submodule is pinned to DDVD233/verl @ hba_public_release,
# which does NOT contain:
#   patches/0001-hba-binary-parquet-loader.patch  — embedded-binary-parquet loader
#       (lazy byte decode of audio/video/image, modality-batching sampler fast-path,
#        Qwen2.5-Omni model detection)
#   patches/0002-hba-harpo-advantage.patch        — the HARPO advantage estimator
#       (adv_estimator=harpo: compute_harpo_outcome_advantage + active ray_trainer branch)
# They are applied on top of the pinned submodule — so no write access to the fork is needed.
#
# Idempotent: safe to re-run. Applies every patches/*.patch in sorted order.
# Usage:  bash scripts/setup_verl.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[setup_verl] Initializing verl submodule..."
git submodule update --init --recursive verl

shopt -s nullglob
PATCHES=("${ROOT}"/patches/*.patch)
shopt -u nullglob
if [ ${#PATCHES[@]} -eq 0 ]; then
    echo "[setup_verl] ERROR: no patches found in ${ROOT}/patches" >&2
    exit 1
fi

cd verl
for PATCH in "${PATCHES[@]}"; do
    name="$(basename "$PATCH")"
    if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        echo "[setup_verl] ${name}: already applied — skipping."
    elif git apply --check "$PATCH" >/dev/null 2>&1; then
        git apply "$PATCH"
        echo "[setup_verl] ${name}: applied."
    else
        echo "[setup_verl] ERROR: ${name} does not apply cleanly to the current verl checkout." >&2
        echo "             The submodule commit may have changed; regenerate the patch." >&2
        exit 1
    fi
done

echo "[setup_verl] Done. verl is ready for HBA binary-parquet training (GRPO + HARPO)."
