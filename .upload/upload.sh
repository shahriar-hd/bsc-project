#!/usr/bin/env bash
# Convert checkpoints and (optionally) upload them to the Hugging Face Hub.
#
#   HF_TOKEN=hf_... bash .upload/upload.sh shahriarhd/mtl-face-auth
#   HF_TOKEN=hf_... bash .upload/upload.sh shahriarhd/mtl-face-auth --private
#
# Convert locally only (no network, no token needed):
#   bash .upload/upload.sh --convert-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-bsc_project}"
DEFAULT_REPO="shahriarhd/mtl-face-auth"

if [[ "${1:-}" == --* ]]; then
  REPO="${DEFAULT_REPO}"
  EXTRA=("$@")
elif [[ $# -ge 1 ]]; then
  REPO="$1"
  EXTRA=("${@:2}")
else
  REPO="${DEFAULT_REPO}"
  EXTRA=()
fi

cd "${ROOT}"

# Load a local, git-ignored .env (HF_TOKEN=...) if present.
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  . "${ROOT}/.env"
  set +a
fi

echo "[upload] repo=${REPO} conda_env=${CONDA_ENV} token=${HF_TOKEN:+set}"

if command -v conda >/dev/null 2>&1; then
  conda run -n "${CONDA_ENV}" python .upload/convert_and_upload.py --repo "${REPO}" ${EXTRA[@]+"${EXTRA[@]}"}
else
  python .upload/convert_and_upload.py --repo "${REPO}" ${EXTRA[@]+"${EXTRA[@]}"}
fi
