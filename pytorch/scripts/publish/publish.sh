#!/usr/bin/env bash
# Automated Maven Central release for io.github.mullerhai JavaCPP stack
# (javacpp, openblas, cuda, pytorch + platform aggregators) with beta-01 suffix.
#
# Quick start:
#   ./publish.sh all                  # stage + GPG sign + bundle + install-local
#   export CENTRAL_USERNAME=...
#   export CENTRAL_PASSWORD=...
#   ./publish.sh upload               # upload newest bundle to Central Portal
#   ./publish.sh all --upload --publish  # full auto (requires validated namespace)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.env"

export STAGE_DIR="${STAGE_DIR:-$ROOT/staging}"
export BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/bundles}"
export GPG_KEY_ID="${GPG_KEY_ID}"
export PUBLISH_GROUP_ID PUBLISH_SUFFIX
export JAVACPP_VERSION OPENBLAS_VERSION CUDA_VERSION PYTORCH_VERSION

# Ensure ~/.m2/settings.xml exists (from template if missing)
if [[ ! -f "$HOME/.m2/settings.xml" ]]; then
  mkdir -p "$HOME/.m2"
  cp "$ROOT/settings.xml.template" "$HOME/.m2/settings.xml"
  echo "Created ~/.m2/settings.xml from template (uses env CENTRAL_USERNAME/PASSWORD)."
fi

# Ensure public key is available for verification tooling
if ! gpg --list-secret-keys "$GPG_KEY_ID" >/dev/null 2>&1; then
  echo "ERROR: GPG secret key $GPG_KEY_ID not found."
  echo "Generate or import the mullerhai signing key first."
  exit 1
fi

PY="${PYTHON:-python3}"
exec "$PY" "$ROOT/prepare_and_publish.py" "$@"
