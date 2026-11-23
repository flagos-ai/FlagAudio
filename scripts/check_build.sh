#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build & install verification for FlagAudio.
#
# Verifies, without pulling heavy ML runtime dependencies:
#   1. sdist + wheel build cleanly from pyproject.toml
#   2. twine check passes on both artifacts
#   3. wheel contains the declared package-data (yaml configs)
#   4. `pip install --no-deps` succeeds in a fresh venv
#   5. installed metadata is importable (importlib.metadata only — no `import flag_audio`)
#
# Mirrors the verify-only posture of .github/workflows/build-{rpm,deb}.yml:
# this is not a release path, just a sanity gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PKG_NAME="flag_audio"
DIST_DIR="${REPO_ROOT}/dist"
BUILD_TMP="${REPO_ROOT}/build"

# Anchors checked against pyproject.toml — fail fast if they drift.
EXPECTED_VERSION="0.4.0-dev1"
EXPECTED_YAML="flag_audio/runtime/backend/_nvidia/enable_configs.yaml"

log()  { printf '[check_build] %s\n' "$*"; }
fail() { printf '[check_build][FAIL] %s\n' "$*" >&2; exit 1; }

cleanup() {
    rm -rf "${BUILD_TMP}"
    if [[ -n "${VENV_DIR:-}" && -d "${VENV_DIR}" ]]; then
        rm -rf "${VENV_DIR}"
    fi
}
trap cleanup EXIT

cd "${REPO_ROOT}"

# --- prereqs -----------------------------------------------------------------
log "checking prerequisites"
command -v python3 >/dev/null  || fail "python3 not found"
python3 -c "import build"      || fail "python 'build' module missing (pip install build)"
python3 -c "import twine"      || fail "python 'twine' module missing (pip install twine)"

PY="$(command -v python3)"

# --- clean previous artifacts ------------------------------------------------
log "cleaning previous dist/build"
rm -rf "${DIST_DIR}" "${BUILD_TMP}"

# --- build sdist + wheel -----------------------------------------------------
log "building sdist and wheel"
"${PY}" -m build --sdist --wheel --outdir "${DIST_DIR}" "${REPO_ROOT}"

SDIST="$(ls "${DIST_DIR}"/${PKG_NAME}-*.tar.gz 2>/dev/null | head -1)"
WHEEL="$(ls "${DIST_DIR}"/${PKG_NAME}-*.whl       2>/dev/null | head -1)"
[[ -n "${SDIST}" ]] || fail "sdist artifact not found in ${DIST_DIR}"
[[ -n "${WHEEL}" ]] || fail "wheel artifact not found in ${DIST_DIR}"
log "sdist: $(basename "${SDIST}")"
log "wheel: $(basename "${WHEEL}")"

# --- twine check -------------------------------------------------------------
log "running twine check"
"${PY}" -m twine check "${SDIST}" "${WHEEL}"

# --- wheel content sanity ----------------------------------------------------
log "inspecting wheel contents"
WHEEL_VERSION="$(printf '%s' "$(basename "${WHEEL}")" \
    | sed -E "s/^${PKG_NAME//-/_}-([^-]+)-.*/\1/")"
[[ "${WHEEL_VERSION}" == "${EXPECTED_VERSION}" ]] \
    || fail "wheel version mismatch: got '${WHEEL_VERSION}', expected '${EXPECTED_VERSION}'"

if ! "${PY}" -m zipfile -l "${WHEEL}" | grep -q "${EXPECTED_YAML}"; then
    fail "expected package-data not in wheel: ${EXPECTED_YAML}"
fi
log "yaml package-data present in wheel"

# --- install into fresh venv (--no-deps) -------------------------------------
log "creating isolated venv for install check"
VENV_DIR="$(mktemp -d -t flag_audio_venv.XXXXXX)"
"${PY}" -m venv "${VENV_DIR}"
VENV_PY="${VENV_DIR}/bin/python"
"${VENV_PY}" -m pip install --quiet --upgrade pip

# Install twine into venv so that an importable `twine` check inside the
# venv also doesn't accidentally pull system-site metadata.
"${VENV_PY}" -m pip install --quiet twine

log "pip install --no-deps the built wheel"
"${VENV_PY}" -m pip install --no-deps --force-reinstall "${WHEEL}"

# --- metadata sanity (no `import flag_audio` — that pulls torch at runtime) ---
log "verifying installed metadata"
"${VENV_PY}" - <<PYEOF
from importlib.metadata import version, PackageNotFoundError
try:
    v = version("${PKG_NAME}")
except PackageNotFoundError:
    raise SystemExit("installed '${PKG_NAME}' not discoverable by importlib.metadata")
expected = "${EXPECTED_VERSION}"
if v != expected:
    raise SystemExit(f"installed version '{v}' != expected '{expected}'")
print(f"[check_build] installed metadata version: {v}")
PYEOF

log "PASS: build, twine check, package-data, and --no-deps install all verified"
