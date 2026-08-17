#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git"
readonly EXPECTED_REVISION="40f8a57e0c0204529e40f0c57b37f283462f687c"
readonly INSTALL_ROOT="${CPP_AISBENCH_AUTO_TOOLS_ROOT:-/home/w00985415/tools/aisbench_auto_tools_prefix}"

if [[ ! -d "${INSTALL_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${INSTALL_ROOT}")"
    git clone "${REPOSITORY}" "${INSTALL_ROOT}"
fi

actual_revision="$(git -C "${INSTALL_ROOT}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${EXPECTED_REVISION}" ]]; then
    echo "AISBench auto tools revision mismatch: expected=${EXPECTED_REVISION} actual=${actual_revision}" >&2
    exit 1
fi

python3 -m pip show ais-bench-benchmark >/dev/null
(
    cd "${INSTALL_ROOT}"
    PYTHONDONTWRITEBYTECODE=1 python3 aisbench_test.py --help >/dev/null
)
echo "CPP_AISBENCH_AUTO_TOOLS_READY root=${INSTALL_ROOT} revision=${actual_revision}"
