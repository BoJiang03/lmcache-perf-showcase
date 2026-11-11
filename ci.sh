#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# CI entry point: run one PRT suite and leave a machine-readable report.
#
#   perf_regression_testing/ci.sh nightly|weekly|quick [out_root]
#
# The CI decides when this runs (Bo 2026-10-11); this script only runs it.
# Output: <out_root>/<suite>_<UTC yyyymmdd-HHMMSS>/report.json (+ report.md);
# out_root defaults to <host.root>/prt-suites from config.toml.
# Suites share the L2 dir under host.root, so a second suite waits on a lock
# (PRT_LOCK_WAIT_S, default 6 h) instead of corrupting the first.
# Exit code: the suite's (0 pass, 1 gated regression, 2 invalid run / no
# images), or 75 when the lock wait timed out.
# Needs: docker, nvidia-smi, flock, Python >= 3.11 as $PYTHON (default python3).
set -euo pipefail
suite=${1:?usage: ci.sh nightly|weekly|quick [out_root]}
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
root=$("$PYTHON" -c 'import tomllib; print(tomllib.load(open("perf_regression_testing/config.toml", "rb"))["host"]["root"])')
out_root=${2:-$root/prt-suites}
out=$out_root/${suite}_$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$out_root"
exec 9>"$root/prt-suite.lock"
if ! flock -w "${PRT_LOCK_WAIT_S:-21600}" 9; then
  echo "ci.sh: another suite still holds $root/prt-suite.lock" >&2
  exit 75
fi
rc=0
"$PYTHON" -m perf_regression_testing.suite --suite "$suite" --out "$out" || rc=$?
echo "report: $out/report.json (exit $rc)"
exit $rc
