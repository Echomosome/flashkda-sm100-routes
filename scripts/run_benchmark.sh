#!/usr/bin/env bash
# Example: scripts/run_benchmark.sh runs/build/<PASS-full-run> C
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ $# -lt 2 ]; then
    echo "usage: $0 <validated-build-dir> <route> [extra args]" >&2
    exit 2
fi
VALIDATED="$1"; shift
ROUTE="$1"; shift
python -m flashkda_routes.cli benchmark --routes "$ROUTE" --validated "$VALIDATED" "$@"
