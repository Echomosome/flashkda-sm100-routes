#!/usr/bin/env bash
# Example: scripts/run_correctness.sh --routes A B C --suite full
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m flashkda_routes.cli correctness "$@"
