#!/usr/bin/env bash
# Fetch the pinned FlashKDA and CUTLASS sources.  Nothing is vendored into git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FLASHKDA_PIN="1ce47ea"
CUTLASS_PIN="5c149f5"

if [ ! -d FlashKDA/.git ]; then
    git clone https://github.com/MoonshotAI/FlashKDA.git FlashKDA
fi

# The recorded source hashes are LF-normalised.  Keep the checkout LF so the
# hash gate is stable on Windows and on machines with autocrlf enabled.
git -C FlashKDA config core.autocrlf false

if ! git -C FlashKDA cat-file -e "${FLASHKDA_PIN}^{commit}" 2>/dev/null; then
    git -C FlashKDA fetch --all --tags
fi
git -C FlashKDA checkout -f "$FLASHKDA_PIN"

if [ ! -f FlashKDA/cutlass/include/cute/tensor.hpp ]; then
    git -C FlashKDA submodule update --init --recursive
fi
git -C FlashKDA/cutlass checkout -f "$CUTLASS_PIN"

python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from flashkda_routes.config import FLA_NAIVE_SHA256, SOURCE_SHA256, sha256_text

missing = []
for name, expected in SOURCE_SHA256.items():
    path = Path("FlashKDA/csrc") / name
    if not path.is_file() or sha256_text(path) != expected:
        missing.append(str(path))
ref = Path("fla_kda_ref/naive.py")
if not ref.is_file() or sha256_text(ref) != FLA_NAIVE_SHA256:
    missing.append(str(ref))

if missing:
    raise SystemExit("pin check failed:\n  " + "\n  ".join(missing))
print("pin check: FlashKDA sources and fla_kda_ref/naive.py match")
PY

echo "done: FlashKDA @ ${FLASHKDA_PIN}, CUTLASS @ ${CUTLASS_PIN}"
