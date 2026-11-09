"""Pins, paths and compiler settings for the study.

The repository is laid out so that a *work directory* (``runs/`` by default)
has the upstream checkouts as siblings.  ``work.parent / "FlashKDA"`` is the
pinned FlashKDA tree and ``work.parent / "fla_kda_ref"`` is the FLA reference
snapshot, matching the layout used when the results were produced.
"""

import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_work():
    """Pick the run directory from an explicit env var or the current checkout.

    ``--work`` points at a directory whose *parent* contains ``FlashKDA/`` and
    ``fla_kda_ref/``.  Running from a checkout therefore uses ``./runs``; an
    installed console script should set ``FLASHKDA_ROUTES_WORK`` or pass
    ``--work`` explicitly.
    """
    env = os.environ.get("FLASHKDA_ROUTES_WORK")
    if env:
        return Path(env)
    cwd = Path.cwd()
    if (cwd / "FlashKDA").is_dir():
        return cwd / "runs"
    return REPO_ROOT / "runs"


DEFAULT_WORK = default_work()

# --- upstream pins ---------------------------------------------------------
FLASHKDA_REPO = "https://github.com/MoonshotAI/FlashKDA.git"
FLASHKDA_PIN = "1ce47ea"
CUTLASS_PIN = "5c149f5"
FLA_COMMIT = "a3edffc"

# --- source and reference identities --------------------------------------
# These hashes are the contract for every patch anchor in this repository.
# A different FlashKDA revision is not "close enough": generation stops.
SOURCE_SHA256 = {
    "fwd.h": "462afefb5c1bfe89428166cd81fc70954606049a41a4b67376025a2b2f91d7cc",
    "flash_kda.cpp": "f736ab87c7c2e47fe416f8d04d62502a257a6ff32a0b23452c07e1bc65c87c04",
    "smxx/fwd_kernel1.cuh": "e0f369b51753d5e1a62b8435f0b692fdc856d87e52eb27d012c22b6575a0370f",
    "smxx/fwd_kernel2.cuh": "f8a634154ca26a18855adbd3db17d6208a874b16f173d0eb9814bfefff03b9b9",
    "smxx/fwd_launch.cu": "dbaf9ba867fc8c799cf147b7e17bd2aba4a08b1b63c5f7df0cca1173a0df5562",
    "smxx/utils.cuh": "211cc28d910758e2658a53f5bcda779f1076df281ff75c09ce9bd1638a54151a",
}
FLA_NAIVE_SHA256 = "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016"

# Recorded identities of the two B300 builds used for the published numbers.
# They are informational: a fresh build on another machine will differ and
# only the shapes/route checks are enforced by default.  ``--strict-evidence``
# turns the recorded baseline hash into a hard gate.
RECORDED_BASELINE_SHA = "38d0498ca1bf67aa88369bf98763cf75df05572f04d2109b46083a4441969721"
RECORDED_CANDIDATE_SHA = "6a543354033d95dd879f783dac07b340c6a50c91df3f4bde1fa6e655aa5a8272"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def sha256_text(path):
    """Hash text with LF line endings, so a CRLF checkout is not a new revision."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def upstream_root(work):
    """Return the directory that contains ``FlashKDA/`` and ``fla_kda_ref/``."""
    return Path(work).resolve().parent


def flashkda_csrc(work):
    return upstream_root(work) / "FlashKDA" / "csrc"


def cutlass_include(work):
    return upstream_root(work) / "FlashKDA" / "cutlass" / "include"


def reference_naive(work):
    return upstream_root(work) / "fla_kda_ref" / "naive.py"


def arch_flags(arch="10.3a"):
    """Translate ``10.3a`` / ``9.0a`` into nvcc ``-gencode`` flags."""
    arch = str(arch).strip()
    if arch.endswith("a"):
        base, suffix = arch[:-1], "a"
    else:
        base, suffix = arch, ""
    major, minor = base.split(".", 1)
    cc = f"{major}{minor}"
    return [
        f"-gencode=arch=compute_{cc}{suffix},code=sm_{cc}{suffix}",
    ]


def cuda_home():
    return os.environ.get("CUDA_HOME", "/usr/local/cuda")


def cuobjdump():
    return Path(os.environ.get("CUOBJDUMP", str(Path(cuda_home()) / "bin" / "cuobjdump")))


def strict_evidence_enabled(args):
    return bool(getattr(args, "strict_evidence", False))
