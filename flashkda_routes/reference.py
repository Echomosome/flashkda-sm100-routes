"""Run the pinned FLA ``naive_recurrent_kda`` as the numerical reference.

The reference is read from ``fla_kda_ref/naive.py`` next to the upstream
checkout.  Its SHA-256 is pinned; an unexpected file stops the run unless
``FLASHKDA_ROUTES_ALLOW_REFERENCE_DRIFT=1`` is set explicitly.
"""

import os
import runpy

import torch

from .config import FLA_NAIVE_SHA256, reference_naive, sha256_text
from .routes.b_rescale import activate_torch


def naive_reference(work, data, initial, spec):
    """Execute the pinned naive recurrence; short cases run on CPU, long on GPU."""
    path = reference_naive(work)
    actual = sha256_text(path)
    if actual != FLA_NAIVE_SHA256 and os.environ.get("FLASHKDA_ROUTES_ALLOW_REFERENCE_DRIFT") != "1":
        raise RuntimeError(
            f"pinned FLA naive.py changed: {path} has {actual}; set "
            "FLASHKDA_ROUTES_ALLOW_REFERENCE_DRIFT=1 only for a deliberately new reference")
    naive = runpy.run_path(str(path))["naive_recurrent_kda"]
    # A CPU activation is an independent control from the candidate's GPU
    # activation.  Very long inputs stay on GPU to avoid a huge CPU state loop.
    large = data[0].shape[1] >= 1024
    rd = data if large else tuple(x.cpu() for x in data)
    q, k, v, g, b = activate_torch(rd)
    outs, states, bos = [], [], 0
    lengths = spec["lengths"] or [spec["time"]] * spec["batch"]
    for n, count in enumerate(lengths):
        bid, start = (0, bos) if spec["lengths"] else (n, 0)
        sl = (slice(bid, bid + 1), slice(start, start + count))
        s0 = None if initial is None else initial[n:n + 1].float().transpose(-1, -2)
        if s0 is not None and not large:
            s0 = s0.cpu()
        o, s = naive(q[sl], k[sl], v[sl], g[sl], b[sl], scale=1.,
                     initial_state=s0, output_final_state=True)
        outs.append(o.cpu())
        states.append(s.transpose(-1, -2).cpu())
        bos += count
    return torch.cat(outs, dim=1 if spec["lengths"] else 0), torch.cat(states, dim=0)
