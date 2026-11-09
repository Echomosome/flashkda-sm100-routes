"""Checks that run without CUDA: source generation, address models, B algebra.

Everything emitted here is labelled ``OFFLINE_CPU``.  A pass proves source
invariants and the chunked algebra; it does not prove that a kernel compiles,
that a CuTe layout is correct at runtime, or that a GPU result matches.
"""

import json
from pathlib import Path

import numpy as np

from .build import generate
from .emit import emit
from .routes.b_rescale import chunk_core, recurrent_numpy


def run(work, build):
    build = Path(build)
    for route in ["A", "A_control", "C"]:
        generate(work, build / route, route)
        emit("SOURCE_AUDIT", route=route, status="PASS",
             scope="source_invariants_only; no_CUDA_compile")

    # The packed map covers all 2048 BF16 elements; 16-byte descriptor
    # alignment is checked independently.
    packed = lambda row, k: (row // 8) * 128 + (k // 8) * 64 + (row % 8) * 8 + k % 8
    offsets = [packed(r, k) for r in range(128) for k in range(16)]
    assert sorted(offsets) == list(range(2048))
    # TMEM readout (key row=tid, value col) and the HMMA control each cover
    # exactly the 128x128 state.
    tmem = {(tid, c + j) for tid in range(128) for c in range(0, 128, 16) for j in range(16)}
    hmma = []
    for tid in range(128):
        w, l = divmod(tid, 32)
        for m in range(0, 128, 16):
            for n in range(4):
                for p in range(4):
                    hmma.append((m + l // 4 + (p // 2) * 8, w * 32 + n * 8 + (l % 4) * 2 + p % 2))
    assert len(hmma) == len(set(hmma)) == len(tmem) == 16384 and set(hmma) == tmem
    emit("A_LOGICAL_LAYOUT", packed_elements=2048, state_elements=16384, status="PASS",
         scope="integer_address_bijections; CuTe_STSM_TMA_and_GPU_synchronization_pending")

    rows = []
    for dtype in [np.float64, np.float32]:
        for dim, time in [(16, 1), (16, 17), (16, 33), (16, 65), (16, 257), (128, 65), (128, 129)]:
            for kind in ["random", "alpha2_5", "alpha5", "zero", "weak_coherent"]:
                rng = np.random.default_rng(20260911 + dim + time)
                q, k, v = [rng.normal(size=(2, time, dim)).astype(dtype) for _ in range(3)]
                if kind == "weak_coherent":
                    q.fill(1)
                    k.fill(1)
                q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(dim)
                k /= np.linalg.norm(k, axis=-1, keepdims=True)
                beta = rng.uniform(.02, .98, size=(2, time)).astype(dtype)
                if kind == "weak_coherent":
                    beta.fill(.01)
                gate = -rng.uniform(0, 5, size=k.shape).astype(dtype)
                if kind != "random":
                    gate.fill({"alpha2_5": -2.5, "alpha5": -5., "zero": 0.,
                               "weak_coherent": -1e-7}[kind])
                s = rng.normal(size=(2, dim, dim)).astype(dtype) / 8
                initial = s.copy()
                ro, rs = recurrent_numpy(q, k, v, gate, beta, s)
                for chunk in [16, 32, 64]:
                    o, st = chunk_core(q, k, v, gate, beta, s, chunk)
                    error = max(float(np.max(abs(o - ro))), float(np.max(abs(st - rs))))
                    relative = max(
                        float(np.linalg.norm(o - ro) / max(np.linalg.norm(ro), 1e-30)),
                        float(np.linalg.norm(st - rs) / max(np.linalg.norm(rs), 1e-30)))
                    tol = 2e-11 if dtype == np.float64 else 3e-4
                    if not np.isfinite(o).all() or not np.isfinite(st).all() \
                            or error > tol or relative > tol:
                        raise AssertionError((dtype, dim, time, kind, chunk, error, relative))
                    assert np.array_equal(s, initial)
                    rows.append(dict(dtype=dtype.__name__, D=dim, T=time, gate=kind,
                                     chunk=chunk, max_abs=error, nrmse=relative, status="PASS"))
    (build / "B_numpy_checks.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for dtype in ["float64", "float32"]:
        subset = [x for x in rows if x["dtype"] == dtype]
        emit("B_CHUNK_VS_TOKEN", evidence="OFFLINE_CPU", dtype=dtype, cases=len(subset),
             max_abs=max(x["max_abs"] for x in subset),
             max_nrmse=max(x["nrmse"] for x in subset), status="PASS",
             reference="independent_NumPy_token_recurrence; actual_FLA_PyTorch_pending")
    # C16/32/64 with strong gates: the negative control fails where the known
    # exponential range boundary predicts, the rebased version stays finite.
    for chunk in [16, 32, 64]:
        for alpha in [2.5, 5.0]:
            q = np.ones((1, chunk, 16), np.float32) / 16
            k = np.ones_like(q) / 4
            v = np.ones_like(q)
            g = np.full_like(q, -alpha)
            beta = np.full((1, chunk), .5, np.float32)
            s = np.zeros((1, 16, 16), np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                bad = chunk_core(q, k, v, g, beta, s, chunk, rescale=False)
            good = chunk_core(q, k, v, g, beta, s, chunk)
            nonfinite = any(not np.isfinite(x).all() for x in bad)
            assert nonfinite == (chunk * alpha > 89)
            assert all(np.isfinite(x).all() for x in good)
            emit("B_RESCALE_ABLATION", evidence="OFFLINE_CPU", chunk=chunk, alpha=alpha,
                 absolute_prefix="NONFINITE_OBSERVED" if nonfinite else "FINITE",
                 relative_prefix="FINITE",
                 scope="CPU_FP32_exp; not_CUDA_ex2_or_full_GPU_result")
    emit("OFFLINE_COMPLETE", status="PASS", numerical_cases=len(rows),
         CUDA_compile=False, GPU_execution=False)
