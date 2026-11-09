# A2: Phase 6, second attempt

**Status: measured on B300.  No H96 win.**

A2 keeps the v1 scope (only Phase 6 uses tcgen05; K1, Phases 1-5, C16 and the
BF16 state boundaries are unchanged) but removes the costs that dominated v1:

- `KR` is consumed in its existing `MN_INTER` layout; there is no packed copy
  of A.
- The real BF16 `U` is written to shared memory once, in the original
  `VOLayout`; its transposed view feeds the MN-major B descriptor.
- The TMEM load uses `.16x256b.x2`, matching the original `m16n16` C fragment;
  the state epilogue goes back through the original CuTe `LDSM_T/STSM_T` path.
- Warp 0/1/2/3 own key rows 0-31/32-63/64-95/96-127.
- `A2_control` pays the same data path but keeps HMMA.

The dynamic shared memory is 110848 B, the same as A v1; extra arrays drop
from 12 KiB to 4 KiB in the design table, but the measured `sizeof` is what
matters and is recorded in the build log.

## Measured result

Fixed H96, complete forward, same protocol as v1:

| Impl | Official / ms | A2 / ms | Official / A2 | v1 / ms | v1 / A2 |
|---|---:|---:|---:|---:|---:|
| `A2_control` | 1.030800 | 1.288688 | 0.7999x | 2.722240 | 2.112x |
| `A2` | 1.031824 | 1.389152 | 0.7428x | 2.914528 | 2.098x |

The control is still 25.02% slower than official, and A2 adds another 7.80%.
That is the evidence that the common U staging, layout and synchronisation cost
is real, and that the regression is not a pure tcgen05 latency effect.

## Files

| File | Purpose |
|---|---|
| `generate.py` | patches the pinned K2 into the A2/A2_control source tree |
| `phase6.cuh` | TMEM storage, descriptors, `tcgen05.ld`, HMMA control |
| `phase6_body.inc` | real-U hand-off and vectorised state epilogue |
| `p2_generator.py` | the pinned source hashes and exact-text helpers used by the generator |
| `rescale_fused.py`, `metadata.py`, `compile_b_offline.py` | the B2 side of the same package |

Run it by placing the directory in the original `student_work` layout and using
the v2 runner from that delivery.  It is not exposed as a CLI subcommand here
because the v2 runner and its evidence package are not part of this repository.
