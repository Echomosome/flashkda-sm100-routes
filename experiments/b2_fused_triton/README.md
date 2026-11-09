# B2: fused Triton large-chunk KDA

**Status: measured on B300.  No H96 win; GPU run succeeded for v2.**

`rescale_fused.py` implements the Route B algebra as three Triton kernels:

1. `normalize_prefix`: q/k L2 normalisation, gate/beta activation and the FP32
   chunk-local prefix `G`, one chunk per sequence, never across a sequence
   boundary.
2. `relative_coefficients`: for each `i`, builds `[C,D]` relative decay in
   registers, reduces over D, and writes only `L[C,C]` and `Mqk[C,C]`.  Future
   positions are masked before `exp`.
3. `recurrence`: one CTA per (sequence, head), loops over chunks; state, RHS/U,
   the full CxC forward substitution, output and state update stay inside the
   kernel.  No Python per-chunk dispatch and no `torch.linalg.solve_triangular`.

Important boundaries:

- The local 16-element GEMM tiling is a register/TMEM working-set decision, not
  a CHUNK of 16.  The triangular system is the full CxC; the next state is only
  produced at the full chunk boundary.
- `input_precision="tf32x3"` is a tensor-core decomposition, not IEEE FP32.
  The numerical gate is therefore still the pinned FLA comparison, not an
  assumption.
- B16 is the same backend and precision with C16, used as the control for the
  effect of chunk size.

## Measured result

Fixed H96, complete forward:

| Impl | Official / ms | B2 / ms | Official / B2 | B1 / ms | B1 / B2 |
|---|---:|---:|---:|---:|---:|
| `B16` | 1.031472 | 11.796192 | 0.0874x | 99.041443 | 8.396x |
| `B32` | 1.030848 | 14.639808 | 0.0704x | 120.475056 | 8.229x |
| `B64` | 1.032032 | 26.473456 | 0.0390x | 199.144356 | 7.522x |

All 20 v2 comparisons (four shapes) are in
[results/v2_ab.json](../../results/v2_ab.json).  B32/B64 remain slower than B16
at H96, so the larger chunk still costs more than it saves in this
implementation.

## Offline compile evidence

The kernels were cross-compiled to `sm_103a` with Triton 3.5.1 and a system
ptxas 13.0.88, without a GPU.  The nine H96 kernels (3 C x 3 stages) all
produce a real compute body; the recurrence kernels use 512 TMEM columns and
still spill (C16/C32/C64 stack frames 776/344/640 B).  These are compiler
resource facts, not dynamic spill traffic.  See the original v2 package for the
full ptxas logs; they are not redistributed here.

The v2 GPU run is `20260911T080847138521Z`; correctness is
`20260911T072454980341Z_task3_ab_v2`.
