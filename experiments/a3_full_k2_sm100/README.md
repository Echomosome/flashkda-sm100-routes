# A3: transpose the whole K2 recurrence to SM100

**Status: offline only.  Compiles and emits `UTCHMMA`; no GPU correctness or
timing result.**

A3 changes the optimisation scope: instead of replacing Phase 6 alone, it moves
every matrix multiply in K2 and keeps one consistent thread/row ownership from
Phase 1 to the state epilogue.  K1 stays as delivered.

## The key idea: transpose the tile, not the math

The official physical state is `S[value,key]` with both axes 128; a chunk has 16
tokens.  Using `(AB)^T = B^T A^T`, the stages become:

| Stage | Official expression | A3 expression | Logical MxNxK |
|---|---|---|---|
| Phase 1 | `KD @ S^T`, `QD @ S^T` | `S @ [KD;QD]^T` | 128x32x128 |
| Phase 3 | `U = INV @ RHS` | `U^T = RHS^T @ INV^T` | 128x16x16 |
| Phase 4 | `Mqk @ U` | `U^T @ Mqk^T` | 128x16x16 |
| Phase 6 | `(KR^T @ U)^T` | `U^T @ KR` | 128x128x16 |

`M=16` was the original implementation's orientation, not a mathematical
requirement.  Transposing lets the value dimension supply M=128, so no token
padding to 64 and no head packing is needed.

Other changes: KD/QD share one `[32,128]` shared-memory pack per chunk; RHS and
U share a 4 KiB buffer (their lifetimes do not overlap); Phase 4 and Phase 6
are committed together with one completion barrier; TMEM is allocated once per
CTA (256 columns); the state epilogue reads a `16x128` strip with
`.16x256b.x16`.

## Offline evidence

| Check | Result |
|---|---|
| baseline source invariants | PASS |
| SMEM integer address model | 24832 coordinates PASS |
| TMEM thread ownership | 16384 state + 2048 U elements PASS |
| CPU stage chain with BF16 rounding | 12 groups, 1/2/8/32 chunks PASS |
| nvcc 13.0.88 full `fwd_launch.cu` | A3 and HMMA control, 14 K2 variants each PASS |
| pinned CuTe host layout enumeration | `errors=0` |
| SASS | every A3 K2 variant: 11 `UTCHMMA`, 0 `HMMA`; control: 104 `HMMA`, 0 `UTCHMMA` |
| registers / spills | A3 94-96 regs/thread, control 56-64, 0 stack/spill |

The SASS counts are static instruction counts, not dynamic executions.

## Why there is no GPU number

The target was a complete forward at fixed H96 of at least 2x the official
baseline (about 0.516 ms from the same-round ~1.03 ms).  The B300 run could not
start: the scratch filesystem returned `ENOSPC` during the NCU attempt
`20260911T121125285990Z`, and no build or correctness PASS existed for A3.  The
offline result therefore proves only that the full-K2 SM100 candidate compiles
and contains the intended instruction; it does not prove correctness or speed.

## Files

| File | Purpose |
|---|---|
| `generate.py` | full independent copy of pinned sources with K2 body replaced |
| `source_pins.py` | source hashes used by the generator |
| `sm100_ops.cuh` | descriptors, TMEM allocation, MMA/commit/wait, wide loads |
| `recurrence_body.inc` | the K2 body with the four transposed stages |
| `compile_offline.py` | GPU-free nvcc compile and SASS extraction |
| `offline_checks.py` | CPU algebra and address-model checks |

Run `compile_offline.py` after pointing it at the pinned checkout; it requires
nvcc/ptxas 13.0.88 and CUTLASS `5c149f5`.
