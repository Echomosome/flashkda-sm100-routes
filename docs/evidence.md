# Evidence model, identities and scope

The study separates three kinds of statement.  A claim is only as strong as the
label attached to it.

| Label | Meaning | What it can support |
|---|---|---|
| `[hardware]` | produced on the B300 by the author and returned as a log | the named run, shape, implementation and version only |
| `[offline source]` | read from the pinned source, ISA documentation or generated patch | structure, formulas, instruction constraints; not execution |
| `[offline derived]` | computed from recorded fields (bytes, FLOP shares, ratios) | the arithmetic shown, with its stated inputs and assumptions |
| `[offline compile]` | nvcc/ptxas/SASS produced without a GPU | the binary contains the instruction; not that it runs correctly |

No offline number is written as a GPU measurement.  No derived ratio is mixed
with a different timing protocol.

## 1. Environment

| Item | Record |
|---|---|
| GPU | NVIDIA B300 SXM6 AC, compute capability 10.3, 148 SM |
| CUDA | 13.0, V13.0.88 |
| Python / PyTorch | 3.12 / 2.14.0+cu130 |
| NumPy / NCU | 2.5.3 (later probes) / Nsight Compute 2025.3.1 |
| FlashKDA | `1ce47ea` (2026-07-29) |
| CUTLASS / FLA | `5c149f5` / `a3edffc` |
| Candidate build | isolated Torch JIT extension, explicit `compute_103a/sm_103a` |
| Clocks / cache | not locked during Graph timing; NCU replay uses `--cache-control all` and `clock-control none` |

The Graph timing and the NCU kernel timings use different cache, replay and
clock conditions.  They are never added, subtracted or divided into each other.

## 2. Run IDs

| Evidence | UTC run ID | Where used |
|---|---|---|
| Official fixed / varlen benchmark | `20260909T164935Z` / `20260909T165802Z` | reproduction table |
| Initial NCU and SASS | `20260909T172953Z` | K1/K2 bottlenecks |
| Detailed fixed NCU | `20260909T182132Z` | resource/occupancy analysis |
| Exponential range probe | `20260910T061622976790Z` | discussion point 1 |
| Neumann numeric / MMA cost | `20260910T064311694334Z` / `20260910T092501138041Z` | discussion point 1 |
| Phase-6 standalone tcgen05 | `20260910T125328470967Z` | 1.1096x local result |
| State precision | `20260910T133341103264Z` | discussion point 5 |
| P2 compile / correctness | `20260910T152803026613Z` | Route C |
| P2 complete performance | `20260910T162614136038Z` | Route C historical |
| P2 NCU and offline recovery | `20260910T165005117299Z` | Route C mechanism |
| Three-route short / full correctness | `20260911T011018603611Z` / `20260911T013931783287Z` | v1 gate |
| Three-route complete performance | `20260911T015639253562Z` | v1 results |
| A/B v2 correctness | `20260911T072454980341Z_task3_ab_v2` | v2 gate |
| A/B v2 complete performance | `20260911T080847138521Z` | v2 results |
| A3 offline compile/SASS | local build, 2026-09-11 | A3 only |
| A3 NCU attempt (ENOSPC) | `20260911T121125285990Z` | failure log, no data |

## 3. Source and binary identities

The six baseline source hashes live in `flashkda_routes/config.py`.  They are
hashes of LF-normalised text; `sha256_text()` is used for the gate.

| Object | SHA-256 | Note |
|---|---|---|
| baseline `.so` | `38d0498ca1bf67aa88369bf98763cf75df05572f04d2109b46083a4441969721` | B300 build, not distributed |
| P2 candidate `.so` | `6a543354033d95dd879f783dac07b340c6a50c91df3f4bde1fa6e655aa5a8272` | B300 build, not distributed |
| P2 `candidate.patch` | `bb29989a93bf1afa137621563c8236216e0c5fc95cca4c67257896bdcba37bed` | original delivery; the regenerated patch differs only in comments |
| P2 performance log | `57659a591a2415c08216ce66330c9092fe963b53ccab81f245be35803aadb7b8` | recovered text |
| P2 per-round JSON | `85a66c84ba52b71be79d9e552853840fca0f4fa0e7d61e1bcba8c06e3b7e36a0` | local recovery |
| NCU wide CSV | `6854fe7957bb104bd90d770a4b000aed4687a310eaee44ea6c751b54c9cdf09f` | original bytes |
| A full `.so` | `a2acd05e67281733c0943ac8c94bbdb035fa7e5fb0f6f7bbcd592dfd37a546be` | B300 build |
| three-route full log | `2a935cd84c57ea0cbe61787d418e4f407a1a41514e3d03f03a697ea229d0311c` | recovered |
| three-route performance log | `d7e8a05a4c599ae1480b90303e3a06419b455c69adea57f2089952f39dce99b9` | recovered |
| A/B v2 performance log | `8aa8b144254e6a4ccf368aad293b6b93d67819bfec2061b1510e206c29f54c20` | recovered |

The binaries and `.ncu-rep` files are not in this repository: they live on the
experiment machine.  The hashes identify them; they are not a claim that the
bytes were re-verified here.

## 4. What was measured, and what was not

**Measured on hardware**

- Official fixed / varlen6 / varlen8 benchmark at H96.
- SASS confirmation of `HMMA.16816.F32.BF16` / `.F16` and `UTCHMMA` after the
  Route A build.
- NCU metrics for K1/K2 at H96 and for the historical Route C at H96/H12.
- Standalone exponential-range, Neumann, MMA-cost, Phase-6 and state-precision
  probes.
- v1: 38 full-KDA cases for each of A_control, A, B16, B32, B64, C; 24
  complete-forward comparisons.
- v2: 38 full-KDA cases for A2/A2_control/B16/B32/B64; 20 complete-forward
  comparisons.

**Offline only**

- A3 whole-K2 SM100 transposition: nvcc/CuTe/SASS pass, no GPU run.
- B2 fused Triton: cross-compiled to `sm_103a`, no GPU correctness or timing.
- The v2 A2/B2 packages in `experiments/` are the measured versions for v2;
  their `.so` and logs are not redistributed.

**Not claimed**

- model accuracy or downstream quality;
- other SM100 devices, other CUDA versions, binary compatibility;
- multi-GPU TP communication or serving latency;
- that the official kernel is globally optimal;
- that the negative results generalise to every possible SM100 implementation.

## 5. Reproducing the published numbers

The exact B300 binaries cannot be rebuilt byte-for-byte on another machine
(compiler, driver and PyTorch versions differ).  `--strict-evidence` enforces
the recorded baseline hash and is therefore only usable on the original
machine or an identical image.  Without it, the harness still enforces:

- the six source hashes and every patch anchor;
- the 38-case correctness matrix and the pre-registered gates;
- build/validation/benchmark separation and post-timing poison checks.

Reported ratios should then be compared as *this machine, this commit*, not as
an attempt to reproduce the last digit of the B300 table.

## 6. Packaging verification

Before publication, the generated A, A_control and C source trees were compared
with the candidate trees that produced the v1 numbers.  After stripping
comments and whitespace, `fwd_kernel2.cuh`, `c1_phase6.cuh`, `fwd_launch.cu`
and `flash_kda.cpp` are token-identical for all three candidates.  The
differences are English comments added for this repository; no kernel
expression, layout, barrier or PTX operand changed.

The recorded `P2 candidate.patch` hash therefore belongs to the original
delivery.  A fresh run of this repository produces a different patch file hash
(comments differ) but the same kernel code.

## 7. Sanitisation

The original logs contain the cluster home path and account name.  Those paths
are replaced by `<work>/...` in this repository.  No student identifier, email
address, course submission file, PPT or raw terminal dump is redistributed.
The numerical fields, run IDs and hashes are unchanged.
