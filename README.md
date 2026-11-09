# FlashKDA on Blackwell: reproduction and three SM100 routes

An independent study of [FlashKDA](https://github.com/MoonshotAI/FlashKDA), the
Kimi Delta Attention forward kernel, on an NVIDIA B300 (SM103).  The official
kernel still uses SM80 `mma.sync` instructions even on Blackwell.  This
repository reproduces that result, quantifies why the design stops there, and
implements three alternative routes with full-KDA correctness checks and
complete-forward timing.

The short version: **the official SM80 path is hard to beat for this shape.**
A Phase-6-only tcgen05 swap, a real CHUNK=32/64 rebasing implementation, and a
value-column split were all built and measured.  Only the column split wins,
and only at the small per-card shape `H=12` (1.155x); it regresses at `H=96`.
The negative results are kept, with the control groups that make them
interpretable.

> Research artifact from a graduate HPC seminar.  Not affiliated with
> MoonshotAI.  All GPU numbers were produced on one B300 SXM6 AC (148 SM,
> CUDA 13.0, PyTorch 2.14).  They are not model-quality or serving claims.

## Headline results

Complete forward, `T=8192, H=96, D=128`, one CUDA Graph per forward, median of
12 alternating rounds.  A ratio above 1.0 is faster than the paired official
baseline.

| Implementation | What changed | Official / candidate | Verdict |
|---|---|---:|---|
| A (v1) | Phase 6 MMA replaced by tcgen05, C16 and prior phases unchanged | 0.353x | 2.83x slower |
| A2 (v2) | same route, U staged once, MN-major descriptors, fragment epilogue | 0.743x | 2.10x faster than A v1, still slower than official |
| B16/B32/B64 (v1) | real C32/C64, relative-prefix rebasing, FP32 triangular solve, PyTorch composite | 0.0104x / 0.0086x / 0.0052x | numerically valid, far too slow |
| B16/B32/B64 (v2) | fused three-kernel Triton implementation | 0.087x / 0.070x / 0.039x | 7.5-11.1x faster than v1, still slower |
| C | value dimension split into two 64-column CTAs | 0.884x at H96 | regresses |
| C | same, at `fixed H12` | **1.155x** | **the only positive complete-forward result** |

The official kernel is 2.29x / 2.72x / 3.27x faster than FLA's Triton
`chunk_kda` on fixed / varlen6 / varlen8 at the same H96 shape, reproducing the
published range.

## What the study found

1. **`CHUNK=16` is three constraints, not one.**  The exponential range breaks
   first when the cumulative gate is large; the 16x16 Neumann inverse is
   ill-conditioned in low precision before the series is truncated; and the MMA
   tile shape matches `m16n8k16`.  Which one "breaks first" depends on the
   input, so a single larger CHUNK is not a free knob.
2. **tcgen05 fits Phase 6 but not the surrounding dataflow.**  Phase 6 is
   `128x16 @ 16x128`, a legal `m128n128k16` operation and 30.8% of K2's MMA
   FLOPs.  A standalone probe gained 1.11x after amortising TMEM setup, but
   feeding it real `U` from Phases 1-5 cost more than the instruction saved.
   The same-data-path HMMA control was already 1.95-3.14x slower than the
   baseline, so the regression cannot be attributed to tcgen05 alone.
3. **Value columns are the clean parallelism axis.**  The state update for
   value shard `p` needs only `S^(p)` and the shared `q/k/g/beta`; the key
   reduction stays complete, so no cross-shard reduction is required.  It
   doubles the number of CTAs but duplicates q/k/gate reads and I/O warps.
   That trade wins when the grid is too small (H12) and loses when the GPU is
   already covered (H96).
4. **"Compute-bound or memory-bound" is the wrong question for the whole
   forward.**  K1 is near both rooflines; K2 has low compute and DRAM
   throughput but is limited by 96 CTAs, an 18.75% theoretical occupancy and a
   long recurrence.  NCU active-cycle and elapsed-cycle denominators answer
   different questions and are never mixed here.
5. **BF16 state needs adversarial tests.**  A Q16 model (BF16 only at chunk
   boundaries) gives 0.26-0.78% NRMSE on random inputs but 2.20-3.77% on a
   slow-write head.  NaN checks alone would miss this; the validation includes
   weak forgetting, slow writes, late-token trajectories and final state.
6. **A v2 architecture-specific backend is justified only behind a shape
   gate.**  The evidence supports keeping the official path as the default and
   keeping C as a candidate for small per-card head counts.  It does not
   support a universal SM100 replacement.

## Repository layout

```text
flashkda-sm100-routes/
|-- README.md                  English overview (this file)
|-- README.zh-CN.md            Chinese overview
|-- docs/
|   |-- report.zh.md           formal report (Chinese) + v2/A3 addendum
|   |-- from-scratch.zh.md     KDA/GPU background for new readers
|   |-- design.md              harness design and route mechanics
|   `-- evidence.md            evidence model, run IDs, hashes, scope
|-- flashkda_routes/
|   |-- cli.py                 offline / correctness / benchmark entry points
|   |-- build.py               isolated source generation + Torch JIT build
|   |-- cases.py               fixtures, metrics, runners
|   |-- correctness.py         full-KDA gate against pinned FLA naive
|   |-- benchmark.py           complete-forward CUDA Graph timing
|   |-- offline.py             CPU source/algebra checks
|   |-- routes/
|   |   |-- a_phase6.cuh       Route A: tcgen05 Phase 6 (and HMMA control)
|   |   |-- b_rescale.py       Route B: C32/C64 relative-prefix rebasing
|   |   `-- c_colsplit.py      Route C: value-column split patch generator
|   `-- probes/                standalone Task-2 measurement scripts
|-- experiments/
|   |-- a2_phase6_v2/          measured A2 sources (not wired into the CLI)
|   |-- b2_fused_triton/       cross-compiled B2 kernels (GPU run pending)
|   `-- a3_full_k2_sm100/      offline-only whole-K2 SM100 transposition
|-- results/
|   |-- official_benchmark.json
|   |-- v1_three_routes.json
|   |-- v2_ab.json
|   |-- a3_offline.json
|   `-- tables.md
|-- scripts/                   fetch upstream, run offline/correctness/benchmark
|-- tests/                     CPU algebra tests (no GPU required)
|-- FlashKDA/                  gitignored upstream checkout (pinned)
`-- fla_kda_ref/               vendored FLA KDA reference (MIT, pinned)
```

## Quick start

### Prerequisites

- Linux with an NVIDIA Blackwell GPU for correctness/timing (the published
  numbers use a B300, `sm_103a`).  CPU-only users can run the offline checks.
- CUDA 13.0+ with `nvcc` and `cuobjdump`, Python 3.10+, PyTorch 2.4+ (2.14 was
  used), Ninja.
- Enough disk for CUTLASS and a JIT build (several GB).

### 1. Fetch and install the pinned baseline

```bash
git clone https://github.com/Echomosome/flashkda-sm100-routes.git flashkda-sm100-routes
cd flashkda-sm100-routes
bash scripts/fetch_upstream.sh          # FlashKDA @ 1ce47ea + CUTLASS @ 5c149f5
python -m venv .venv && source .venv/bin/activate
pip install torch          # choose the wheel that matches your CUDA 13 install
pip install -r requirements.txt
pip install -v --no-build-isolation ./FlashKDA
```

`fetch_upstream.sh` sets `core.autocrlf=false` so the source hashes below are
stable on every platform.

### 2. Offline checks (no GPU)

```bash
python -m flashkda_routes.cli offline
python -m pytest -q
```

The offline run generates the A/A_control/C candidate trees, checks every
patch anchor and the K1/utils byte invariants, verifies the packed address
map, and compares the C16/32/64 chunked algebra against an independent NumPy
token recurrence (210 cases).  It does not compile CUDA.

### 3. Full-KDA correctness

```bash
python -m flashkda_routes.cli correctness --routes A B C --suite full
```

Each route runs in its own process.  `full` is 38 cases per implementation:
seven state-interface modes, fixed and varlen, batch 2, tail lengths
1/15/16/17/31/32/33/63/64/65, strong/weak/zero gates, zero q/k, H12/H96 short
grids, the four T8192 shapes, and a weak-forgetting long chain.  A and C must
match the baseline bitwise; B must meet the pre-registered FLA envelope.  A
failure keeps the input/output arrays and writes `FAIL`; thresholds are not
relaxed after seeing results.

### 4. Complete-forward benchmark

```bash
python -m flashkda_routes.cli benchmark --routes C \
  --validated runs/build/<PASS-full-run>
```

The benchmark refuses to time a build without a `PASS/full` verdict, re-checks
the runner source hashes and binary hashes, compares eager vs graph outputs,
poisons outputs after timing and replays to prove coverage.  It reports the
median of 12 alternating rounds; raw samples remain in `bench_C.json`.

### 5. Standalone probes

```bash
python -m flashkda_routes.probes.chunk_range --paper-only
python -m flashkda_routes.probes.neumann_probe --offline-check
python -m flashkda_routes.probes.neumann_mma_bench --offline-check
python -m flashkda_routes.probes.phase6_microbench --offline-check
python -m flashkda_routes.probes.state_precision --mode offline
```

These are the original measurement scripts behind the six discussion points.
They are independent of the harness and keep their original Chinese comments.

## Results and evidence

- [results/tables.md](results/tables.md): every table, including all 24 v1 and
  20 v2 comparisons, control deltas and A3 status.
- [docs/evidence.md](docs/evidence.md): run IDs, binary hashes, what was
  measured on hardware versus derived offline, and what is deliberately not
  claimed.
- [docs/report.zh.md](docs/report.zh.md): the formal Chinese report, including
  the v2/A3 addendum.
- [docs/from-scratch.zh.md](docs/from-scratch.zh.md): a from-zero explanation
  of KDA, GPU hierarchy, chunking and the three routes.

## Scope and limitations

- All GPU results are from one B300 SXM6 AC.  No other SM100 device, no
  multi-GPU TP communication, no serving latency.
- `H12` is the per-card head count under TP8 (`96/8`), a single-GPU shape.  It
  is not an 8-GPU deployment measurement.
- Route B is a PyTorch CUDA composite (v1) or a three-kernel Triton
  implementation (v2), not a fused SM100 kernel.  Its cost does not bound a
  hypothetical fused implementation.
- Route A replaces only Phase 6.  A3 (whole-K2 transposition) compiles and
  produces `UTCHMMA` machine code, but has **no GPU correctness or timing
  result**; it is included as clearly labelled offline work.
- No model-level accuracy, no downstream quality, no claim that the official
  kernel is globally optimal.

## License and acknowledgements

Original code and documentation in this repository are MIT licensed.  The
vendored `fla_kda_ref/` snapshot is MIT, copyright Songlin Yang, Yu Zhang and
Zhiyuan Li; the FlashKDA license is reproduced in
`third_party_flashkda_LICENSE.txt`.  Upstream FlashKDA is not vendored: the
fetch script pins commit `1ce47ea`, and every generated patch checks the
recorded source hashes before touching a file.

The study was carried out for the WMHPC x LCPU AI Infra seminar.  Thanks to the
course staff for the fixed FlashKDA snapshot and the B300 allocation.
