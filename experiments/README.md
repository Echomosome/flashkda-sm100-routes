# Follow-up experiments

These directories preserve the sources of the versions that came after the v1
three-route delivery.  They are **not wired into the `flashkda_routes` CLI**:
they were built and measured by the original version-specific runners, and the
honest status of each is recorded below.

| Directory | Version | Hardware status | Result |
|---|---|---|---|
| `a2_phase6_v2/` | A2/A2_control | **measured on B300** | 2.10x faster than A v1; still 0.743x of official at fixed H96 |
| `b2_fused_triton/` | B2 (C16/32/64) | **measured on B300** | 7.5-11.1x faster than B v1; still 11.4-25.7x slower than official at H96 |
| `a3_full_k2_sm100/` | A3 whole-K2 transposition | **offline only** | compiles, CuTe/SASS checks pass; no GPU correctness or timing |

The v2 correctness run is `20260911T072454980341Z_task3_ab_v2` (350
comparisons, full PASS).  The v2 performance run is
`20260911T080847138521Z` (20 comparisons, 480 raw timings, 80 postchecks).
The complete tables are in [results/tables.md](../results/tables.md) and
[results/v2_ab.json](../results/v2_ab.json).

These sources keep their original Chinese comments and are intended for
inspection and reuse, not as a second installable package.
