# Results

| File | Contents |
|---|---|
| `official_benchmark.json` | reproduction of the official fixed/varlen H96 benchmark |
| `v1_three_routes.json` | 24 complete-forward comparisons, control deltas, SASS counts, compiled resources |
| `v2_ab.json` | 20 complete-forward comparisons for A2/B2, parsed from the recorded v2 log |
| `a3_offline.json` | A3 offline compile/layout/SASS facts; explicitly no GPU result |
| `tables.md` | human-readable tables generated from the four JSON files |

All ratios are paired baseline/candidate medians from 12 alternating rounds
unless a file says otherwise.  The v1 and v2 runs are different sessions; the
`v1/v2` columns describe implementation progress, not a paired experiment.
