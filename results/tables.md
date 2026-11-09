# Result tables

All GPU numbers below come from the recorded B300 runs.  Rows are complete-forward CUDA Graph medians of 12 rounds.  A ratio below 1.0 means the candidate is slower than the official FlashKDA baseline in the same run.

## 1. Official FlashKDA vs FLA Triton (reproduction)

| Case | FlashKDA BF16-state / ms | FlashKDA FP32-state interface / ms | FLA `chunk_kda` / ms | FLA / FlashKDA (FP32 interface) |
|---|---:|---:|---:|---:|
| fixed | 1.7553 | 1.7135 | 3.9279 | 2.2923x |
| varlen6 | 1.4759 | 1.4976 | 4.0696 | 2.7174x |
| varlen8 | 1.1866 | 1.2136 | 3.9648 | 3.2670x |

Shape: T=8192, H=96, D=128.  `varlen6` lengths `[1300,547,2048,963,271,3063]`; `varlen8` eight sequences of 1024.  The FP32-state *interface* does not imply an internal FP32 recurrence.

## 2. v1: three routes, fixed H96

| Impl | Paired baseline / ms | Candidate / ms | baseline / candidate | Time change |
|---|---:|---:|---:|---:|
| A_control | 1.029104 | 2.722240 | 0.378036x | +164.53% |
| A | 1.030224 | 2.914528 | 0.353479x | +182.90% |
| B16 | 1.031152 | 99.041443 | 0.010411x | +9504.93% |
| B32 | 1.033008 | 120.475056 | 0.008574x | +11562.55% |
| B64 | 1.033264 | 199.144356 | 0.005189x | +19173.33% |
| C | 1.028480 | 1.162928 | 0.884388x | +13.07% |

## 3. v1: the one positive result (fixed H12)

| Impl | Paired baseline / ms | Candidate / ms | baseline / candidate | Time change |
|---|---:|---:|---:|---:|
| A_control | 0.793920 | 2.490320 | 0.318802x | +213.67% |
| A | 0.793392 | 2.681632 | 0.295862x | +238.00% |
| B16 | 0.791088 | 46.081697 | 0.017167x | +5725.10% |
| B32 | 0.790256 | 32.832479 | 0.024069x | +4054.66% |
| B64 | 0.790384 | 35.571184 | 0.022220x | +4400.49% |
| C | 0.791984 | 0.685872 | 1.154711x | -13.40% |

`H12` is the single-GPU per-card head count (96/TP8); it is not an 8-GPU serving measurement.

## 4. v2: A2/B2 after optimisation (fixed H96)

| Impl | Official / ms | v2 / ms | official / v2 | v2 vs official | v1 / ms | v1 / v2 |
|---|---:|---:|---:|---:|---:|---:|
| A_control | 1.030800 | 1.288688 | 0.7999x | +25.02% | 2.722240 | 2.1124x |
| A | 1.031824 | 1.389152 | 0.7428x | +34.63% | 2.914528 | 2.0981x |
| B16 | 1.031472 | 11.796192 | 0.0874x | +1043.63% | 99.041443 | 8.3961x |
| B32 | 1.030848 | 14.639808 | 0.0704x | +1320.17% | 120.475056 | 8.2293x |
| B64 | 1.032032 | 26.473456 | 0.0390x | +2465.18% | 199.144356 | 7.5224x |

v2 is a different run from v1; the last two columns describe implementation progress, not a paired experiment.

## 5. v2: all 20 comparisons

| Shape | Impl | Official / ms | v2 / ms | official / v2 | Time change |
|---|---|---:|---:|---:|---:|
| fixed_h96 | A_control | 1.030800 | 1.288688 | 0.7999x | +25.02% |
| fixed_h96 | A | 1.031824 | 1.389152 | 0.7428x | +34.63% |
| varlen6_h96 | A_control | 0.863904 | 0.995664 | 0.8677x | +15.25% |
| varlen6_h96 | A | 0.863600 | 0.995824 | 0.8672x | +15.31% |
| varlen8_h96 | A_control | 0.691136 | 0.781024 | 0.8849x | +13.01% |
| varlen8_h96 | A | 0.690752 | 0.782288 | 0.8830x | +13.25% |
| fixed_h12 | A_control | 0.788656 | 1.051840 | 0.7498x | +33.37% |
| fixed_h12 | A | 0.788944 | 1.151984 | 0.6849x | +46.02% |
| fixed_h96 | B16 | 1.031472 | 11.796192 | 0.0874x | +1043.63% |
| fixed_h96 | B32 | 1.030848 | 14.639808 | 0.0704x | +1320.17% |
| fixed_h96 | B64 | 1.032032 | 26.473456 | 0.0390x | +2465.18% |
| varlen6_h96 | B16 | 0.859344 | 9.403264 | 0.0914x | +994.24% |
| varlen6_h96 | B32 | 0.862976 | 12.175424 | 0.0709x | +1310.86% |
| varlen6_h96 | B64 | 0.865760 | 22.577264 | 0.0383x | +2507.80% |
| varlen8_h96 | B16 | 0.690464 | 8.960080 | 0.0771x | +1197.69% |
| varlen8_h96 | B32 | 0.691392 | 11.601344 | 0.0596x | +1577.97% |
| varlen8_h96 | B64 | 0.690992 | 21.287040 | 0.0325x | +2980.65% |
| fixed_h12 | B16 | 0.785968 | 10.138944 | 0.0775x | +1189.99% |
| fixed_h12 | B32 | 0.786112 | 11.513312 | 0.0683x | +1364.59% |
| fixed_h12 | B64 | 0.785936 | 20.054144 | 0.0392x | +2451.63% |

## 6. Control-group deltas (v1)

| Case | A vs A_control | B32 vs B16 | B64 vs B16 |
|---|---:|---:|---:|
| fixed_h96 | +7.06% | +21.64% | +101.07% |
| varlen6_h96 | +18.72% | +20.85% | +98.57% |
| varlen8_h96 | +20.69% | +21.52% | +99.19% |
| fixed_h12 | +7.68% | -28.75% | -22.81% |

## 7. A3 status

- Status: **OFFLINE_ONLY**.
- each A3 K2 variant: 11 UTCHMMA, 0 HMMA; control: 104 HMMA, 0 UTCHMMA (static counts).
- Target: fixed T8192/H96/D128 complete forward >= 2x official.
- Measured on GPU: **False**.  Blocker: B300 scratch filesystem ENOSPC during the NCU run 20260911T121125285990Z; no GPU correctness or timing for this version.

## 8. Reading the numbers

- A ratio > 1 means faster than the paired official baseline; < 1 means slower.
- `+13.40%` time change corresponds to `1.1547x` speedup, not `+15.47%`; the two percentages are not interchangeable.
- NCU kernel durations and Graph times use different cache/clock/replay conditions and are never added or divided into each other.
- None of these are model-level quality or serving-latency claims.
