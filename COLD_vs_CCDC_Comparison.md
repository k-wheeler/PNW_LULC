# COLD vs CCDC: break-date comparison on the Holiday Farm Fire

Change-dating comparison between **CCDC** (`ee.Algorithms.TemporalSegmentation.Ccdc`, run in
Google Earth Engine) and **COLD** (Zhu et al. 2020, run locally via the `pyxccd` library) on the
2020 Holiday Farm Fire pilot. Both algorithms are dated against the known ignition date,
**2020-09-08**. Each is run at two confirmation-threshold settings so the comparison is fair
(CCDC's `minObservations` and COLD's `conse` are the same lever: the number of consecutive
residual-exceeding observations required to confirm a break).

## Sample

50 points drawn from the classifier's own Trees to non-Trees change map (same seed as the pilot):

- **30 random** change points (any severity, any transition)
- **10 stratified Trees to Water** points
- **10 stratified Trees to Developed** points

Water and Developed transitions are largely classifier noise rather than real fire change, and
are included deliberately as a false-positive check.

## Methods compared

| Label | Algorithm | Confirmation setting | Where it runs |
|---|---|---|---|
| CCDC minObs=6 | CCDC | `minObservations=6` (default) | GEE (server-side) |
| CCDC minObs=4 | CCDC | `minObservations=4` (pilot's tuned value) | GEE (server-side) |
| COLD conse=6 | COLD (pyxccd) | `conse=6` (default) | local (isolated venv) |
| COLD conse=4 | COLD (pyxccd) | `conse=4` | local (isolated venv) |

All four use `p_cg / chiSquareProbability = 0.99`, `lambda = 20`, and the same Landsat 7/8/9
Collection-2 surface-reflectance observations over 2017-01 to 2026-08.

## Accuracy

Break date measured against the 2020-09-08 ignition, over the 50 points.

| Method | Breaks found | On-target (within 25 days) | Spurious pre-fire | Median distance from ignition |
|---|---|---|---|---|
| CCDC minObs=6 | 36 / 50 | 15 | 1 | 176 days |
| CCDC minObs=4 | 41 / 50 | 22 | 2 | 8 days |
| COLD conse=6 | 42 / 50 | 23 | 1 | 8 days |
| **COLD conse=4** | **45 / 50** | **30** | **1** | **1 day** |

- **On-target** = break within +/-25 days of the ignition.
- **Spurious pre-fire** = break more than 40 days *before* the ignition (a likely false positive).
- **Median distance** = median of the absolute distance in days from the ignition, over points
  that had a break. Because most points sit inside the fire, closer to zero is better; the metric
  does fold in genuine non-fire changes (some Developed/Water points), so read it as a relative
  comparison across methods on identical points, not an absolute accuracy.

**COLD dates the fire markedly closer than CCDC at matched settings**, and with no increase in
spurious pre-fire breaks. At the tuned setting, COLD conse=4 puts 30 of 50 points within 25 days
of the ignition (vs 22 for CCDC minObs=4) and has a median distance of 1 day. Even at its default
`conse=6`, COLD matches CCDC's *tuned* minObs=4 result.

## Timing

| Method | Total (50 points) | Per point | Notes |
|---|---|---|---|
| CCDC minObs=6 | 321.0 s | 6.42 s | GEE server-side compute + one `getInfo` per point |
| CCDC minObs=4 | 281.5 s | 5.63 s | same |
| COLD band extraction | 322.3 s | 6.45 s | one-time GEE `getRegion` per point (shared by both COLD runs) |
| COLD conse=6 | 0.343 s | 0.007 s | local compute only |
| COLD conse=4 | 0.342 s | 0.007 s | local compute only |

**Reading the timings.** The two algorithms spend their time in completely different places.
CCDC's cost is dominated by GEE round-trips: every configuration re-run pays the full ~5-6 s per
point again (the two CCDC runs cost ~600 s combined). COLD's cost is almost entirely the one-time
band extraction from GEE (~6.5 s per point); once the observations are on disk, the COLD algorithm
itself runs in **~7 milliseconds per point**, roughly a thousand times faster than a CCDC round-trip.
So re-running or re-tuning COLD (e.g. sweeping `conse`) is effectively free, whereas each CCDC
re-tune is another full GEE pass. End-to-end for a single setting the two are comparable
(~6.5 s/point); COLD pulls ahead sharply the moment you iterate.

This is a 50-point test, so the absolute numbers are small and local (about zero dollars). At
wall-to-wall scale COLD's economics invert: it must move Landsat stacks out of GEE (ARD tiles,
egress, a VM), which is where its real cost lives.

## Select time series

Each panel shows the point's clear-observation NDVI (green) and NBR / burn index (red), the fire
ignition (black dotted), and all four break dates: CCDC minObs=6 (light blue dotted), CCDC minObs=4
(dark blue dashed), COLD conse=6 (light purple dash-dot), COLD conse=4 (solid purple).

### 1. Agreement on a clean fire pixel (point 16, high severity)

All four methods land on 2020-09-08, exactly on the NDVI/NBR crash. On unambiguous stand-replacing
pixels the two algorithms agree.

![agreement](CCDC_Outputs/cold_ccdc_comparison_n30_water10_dev10/agreement_pid16.png)

### 2. COLD fixes a CCDC lag (point 24, high severity)

The crash is unmistakably at September 2020. Both COLD settings date it to 2020-09-08; both CCDC
settings lag ~6 months to 2021-03-03, landing deep in the post-fire trough. This is a genuine
high-severity fire pixel that CCDC mis-dates and COLD gets right.

![cold fixes lag](CCDC_Outputs/cold_ccdc_comparison_n30_water10_dev10/cold_fixes_lag_pid24.png)

### 3. The confirmation-threshold lever affects COLD too (point 47, Developed)

COLD conse=4 recovers the September 2020 date, while COLD conse=6 and CCDC minObs=4 both lag to
2021-03-28. The `conse` / `minObs` setting matters for both algorithms; COLD at conse=4 simply
recovers more of the true fire timing.

![conse sensitivity](CCDC_Outputs/cold_ccdc_comparison_n30_water10_dev10/conse_sensitivity_pid47.png)

### 4. Classifier noise, correctly left undated (point 33, Water)

A stratified Trees to Water point with no real discontinuity. No method reports a break, which is
the correct null result: absence of a break is the signal that the classifier's "change" here is
noise, not a real event.

![water no break](CCDC_Outputs/cold_ccdc_comparison_n30_water10_dev10/water_no_break_pid33.png)

The full 50-page per-point comparison is in
[`CCDC_Outputs/cold_vs_ccdc_timeseries_n30_water10_dev10.pdf`](CCDC_Outputs/cold_vs_ccdc_timeseries_n30_water10_dev10.pdf),
and the per-point break table is in
[`CCDC_Outputs/cold_vs_ccdc_breaks_n30_water10_dev10.csv`](CCDC_Outputs/cold_vs_ccdc_breaks_n30_water10_dev10.csv).

## Caveats

- **One fire, one disturbance type.** This tests abrupt disturbance dating, which is COLD's
  designed strength. It does not establish that COLD is better for gradual change.
- **Not a masking-controlled comparison.** COLD used a cfmask decoded from Landsat `QA_PIXEL`;
  GEE-CCDC used its own internal masking. The input scenes are the same, so this is a fair
  "as-run" comparison, not a pure algorithm-only diff.
- **Off-GEE dependency.** COLD is not available in Earth Engine; it runs through `pyxccd` in an
  isolated environment (it requires numpy 2.x, which conflicts with the project's `gee` env). For
  a point test that is trivial and free; at wall-to-wall scale it means moving data out of GEE.
- **Metric mixes strata.** The accuracy metrics treat "near 2020-09-08" as correct, which is right
  for fire pixels but not necessarily for the Developed/Water stratified points. All methods run on
  identical points, so the comparison is valid; the absolute numbers should not be read as a
  fire-only accuracy.

## Reproducibility

- CCDC: `Change_Detection.run_ccdc(..., min_observations=6|4)` over the 50 points.
- COLD: `pyxccd 1.0.3`, `cold_detect(..., conse=6|4, p_cg=0.99, lam=20, b_c2=True)`, fed Landsat
  C2 L2 surface reflectance scaled to [0, 10000], thermal as Kelvin x 10, and a cfmask QA decoded
  from `QA_PIXEL`. Validated by confirming a known high-severity pixel dates to 2020-09-08 before
  trusting the batch.
