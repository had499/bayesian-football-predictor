# WP055 — Deployment-accurate measurement: the horizon reframing does NOT hold

**Verdict: refitting every round is worth −0.00037. The gap to Pinnacle's opening line
goes from +0.00319 to +0.00282 — still CI-excluding-zero, nowhere near parity. The
115-window grid was NOT materially misleading.**

    model (all rounds at horizon 1)   0.19665
    vs Pinnacle OPENING   0.19295    gap +0.00282 [+0.00162,+0.00404]   market ahead
    vs Pinnacle CLOSING   0.19192    gap +0.00385 [+0.00249,+0.00524]   market ahead
    vs bet365  OPENING    0.19400    gap +0.00266 [+0.00149,+0.00387]   market ahead

    old TEST_LEN=3 grid: model 0.19692, gap vs OPEN +0.00319 [+0.00198,+0.00442]

343 windows, STEP = TEST_LEN = 1, the same 3,780 test matches, 0 failures, 156 min.

## The hypothesis, and why it was wrong

Every result in this project came from STEP = TEST_LEN = 3, so each window predicts THREE
rounds ahead and the reported RPS averages forecast horizons 1-3. Deployment refits every
round and only ever predicts the next one. Subsetting the old grid to each window's FIRST
test round showed:

    horizon   n     model RPS  market RPS      gap vs Pinnacle OPEN
          1  1205     0.19162     0.19068   +0.00093 [-0.00101,+0.00297]  CI SPANS ZERO
          2  1202     0.19885     0.19407   +0.00478
          3  1190     0.19796     0.19411   +0.00385

I then decomposed: the market degrades +0.0034 from h1 to h2, so that is match
difficulty; the model degrades +0.0072, so the extra +0.0038 must be horizon cost that
refitting would recover. That implied a deployed gap near +0.0010.

**The attribution was invalid.** The model and the market have DIFFERENT SENSITIVITIES to
match characteristics, so their differential degradation across the horizon split is not
attributable to horizon — it is mostly COMPOSITION. The rounds that only ever appear at
h2/h3 are harder for this model in ways that persist at h1. Measured: predicting ALL
rounds at horizon 1 gives 0.19665, not the 0.19162 the easy-third subset showed.

The warning sign was in the data and I talked past it: the MARKET scored 0.19068 on the
horizon-1 subset versus 0.19295 overall, i.e. those rounds were easier for everyone. I
noted that, then built a decomposition assuming it could be netted out.

**Same error shape as WP032's partial-sweep read and WP035's one-window reads**: a
non-random subset extrapolated to the whole.

## What IS established

* **Refitting every round is worth −0.00037.** Small, real, and free if you are deploying
  anyway.
* **The TEST_LEN = 1 grid is the deployment-accurate protocol** and is now built
  (`cv_shared_data_h1.pkl`, 343 windows). Future arms should be measured on it.
* **The 115-window grid overstated the gap by 0.0004, not 0.002.** Every arm result in
  this project stands roughly as measured — the design choice was not a serious flaw.
* We are **not** more accurate than bet365's opening line either (+0.00266 behind), so no
  "better than the bookies" claim is supportable.

    python work_products/wp055_horizon1/prepare_data.py
    python work_products/wp055_horizon1/run_cv.py --workers 6
