# WP004 — Recalibration Layer (diagnostic)

**Status: complete. Conclusion — recalibration is NOT the lever. The model's gap to the market is a *resolution* problem (its match rankings are genuinely worse), not a *calibration* problem. Cheap fix ruled out; next step is structural (WP005).**

## The question

WP003 showed the model's RPS gap to Pinnacle is **+0.0139** (CI [+0.0080, +0.0198], ~8% worse), that the model over-shrinks (under-confident on home favourites), and that its gap to the market *grows* with disagreement. The single cheapest thing to try next: fit a cross-validated recalibration map on the model's own CV predictions and see how much of that gap it closes. If most → the model's rankings were fine and it was pure miscalibration, ship a one-line transform. If little → the model genuinely ranks matches worse and needs structural work.

## Method

No retraining. Reuses WP001's 401 CV match predictions and WP003's odds join / scoring harness verbatim. Four recalibration maps, all **cross-fitted leave-one-window-out** (to recalibrate window *w*, the map is fit on the other 34 windows — no fixture's recalibrated probability sees its own outcome):

- **temperature** — `softmax(log(p) / T)`, one scalar `T` fit by minimising multiclass log-loss. `T<1` sharpens. The clean test of "is it just global under-confidence."
- **temperature (RPS-fit)** — same, `T` chosen to minimise mean RPS directly. Sensitivity check.
- **vector Platt** — per-class 2-parameter logistic on `logit(p_c)`, renormalised. Can fix per-outcome bias too.
- **isotonic** — per-class monotonic non-parametric fit, renormalised. Most capacity.

## Results

### Fitted temperature

`T ≈ 0.919` (log-loss fit), very stable across folds (sd 0.015, range [0.877, 0.945]); `T ≈ 0.882` for the RPS-fit variant. `T < 1` confirms the model *was* mildly under-confident and wants sharpening — but only mildly.

### Pooled RPS (401 matches)

| Predictor | RPS | 95% CI |
|---|---|---|
| model (raw) | 0.1963 | [0.1851, 0.2081] |
| model + temperature | 0.1965 | [0.1845, 0.2090] |
| model + temperature (RPS-fit) | 0.1966 | [0.1842, 0.2095] |
| model + vector Platt | 0.1966 | [0.1832, 0.2105] |
| model + isotonic | 0.2037 | [0.1895, 0.2187] |
| Bet365 | 0.1833 | — |
| Pinnacle | 0.1786 | — |
| naive | 0.2341 | — |

**No map improves RPS.** Temperature and Platt are flat (within noise, marginally worse). Isotonic is clearly *worse* — it overfits the ~390-match calibration set.

### Gap to Pinnacle closed (paired bootstrap, n=361)

| Candidate | Gap vs Pinnacle | 95% CI | % of raw gap closed |
|---|---|---|---|
| raw | +0.0139 | [+0.0080, +0.0198] | 0% |
| temperature | +0.0137 | [+0.0077, +0.0197] | **2%** |
| temperature (RPS-fit) | +0.0136 | [+0.0075, +0.0197] | 2% |
| vector Platt | +0.0132 | [+0.0065, +0.0198] | 5% |
| isotonic | +0.0205 | [+0.0131, +0.0281] | −48% |

Temperature scaling — even fit directly to RPS — closes **2%** of the gap. Effectively nothing.

### Calibration — recalibration *does* fix the curve, it just doesn't matter for score

| Bin | Raw pred / actual | +Temperature pred / actual | Pinnacle pred / actual |
|---|---|---|---|
| 0.12–0.25 | 0.195 / 0.172 | 0.198 / 0.203 | 0.200 / 0.203 |
| 0.38–0.50 | 0.437 / 0.465 | 0.439 / 0.452 | 0.438 / 0.430 |
| 0.62–0.75 | 0.669 / 0.750 | 0.679 / 0.756 | 0.687 / 0.881 |
| 0.75–0.88 | 0.782 / **0.929** | 0.795 / **0.812** | 0.807 / 0.818 |

Temperature scaling genuinely repairs the WP003 defect: the 0.75–0.88 bin (14 favourites the raw model called at 0.78 when they won 93% of the time) tightens to 0.80/0.81, and the low bins line up too. The recalibrated curve is about as good as Pinnacle's. **And the pooled RPS is unchanged.** That is the whole finding: proper scores decompose into calibration + resolution, temperature scaling is a monotonic (resolution-preserving) transform, so it can move the calibration term to zero and still not help — because the gap to the market lives in the *resolution* term. The model isn't systematically mis-scaled; it sorts the wrong matches into the wrong buckets, and no recalibration can reorder them.

(Note also: even the raw model was only badly miscalibrated in the top two thin bins, n=40 and n=14 — and Pinnacle's own 0.62–0.75 bin is *worse* calibrated than the model's. WP003's "the model is badly miscalibrated" read was somewhat overstated; the real gap was never mostly calibration.)

### Disagreement pattern — not fixed

Raw model's gap to Pinnacle grew with disagreement (+0.014 all → +0.041 top-10%). After temperature scaling: +0.014 → +0.046. Vector Platt makes the tail *worse* (+0.084 top-10%). Recalibration does not touch the "most wrong exactly where it's most confident it disagrees with the line" pattern.

### Betting sim — still unprofitable

Temperature: −17% ROI at τ=0 (raw was −17%). Vector Platt: −12% at τ=0 (raw −17%) — a small improvement, but the CI still straddles/sits below zero and it degrades at higher thresholds. No map produces a positive-expectation strategy.

## Verdict

**Do not ship a recalibration layer as a performance fix.** Temperature scaling with `T ≈ 0.92` is a legitimate cosmetic tidy-up if raw probabilities are ever shown to users (it makes the reliability curve honest, especially on strong favourites), and it costs one line — `softmax(log(p) / 0.92)` on `dc_outcome_probs` output. But for RPS, for closing the gap to the market, and for betting, it does nothing.

The gap to the market is **structural resolution**, not calibration. The model's match-by-match probability *ordering* is genuinely less accurate than the market's, and that is not recoverable post-hoc. **WP005** is the real work: changes that improve resolution —

- **Loosen the shrinkage priors.** `sigma_att`/`sigma_def ~ HalfNormal(0.008)` and `home_adv_sd = 0.02` are very tight; the AR(1) `rho ~ Beta(29,1) ≈ 0.97` with tiny innovation makes strengths extremely smooth. All three pull team strengths toward the population mean — exactly the over-shrinkage signature. Widen them (targeted mini-sweep) so standout teams can actually separate.
- **Per-team `sigma` (partial pooling)** so high-variance / high-ceiling teams aren't forced to evolve at the league-average rate.
- **New ranking signal** (rest days, fixture congestion, a market-informed prior at season start) — but only after the prior work, since WP002 showed features added on top of an over-shrunk core don't move the needle.

WP003's 401-match comparison stays the yardstick: a WP005 change earns its keep only if it closes the gap to Pinnacle there.

## Reproducing

```bash
cd work_products/wp004_recalibration
jupyter lab wp004_recalibration.ipynb
```

Run top to bottom — pure numpy/scipy/sklearn, no PyMC, seconds. Requires `../wp003_bookmaker_benchmark/odds_raw.pkl` (produced by running WP003 once).
