# WP018 — Score-state dependence: a joint outcome distribution off the Poisson surface

**Status: complete. Verdict: the mechanism is REAL and correctly diagnosed, but it cannot help.** State dependence nearly eliminates the home/away tilt WP017 proved unreachable (+0.0137 → +0.0023, an 83% reduction) — and simultaneously overshoots into draws (+0.0015 → +0.0271), leaving RPS unchanged. **Even the in-sample best over the whole parameter grid gains only +0.00008**, about 1/30th of the xG channel. The ceiling is essentially zero, so this isn't a calibration failure.

No retraining — this is a prediction-time transformation applied to WP013's existing checkpoints. Minutes of compute against WP017's 5.7 hours.

## Why this was the lead worth taking

WP017 established something specific: given `λ_home` and `λ_away`, the three outcome probabilities are **fully determined** by convolving two independent Poissons. Re-weighting the training objective toward outcomes only slides along that surface — it cannot reach a point lying off it, which is why weights spanning 6× moved the tilt the *wrong* way.

So the question became: what mechanism would put the truth off that surface? The obvious candidate is that goals don't arrive at a constant rate — a trailing side pushes, which makes the two goal counts dependent in a way independent Poisson cannot express and Dixon-Coles' four-cell patch cannot fix.

## The diagnostic — and a confound I walked into first

**First attempt was wrong.** Measuring rate by score state across all 4,560 matches showed the *leading* side scoring **more** (home rate 1.420 at level → 1.742 when two ahead), the opposite of the hypothesis. That is **selection, not causation**: "home leads by two" is populated almost entirely by strong teams beating weak ones. I built exactly this control into WP011 ("is it just bad teams?") and then forgot it here.

**Controlled for pre-match mismatch** (bottom quartile of |P(home) − P(away)| from pre-close Pinnacle odds, 960 matches), the real effect appears:

| Home lead | home rate | away rate |
|---|---|---|
| −1 | **1.344** | 1.266 |
| 0 | 1.227 | 1.187 |
| +1 | 1.241 | **1.363** |

The **trailing** side's rate rises ~15% (1.187 → 1.363); the leader's barely moves. In the *mismatched* quartile the same table reads 1.876/1.027 at level and 2.108/0.869 at +1 — pure team quality, which is what swamped the raw pass.

A prior version of the shot cache also lacked the `minute` field (trimmed "to keep the cache small"), so every goal looked like minute 0. Re-fetched with `minute`/`X`/`Y`, and the fetcher now tops up thin caches rather than skipping them.

## The implementation

Exact **dynamic programming** over (time step, home goals, away goals) rather than simulation, so there is no Monte Carlo noise to mistake for an effect. Two properties, both unit-tested:

1. **With the state effect off it reproduces independent Poisson** — outcome-level error 1.1e-4 at 16 steps/minute, converging first-order (9.0e-4 at 2 steps, halving as steps double).
2. **Marginal expected goals are calibrated back to the input λ**, so any change comes from the *dependence* and not a shifted mean.

## Results

| | H bias | D bias | A bias | pooled RPS |
|---|---|---|---|---|
| Independent Poisson | **+0.0137** | +0.0015 | −0.0152 | 0.20481 |
| State-dependent | **+0.0023** | **+0.0271** | −0.0294 | 0.20493 |

| Comparison | ΔRPS | CI (cluster) |
|---|---|---|
| state-dep vs analytic Poisson | +0.00012 | [−0.00034, +0.00058] |
| state-dep vs DP-with-effect-off *(isolates dependence at identical discretisation)* | +0.00012 | [−0.00034, +0.00057] |
| **multipliers fitted on held-out windows** | **+0.00030** | [−0.00036, +0.00091] |
| *in-sample best over the entire grid (hindsight, invalid as a result)* | **+0.00008** | — |

Two things make this conclusive rather than merely null:

- **The in-sample ceiling is +0.00008.** Fitting and scoring on the same data — cheating as hard as possible — the best achievable gain is 1/30th of the xG channel's. No calibration of this mechanism can help.
- **The two folds chose contradictory multipliers**: (lead 0.90, trail 1.15) and (lead 0.95, trail 0.95). A real optimum would show agreement.

## Why it can't work, mechanically

The home-tilt fix and the draw overshoot are **the same effect**. Negative dependence — a goal raising the opponent's subsequent rate — pulls scorelines back toward level. That simultaneously reduces the home skew *and* inflates draw mass, in a ratio fixed by the structure. Dialling the multipliers down shrinks both proportionally; there is no setting that buys one without the other.

**That is a genuinely useful narrowing of WP017's conclusion.** A state-dependent arrival process mechanically *couples* the home/away tilt and the draw mass. Fixing the tilt without breaking draws needs a structure with **separate parameters** for the two — a copula with an asymmetric dependence term, or a direct multinomial outcome model — not a goal-arrival process, however well calibrated.

## What's reusable

`state_model.py` is a validated exact joint-scoreline DP with calibrated marginals. Any future work needing a non-independent joint distribution (a copula, an asymmetric dependence structure) can score it the same way — against existing checkpoints, no retraining.

## Caveats

* The controlled diagnostic rests on **960 evenly-matched fixtures**; the ~15% trailing-side effect is measured, not precise.
* Multipliers are constant within a state. Real state dependence plausibly varies with time remaining — a one-goal lead at minute 85 is not a one-goal lead at minute 20. Untested, but given the +0.00008 ceiling it is very unlikely to rescue the approach.
* The DP carries a residual ~1e-4 outcome-level discretisation bias. That is ~1% of the effects examined and the same in both arms of the paired comparison.

## Reproducing

```bash
python work_products/wp018_score_state/diagnose_state.py   # raw (selection-confounded) state rates
# the mismatch-controlled version and the RPS comparison are in this README's tables;
# state_model.py is unit-tested in tests/test_wp018_state_model.py
```
