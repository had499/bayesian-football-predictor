# WP056 — five never-used feature families for the GBM: NULL

Prompted by the user: "Is there any other features we can use in the GBM that aren't being
used anywhere so far? Previous lineup data? Injuries? Anything!"

The premise was right. WP053's 253 features are ALL team-match aggregates, and five
families were sitting on disk untouched. `build.py` adds **228 features** from them, with
the old 253 reproduced at exact parity (asserted), on wp055's 343-window horizon-1 grid.

| family | n | source | why it should have worked |
|---|---|---|---|
| A lineup history | 8 qty | `wp030/player_minutes.pkl` (207k rows, pos + importance) | XI Jaccard vs last match, **GK continuity**, rotation, minutes HHI, importance-weighted XI, individual 7/21-day minute load. Used once before as a covariate for the CURRENT team sheet (illegal at the open); from PLAYED matches it is legal and had never been a feature. WP016 noted the absence nulls never tested goalkeepers at all |
| C score-state exposure | 8 qty | shot feed `minute` | minutes leading/level/trailing, xG while level vs trailing, xG conceded while leading, late (75'+) xG. WP018 proved score-state dependence is REAL, and rolling xG is biased by it |
| D shot result / location | 6 qty | shot feed `result`, `Y` | faced-shot SAVE rate = direct GK shot-stopping, the quantity WP016 failed to isolate; block rate, woodwork, central xG |
| E referee | 4 | football-data `Referee` | 100% coverage, 51 refs, rolling cards / home-points bias. Never used anywhere |
| B head-to-head | 4 | panel | prior meetings' GD and xGD, venue-specific. Never tried |

`minute`, `Y` and `result` had literally never been read from the shot feed.

## Result (`gbm.py`): the new families add NOTHING

Four feature sets, each tuned on windows 70-200 then LOCKED before 201-343. n=1,558 matches.

| arm | feats | dRPS @beta=0.25 | dRPS @beta=0.50 | dRPS @beta=1.00 |
|---|---|---|---|---|
| OLD | 253 | −0.00022 [−0.00044,−0.00000] | −0.00040 [−0.00083,+0.00005] | −0.00059 [−0.00146,+0.00030] |
| **NEW** | 228 | −0.00010 [−0.00034,+0.00013] | **−0.00014 [−0.00060,+0.00033]** | −0.00000 [−0.00094,+0.00093] |
| **BOTH** | 481 | −0.00022 [−0.00046,+0.00002] | **−0.00039 [−0.00086,+0.00010]** | −0.00052 [−0.00146,+0.00046] |
| +MKT | 483 | −0.00022 | −0.00039 | −0.00052 |

**BOTH ≈ OLD to within 0.00001.** The 228 new features are entirely redundant with the
team-match aggregates. **NEW alone is null**, so the information does not even stand up by
itself. Coverage is not the explanation: NEW features are 99.8% covered in the evaluation
range (min 74.7%), none constant — this is a fair, well-covered null.

## Two further findings

**1. The GBM wrapper is WEAKER on the deployment-accurate grid.** −0.00040
[−0.00083,+0.00005] at beta=0.5 here, against WP053's −0.00063 [−0.00112,−0.00015] on the
115-window TEST_LEN=3 grid. The CI now INCLUDES zero; only beta=0.25 excludes it. This also
**falsifies my own horizon prediction** — the earlier decomposition (−0.00106 at horizon 1
vs −0.00040 at horizon 3) suggested the wrapper should be LARGER at horizon 1. It is not.
That is the second time a horizon-based prediction from that decomposition has failed.

**2. The +MKT ceiling arm is INCONCLUSIVE, not null.** Identical to BOTH to 5 d.p., which
looked like "the market adds nothing" — but the market features are only **49-56% covered**
in the evaluation range and are 2 of 483 features, so the trees never split on them
meaningfully. Not a valid ceiling test. The low coverage is a join problem in
`market_extra()` (long team names + date), not missing odds. Not worth fixing: the arm is
circular and unshippable either way.

## What this closes

The squad/lineup direction is now dead for the **third** time, by a different mechanism each
time: as a model covariate (WP011 instability, WP012 absence, WP030 intactness), as the
exogenous-absence test (WP047 reds), and now as **leak-safe GBM features**. Specifically
closed: **goalkeeper continuity and faced-shot save rate**, the gap WP016 identified and
nothing had tested. Also null as features: score-state exposure (despite WP018 proving the
EFFECT is real), referee, head-to-head.

Lesson consistent with [[feedback-only-new-observations-win]]: a real *effect* (score-state
dependence) does not imply usable *predictive features* for it.
