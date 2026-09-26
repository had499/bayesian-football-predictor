# WP011 — Does squad instability explain where the model loses to the market?

**Status: complete. Verdict: null on the channel tested — but see the scope limit below, which is narrower than the first draft of this README claimed.** Squad instability, rotation and key-contributor absence show **no detectable contemporaneous relationship** to where the model's predictions diverge from the market's: all five tests land within noise, and all four features together explain just **0.87% of the variance** in model-minus-market supremacy.

**What this does not establish:** that adding these features to the model wouldn't help. This work product regressed squad features against *existing* predictions; it never trained a model with them. That misses at least one real mechanism — see "Why a correlational null is weaker than a sweep" below.

Diagnostic only — no model changes, no CV sweep. One roster fetch (~37 min) plus seconds of analysis.

## Why this was the next question

WP010 established that ~71% of the model's gap to the closing line exists against the *pre-closing* line too — a gap at equal timing but not equal information, since the market already prices squad state days ahead while this model carries no player-level information at all. WP010 also found the model's *entire* deficit sits in the top 25% of matches by line movement.

The obvious candidate: the market knows who's playing and the model doesn't. This tests that directly, before committing to a player-level rebuild.

## Pre-registered criteria (fixed before running)

- **Positive** → build the player-level model: the crux test shows a clear directional effect, CI excluding zero, big enough to plausibly account for a real chunk of the 0.0039 RPS gap.
- **Negative** → the lineup direction is dead; stop proposing it.
- **Ambiguous** (real but tiny) → given the gap is only 0.0039, tiny means no.

## Data and features

2,658 matches of Understat per-player rosters (2019–2025; minutes, position, xG, xA) — same provider the project already uses, deeper endpoint, no new vendor. Test period is 2020+; 2019 is lead-in for trailing history.

Four leak-safe team-match features, each computed **only** from matches strictly before the target date:

| Feature | What it measures |
|---|---|
| `instability` | share of trailing minutes not going to the top-11 — rotation breadth |
| `absent_share` | share of recent xG+xA from players not expected to be on the pitch |
| `absent_drop` | share of recent xG+xA from players whose availability has fallen **below their own norm** |
| `staleness` | how unlike the current expected XI were the XIs that produced recent results |

Face validity checks out: the features independently identify **Man City as the league's heaviest rotator** (highest `instability`, but *low* absence — rotation by choice, not unavailability) and **Sheffield United, Wolves, Burnley, Liverpool as the most settled** — all famously small-squad or unchanged-XI sides in 2019-20.

### Two feature-design problems found and fixed en route

1. **A hard bench-minutes threshold fired far too late.** At half-life 3, a player who stops playing still shows ~37 expected minutes after three missed matches, so any sane cutoff needed *four* missed games to register — long after the market reprices. Replaced with continuous availability weighting, which registers after one missed match.
2. **`absent_share` turned out to carry a ~0.31 structural baseline for every team.** In any squad roughly a third of xG comes from players who don't average a full 90 (subs, rotation), so it was scoring squad-production *dispersion*, not "our key man is out" — Chelsea and Ipswich topped it, not injury-hit sides. That prompted `absent_drop`, which measures each player's fall against **their own** baseline, so a rotation player who always plays 45 scores 0 while a stopped starter scores ~1.

Without the second fix the null would have been uninterpretable — a mis-specified feature can't rule anything out.

## Results — all five null

| Test | Result |
|---|---|
| **1. Predicts line movement?** (is instability a news proxy?) | No — all coefficients' CIs include zero |
| **2. Predicts disagreement with market?** (primary low-noise chain) | No — `max_absent_drop` +0.004 [−0.033, +0.042]; flat across buckets |
| **3. Explains model-minus-market supremacy?** (THE CRUX) | **No** — see below |
| **4. Predicts realised model error?** (weaker instrument) | No — CIs include zero |
| **5. Are away favourites less stable?** (WP009 loose end) | **No** — see below |

### The crux test, and a near-miss worth recording

The first run of Test 3 appeared to **confirm** the hypothesis: `d_absent_drop` = +0.135, CI [+0.011, +0.253], positive exactly as predicted. That result is **an artifact and should not be believed.**

`absent_share` and `absent_drop` correlate at 0.76. Fitted together they split into large opposite-signed coefficients (+0.135 and −0.142) that partly cancel — textbook multicollinearity. Tested cleanly:

| Specification | `d_absent_drop` | CI (cluster) |
|---|---|---|
| Alongside its correlated twin | +0.135 | [+0.011, +0.253] ← **artifact** |
| **Alone** | **+0.052** | **[−0.027, +0.139]** — includes zero |
| Alone + market-supremacy control | −0.005 | [−0.086, +0.077] — flips to zero |
| `absent_share` alone | +0.002 | [−0.071, +0.081] — includes zero |

It also failed the confound check that was part of the original design: controlling for market supremacy (i.e. "is this just *bad* teams?") killed it even in the collinear specification.

**Practical size**, taking even the inflated estimate at face value: a 1-SD (0.107) swing in absence-drop moves model-minus-market supremacy by **0.56 percentage points**. Against a 0.0039 RPS gap that is negligible regardless of significance.

`run_diagnostics.py` now prints the clean single-feature specification as its verdict, so the misleading version can't be read off the output again.

### Test 5 also kills the away-favourite hypothesis

WP009 found the model specifically under-rates away favourites, and I proposed that bigger clubs rotate more for European fixtures, making their strength stalest. The data says no:

| | n | favourite `instability` | favourite `absent_share` | model_worse |
|---|---|---|---|---|
| Home favourite | 713 | 0.1898 | 0.3105 | +0.00106 [−0.00227, +0.00432] |
| Away favourite | 473 | 0.1884 | 0.3050 | +0.00817 [+0.00360, +0.01256] |

Squad instability is **essentially identical** between the two (0.1884 vs 0.1898). The away-favourite deficit is real and persists, but rotation does not explain it. That hypothesis is dead too.

## What this does and doesn't rule out

**Does**: these team-level squad aggregates carry no detectable *contemporaneous* information about where the model diverges from a line set at the same time. All four together explain 0.87% of the variance in that residual — a genuinely tight bound on any large effect through this channel.

**Doesn't** — and this is a real limit, not a hedge:

- **It never tested the latent-cleaning channel.** With `absent_drop` in the model *during training*, the attack/defence random walk would attribute a bad run to the absence rather than to "this team got worse," keeping the latent correct and improving predictions on *later* matches where absence is zero. A per-match regression against finished predictions cannot see that at all.
- **The test was linear and contemporaneous** against a non-linear, hierarchical, temporal model.
- **It collapses player-level information into one scalar per team**, when *which* player is absent matters enormously.
- **The CI is not that tight.** `absent_drop` alone is +0.052 [−0.027, +0.139]; the upper bound is 1.49pp of supremacy per 1-SD swing, ~13% of one residual SD. A modest effect is not excluded.

### Why a correlational null is weaker than a sweep

The precondition tested here — "does this feature correlate with the residual?" — is a cheap filter, not a proof. It was chosen because a player-level model is weeks of work and the same filter killed rest days in WP009 in minutes. It is good evidence against a *large*, directly-priced effect. It is **not** equivalent to adding the feature and measuring RPS.

The direct test is one new CV arm — `L5_xgchannel` plus `absent_drop` as a covariate — paired against `L5` over the same 30 windows. **This has now been run: see WP012 (`work_products/wp012_absence_covariate/`). It is also null** (+0.00011 RPS, CI [−0.00025, +0.00044]), so the correlational and in-model tests agree. WP012 also found the fitted coefficient comes out *positive*, opposite to the hypothesis — evidence that `absent_drop` is an imperfect proxy for 'this side is weakened'.

## Where that leaves the project

Now ruled out as explanations for the remaining gap: recalibration (WP004), prior loosening (WP005), per-team sigma (WP006), random-walk dynamics / intercept / promotion offset / Dixon-Coles (WP007), post-hoc sharpening (WP009) and rest days (WP009) — all of which were tested by actually fitting and scoring. Squad instability (this WP) is ruled out only through the weaker correlational channel described above. The gap is +0.0039 RPS against the pre-close line, the model is already at parity on ~75% of matches and in 3 of 6 seasons, and the deficit is confined to the high-line-movement quarter.

The honest reading is that the remaining edge is small, hard, and not reachable by the classes of change tried so far. If the aim is a good working model rather than beating Pinnacle, this is a reasonable place to stop — and five work products are still uncommitted.

## Reproducing

```bash
# one-off, ~37 min, resumable (set REQUESTS_CA_BUNDLE/SSL_CERT_FILE behind TLS interception)
python work_products/wp011_squad_instability/fetch_rosters.py

python work_products/wp011_squad_instability/run_diagnostics.py   # seconds
```

Needs WP007's `checkpoints/` and WP008's `odds_raw.pkl`. Feature logic is unit-tested in `tests/test_wp011_features.py`, including a leak-safety test asserting that a match's own roster cannot change its own features.
