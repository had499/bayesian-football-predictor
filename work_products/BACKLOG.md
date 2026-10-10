# Research backlog

**The current model is `ModelConfig.champion()` — see [CHAMPION.md](CHAMPION.md).**

Ideas not yet tested, with why they might work and what they'd cost. Kept here so
they don't get lost between work products. Ordered roughly by expected value.

**Updated through WP029.** The organising principle that emerged from WP004–WP014: **information already
reflected in past results is largely redundant**, because the attack/defence walks
absorb it. What pays is either (a) a parameter the model is forced to hold constant
where reality varies, or (b) information that is knowable before kickoff but
invisible in past results.

---

## Closed by diagnostics (2026-09-29) — do not start these

| Idea | Why it's dead |
|---|---|
| ~~Multi-league hierarchical hyperparameters~~ | **Rationale FALSE.** It existed to get ~5x data for "badly pinned" hyperparameters. On mature windows they are pinned to 0.1-2% (`rho_att` 0.1%, `deep_kappa` 0.2%, `sigma_att` 1.1%, `home_mu` 1.4%). Only `home_sd` is loose (14%), and WP015 showed HFA variation is null. This was the largest job on the backlog and would have bought nothing. |
| ~~Opponent-interaction / style effects~~ | **NULL.** Tested four style interactions against the champion's residuals on 3,758 matches (volume x volume-conceded, quality x quality-conceded, deep x deep-conceded, and quality x deep-conceded i.e. "counter-attack vs high line"): all |r| < 0.025, all CIs include zero. The additive `attack - defence` structure is not measurably losing matchup information. |
| ~~Betting/trading the model~~ | **WP024: NOT profitable.** Best tradeable result +0.12% ROI [-6.51%, +6.65%]. Model is well calibrated but trails the opening line by 0.0044 RPS, so apparent value is mostly its own error. Don't revisit until it BEATS the opening line. |

**Still open:** the goalkeeper covariate. A backup keeper (15.4% of team-matches) is
associated with the model under-predicting goals conceded by **+0.1235 +-0.1491** —
right sign, material size, but underpowered because the roster cache covers 2019-2025
while the grid starts 2016 (only 937 of 3,780 matches joined). Fetching 2016-2018
rosters would roughly triple the sample and halve the CI. This is the last
well-motivated idea using obtainable data.

## A. Parameters frozen where reality varies — CLASS LARGELY DISCREDITED BY WP015

This looked like the most promising class. WP015 tested its two best-evidenced
members and both were null, so **downweight the whole class**.

| Idea | Outcome |
|---|---|
| ~~Time-varying home advantage~~ | **WP015: NULL** (+0.00026). And the falsification test failed — fixing HFA did *not* shrink WP009's away-favourite asymmetry, so that link is dead too. The apparent 0.008–0.300 range was a raw spread over season estimates each carrying ±0.062 sampling noise; noise-corrected true SD is **0.069**, and the model learned a walk implying 0.071 — almost exactly right — and it still didn't help. |
| ~~Time-varying league scoring level~~ | **WP015: NULL** (+0.00000, tight CI). Also doubles as the `L2_level`/hard-centering re-test, so that item is closed. |
| Time-varying `sigma_att`/`sigma_def` | Untested, but **downgraded**: same class as the two above, and some of the 0.135–0.334 range is sampling noise. Low priority. |

## RETIRED (2026-10-03) — home advantage as a STEP change: tested in WP031, NULL

> **Do not restart this.** WP031 ran it at 115 windows. `A_crowd` (crowdless indicator)
> +0.00006 [-0.00002,+0.00016]; `B_hfa_wide` (prior 0.03->0.15) -0.00005
> [-0.00020,+0.00009]. The motivating COVID story dissolved: 2024 with FULL crowds had a
> home tilt of -0.2251, nearly as extreme as crowdless 2020's -0.2757, while 2022 was
> +0.1250 -- season-level HFA is volatile and NOT crowd-aligned, and there is no
> permanent post-COVID step. The original reasoning is kept below for the record.


WP015 tested time-varying home advantage as a slow walk and found null (+0.00026), but
its own caveats say verbatim: *"This tests slow-walk drift. An abrupt regime change (a
COVID indicator variable) is a different specification and untested."*

There is now direct evidence the regime DID shift abruptly and has not reverted:

| period | actual home win rate | model home tilt |
|---|---|---|
| 2016-18 | **0.477** | -0.0351 |
| 2019-20 | 0.416 | +0.0267 |
| 2021-25 | **0.442** | +0.0042 |

A walk with sigma ~0.012/round structurally cannot represent a one-season collapse
followed by a partial, PERMANENT recovery -- which is exactly why WP015's slow walk
found nothing. The untested specification is a step/level term on home advantage with a
break at 2020-21 (and possibly a second, partial step back). Cheap: no new data, one or
two parameters. Caveat: per-season tilts rest on 380 matches each, so the individual
numbers are noisy; the pre/COVID/post pattern and the raw win rates are the signal.

## B. Forward-looking information invisible in past results

Available in data already cached:

| Idea | Why it can't be inferred from results | Sizing |
|---|---|---|
| **Goalkeeper absence** | **NEW — WP016 found a real blind spot.** WP011/WP012's absence nulls weighted players by xG+xA; keepers average 0.003 vs 0.193 for outfielders, so a missing first-choice keeper scored ~ZERO absence. Those nulls are "outfield absence doesn't help". Keeper changes occur in 9.4% of matches. Note WP016 kills the *latent-split* route to this; a simple keeper-identity covariate is untested. |
| **Suspensions** | A red card in match N means a ban in **N+1**. The card's effect on N is in the results; the ban on N+1 is purely forward-looking. `red_card`/`yellow_card` are already in WP011's roster cache. | ~45 reds + ~113 players reaching 5 yellows per season → **~41% of matches** involve a suspended player. Caveat: sent-off players average 0.089 xG+xA vs 0.180 squad average (mostly centre-backs — 95 DC, 33 DMC), so attacking impact is below average, but the *defensive* impact isn't captured by xG+xA at all. |
| ~~End-of-season stakes~~ | **DONE — WP028. The effort arm is now IN THE CHAMPION.** Stakes = how much this result moves a team's end-of-season probabilities (3,000 simulations per season-round; relegation/Europe/title; 23.9% of matches are dead rubbers). Entered MULTIPLICATIVELY, `theta = intercept + (1+g*stakes_h)*attack[h] - (1+g*stakes_a)*defence[a] + hfa`, because every additive covariate tried here fitted at ~2% of a prior SD. Result **-0.00014 [-0.00031, +0.00001]**: null by CI but negative in **10/10** leave-one-season-out folds, 7/10 seasons, coefficient 91% of prior SD sign-consistent 115/115, survives residualising against calendar (r -0.004) and strength gap (r -0.089), and needs NO external data. The DISPERSION variant (`NegBinomial` with stakes-dependent alpha) was my predicted winner and is measurably WORSE: +0.00019, CI excludes zero, positive 10/10 folds -- the dead-rubber variance finding (ratio 1.160) was real conditionally and wrong as a global model. |

Needs an external source:

| Idea | Why it matters | Note |
|---|---|---|
| **Managerial change** | Plausibly the largest single effect here — the old regime's results don't predict the new one's. | ~8–10/season, so thin but each is a big shift |
| ~~European / cup fixture congestion~~ | **DEAD — WP027, definitively.** The measurement WAS broken as this entry claimed: rebuilding rest from the complete calendar (league + FA Cup + EFL Cup + CL/EL/UECL) changes 27.9% of matches and raises home sides on <=3 days' rest from 981 to 1,444. But fitted as a covariate it does nothing: rest -0.00000 [-0.00004,+0.00003], load -0.00002 [-0.00006,+0.00001] -- the TIGHTEST intervals in the project, because both coefficients sit at 2% of a prior SD (the model declines them). Congestion is worth <0.00006 RPS, under 2% of the gap. Fixture calendar cached at `wp027_congestion/fixtures_raw.pkl` for any future use. |
| **Transfer-window activity** | A team that just sold its top scorer — results don't reflect it yet. | Detectable retrospectively from squad changes; forward-looking needs transfer news |
| **Returning players** | Inverse of injury: a star back from three months out has decayed out of expected minutes, so the model still prices the team as weakened. | Partially derivable from the roster cache |

## C. Structural changes not yet attempted

| Idea | Rationale |
|---|---|
| ~~Time-decay likelihood weighting~~ | **DEAD — WP022. WORSE, not null.** Monotone degradation: 810-day half-life +0.00019 [-0.00050,+0.00089], 270-day **+0.00471 [+0.00267,+0.00690]**, 90-day **+0.01395 [+0.01015,+0.01768]**. Normalising preserves likelihood MASS but not INFORMATION — effective N falls to 29% at 270 days and 9% at 90 days, and the RPS ordering is exactly the effective-N ordering. The AR(1) walk already extracts the time structure; Dixon & Coles needed decay because their strengths were STATIC. The 90-day arm also fails to converge (5 divergences, rhat 1.755), the same fact showing up twice. |
| ~~3x CV windows to confirm WP020~~ | **DONE — WP021, 115 windows, 6.6 h. Not confirmed.** The grid (2016 onward, STEP=TEST_LEN=3) is reusable for any future arm needing real power: `wp021_power_confirmation/cv_shared_data_big.pkl` carries deep + shots on 115 windows / 3,780 matches, with a refitted `S3_xgchannel_only` baseline. |
| ~~Combined `xG + deep + shots` arm~~ | **DEAD — WP021.** Worse than deep alone: K vs H = +0.00082 [-0.00005, +0.00170], and decisively worse in thin-history windows (+0.00197 [+0.00027, +0.00363]). The shot channel does not merely fail to add, it DISPLACES what deep contributes. |
| **Multi-league hierarchical hyperparameters** | `get_data.py` already supports six leagues. Fit `sigma`/`rho`/`kappa`/HFA hierarchically across leagues while keeping team latents league-specific → ~5× data for exactly the parameters that are badly pinned. Fixes the *root* cause of the under-powered nulls. Biggest job, best long-run payoff. |
| ~~**Shot volume vs conversion**~~ | **Downgraded by WP016.** The defensive equivalent of this idea (suppression vs shot-stopping) was tested and made the model measurably WORSE (+0.00027, CI excludes zero), despite the persistence asymmetry being real and measured (r=+0.65 vs +0.22). Splitting a latent adds ~14,600 parameters whose noise flows into the combined prediction. The attacking version would face the same problem — don't attempt without a reason to expect different behaviour. |
| ~~A copula / multinomial outcome model with SEPARATE parameters for tilt and draw mass~~ | **LARGELY RETIRED (2026-10-01): the tilt it targets was mostly a COVID artefact.** WP017/WP018 both ran on the 30-window grid, which STARTS at 2020-21 -- the empty-stadium season. Measured by season on the 115-window grid: home tilt is **-0.0351 pre-COVID (2016-18)**, **+0.0267 across 2019-20**, **+0.0042 for 2021-25**, with 2020-21 alone at **+0.0602**. Actual home win rate: 0.477 -> 0.379 -> 0.442. So WP017's "+1.4pp home / -1.5pp away" was substantially one anomalous season, and on the fuller grid the tilt is a third the size and the opposite sign pre-2019. Not worth a new outcome model. |
| ~~A different outcome model (state-dependent arrival)~~ | **Closed by WP018 — see above.** Original reasoning: Given `lambda_home`/`lambda_away`, the three outcome probabilities are *fully determined* by the Poisson convolution; they are not free parameters. WP017 showed re-weighting the objective toward outcomes cannot fix the measured home/away tilt (+1.4pp home, −1.5pp away) — weights spanning 6x moved it the *wrong* way. So the tilt likely lies OFF the surface reachable by any pair of independent-Poisson rates. Dixon-Coles is the existing patch and is structurally incapable of it (one global rho on 4 low-score cells = uniform draw-ward nudge). A fix needs an outcome model with its own asymmetric dependence structure. **Most specific remaining lead in the project.** |
| ~~Deep-completions channel (`deep`)~~ | **DEAD — WP021 (115 windows).** -0.00083 [-0.00177, +0.00013], still null at 2.9x the power, and NOT robust: dropping 2016 alone halves it to -0.00044. The only training-size band clearing zero is the thinnest, and the pattern is non-monotone (-0.00197 / +0.00051 / -0.00101), so there is no clean mechanism. `deep_kappa` learned 4.84 vs the data's 5.05, so the null is not a specification artifact. |
| ~~Defensive context channels (`deep_allowed`, `ppda`, on-target xG)~~ | **All dead.** OOS gains on top of shots-faced + xG-faced: `deep_allowed` +0.0002 [-0.0141,+0.0148], `ppda` -0.0021, on-target xG -0.0016. Note on-target xG is the PROPER measurement of WP016's shot-stopping idea — 43% of the xG in WP016's signal came from shots the keeper never faced, and the clean version persists 2.5x better (r=+0.273 vs +0.111, difference +0.162 [+0.068,+0.249]) — but it still adds nothing predictively. The defensive-channel direction is closed. |
| ~~Dead-ball vs open-play xG~~ | **Dead.** Coefficients genuinely differ (+0.692 [+0.451,+0.891]; open-play xG worth ~5x dead-ball) and dead-ball threat barely persists (r=+0.365 vs +0.815), but dead ball is only 21.8% of xG with a near-zero coefficient, so splitting buys nothing: OOS -0.0051. |
| ~~Big-chance vs small-chance xG~~ | **Dead — redundant.** In-sample it looked like the best idea in the project (R^2 +0.0249, and small-chance xG out-predicts big-chance xG +0.977 vs +0.341). But log small-chance xG correlates **0.932** with log shots: it IS shot volume. Adds +0.0005 [-0.0045,+0.0101] over shots+xG in-sample, -0.0022 OOS. |
| **Opponent-interaction / style effects** | The model is strictly additive in attack and defence; real matchups interact (counter-attacking side vs a high line). Could be a low-rank interaction term. Entirely untested class. |
| **Ensemble with Elo** | `features/elo.py` exists and is unused. Different error structure often combines well with a Bayesian rate model. Cheap. |
| ~~**Within-match state dependence**~~ | **Tested in WP018 — real but cannot help** (ceiling +0.00008). Shot cache now carries `minute`/`X`/`Y`. |

## D. Re-tests — mostly settled by WP015

| Re-test | Status |
|---|---|
| ~~`L2_level` / hard centering~~ | **Done — WP015's `F_level_only` IS this arm. Null** (+0.00000). |
| ~~Away-favourite asymmetry~~ | **Done — falsification test FAILED.** Fixing HFA didn't shrink it (+0.00745 → +0.00738). Still unexplained; HFA is not the cause. |
| **`L3_randomwalk`** | Still open on power grounds only (MDE 35% of the gap). The WP015 rationale for re-testing it — that team walks were absorbing league drift — is weakened now that global drift is known not to matter. |
| **Post-hoc sharpening** | Still nearly free, but the WP015 rationale is gone: there was no systematic HFA bias to unmask. Low priority. |

**Not worth re-testing:** Dixon-Coles (MDE 4% of gap + mechanism understood), npxG,
per-season `kappa`, absence covariate, per-team sigma. Each has a mechanism
independent of global drift.

## F. DONE (2026-10-10) — data durability: `archive/` committed

**Resolved by `scripts/build_archive.py` + commit `98c1045` on branch `archive-artefacts`.**
105.8MB of pickles -> **23.9MB** of Parquet + JSON, 8,219 checkpoint files -> 2, and fidelity
asserted rather than assumed: `build_archive.py verify` rebuilds exactly what
`load_checkpoints()` returns for all 34 work products and compares against the live pickles
with bit-identical checks on lambda / probability / RPS. All 34 match; the archive reproduces
the champion's 0.19657 exactly.

**MY TIERING BELOW WAS WRONG ON ONE POINT, and only measuring caught it.** I had put the big
raw assets in "back up outside git, too large and too binary". Parquet+zstd gets them to
6-10% of pickle size, so they fit comfortably:
`player_minutes` 18.8->1.9MB, `odds_all_leagues` 25.9->1.6MB, `elo_ratings` 11.1->0.8MB,
odds 1x2+totals 7.4->0.5MB. **Measure compression before deciding something cannot be
tracked.**

Also fixed: a stray backtick in `.gitignore` had turned a comment into a literal pattern, and
the per-work-product checkpoint exclusions are now one generalised `work_products/*/checkpoints/`
rule so new work products are covered automatically.

Deliberately still excluded: `cv_shared_data_*` (52MB) and `features*.pkl` — derived, regenerate
in minutes, would triple the archive for no durability gain.

### Original analysis, kept for the reasoning

### (superseded tiering)

**Decide what to put in version control, and stop using pickle as the archive format.**
Measured sizes: checkpoints 20M (7,959 files, ~2KB each -- they hold only `results` +
`cv_match_predictions`, NOT posterior traces), cv_shared_data 52M, other .pkl 150M,
scraped JSON 2.5M, `.git` 1.4M. `.gitignore` currently has a blanket `*.pkl`, so none of
it is tracked.

Tiered by how painful losing each would be:

1. **COMMIT — the scraped JSON (2.5M).** `wp059_injuries/injuries.json` and `squads.json`.
   Text, diffable, and cost 2,700 polite HTTP requests over ~1.5h. Critically, a THIRD
   PARTY controls the source: Transfermarkt can restructure, rate-limit or block at any
   time, and then this is unrecoverable. Not covered by the `*.pkl` rule, so committable
   as-is. Highest value of anything here.
2. **COMMIT AFTER CONVERTING — checkpoints (20M).** They are the output of 150+ hours of
   MCMC and are tiny. But do NOT commit them as pickles: pickle is version-fragile and a
   pandas/numpy upgrade can render them unreadable, which is a terrible property for
   something representing weeks of compute. Dump to Parquet (or CSV -- they are small
   enough) and commit that.
3. **BACK UP OUTSIDE GIT — raw scraped binaries.** `wp044_league_scan/odds_all_leagues.pkl`
   (25M), `wp030_squad_intactness/player_minutes.pkl` (18M),
   `wp025_elo_channel/elo_ratings.csv` (11M). Expensive to re-acquire but too large and too
   binary for git history, where every rewrite stores a full new blob permanently.
4. **DO NOT KEEP — derived pickles.** `cv_shared_data_*` (52M) and `features*.pkl`
   regenerate in minutes from code plus (1)-(3).

Principle: **git is for things that diff, not binaries you cannot prune later; and pickle
is a fine cache but a bad record.** Anything meant to survive a couple of years should be
Parquet / CSV / JSON.

Cost: ~30 min for a conversion script plus a `.gitignore` update. No modelling value, so
it should never pre-empt a sweep -- but the Transfermarkt JSON in (1) is a real
single-point-of-failure until it is committed.

## E. Ruled out — don't revisit without a new reason

| Idea | Why |
|---|---|
| Referee effects | Available in the odds data (9,880 matches, 164 refs) but across refs with 30+ matches, total goals/match varies with SD **0.218** against a Poisson noise floor of **0.310** — observed spread is *smaller* than chance produces. No signal to extract. |
| Dixon-Coles τ | MDE 4% of the gap, plus a mechanism: `rho` is one global scalar, so it applies a near-uniform ~0.7pp shift toward Draw on every match — and the model already slightly over-predicts draws (23.6% vs 23.1% actual). Pushes the wrong way. |
| npxG | `kappa` absorbs the level change (learned 0.957 vs 1.029 — a 7.0% move matching the penalty share), and penalties occur in only 12% of matches, so 88% of observations are unchanged. |
| Per-season `kappa` | Common-mode: shifts all teams together, cancels in H/D/A. Also **collinear with a time-varying intercept**, so don't add both. |
| Post-hoc calibration / isotonic / Platt | WP004 and WP009 both null. Sharpening helps non-upsets and hurts upsets in almost exactly offsetting amounts. |

## NEW LEAD (2026-10-03) — AUDIT THE ASSUMED CONSTANTS

WP032 moved the xG channel weight 0.5 -> 0.75 for **-0.00029, 10/10 LOSO folds**, with no
new data and no new code. 0.5 had never been measured: it was a hedge against
double-counting, tested once by WP007 on a grid whose MDE was 4x the effect, called null,
and then cited as settled by WP013. **The cheapest win in the project's history came from
re-measuring an assumption, not from a new mechanism.**

Still unaudited, same pattern, same reasoning:
* ~~**`deep_channel_weight` = 0.5**~~ -- **DONE, WP033: already optimal, no change.** A
  clean symmetric interior peak at 0.5 (w1.0 is +0.00063, CI excluding zero). The hedge
  was right this time. Worth it anyway: the same sweep's `D000` arm finally ESTABLISHED
  the deep channel (removing it is worse in 0/10 LOSO folds), which WP021 had failed to
  do. Confirmed the channels interact -- `xg_kappa` tracks deep's weight 1.003 -> 1.017 --
  so (xG 0.75, deep 0.5) is the best point on two ORTHOGONAL slices, not a verified joint
  optimum. A 2D sweep is 16+ arms / ~13h and is not worth it.
* **`stakes_effort_sd` = 0.3**, **`home_adv_sd` = 0.02**, **`intactness_sd` = 0.5** --
  prior scales chosen by judgement. But see caution 21: widening a prior only matters if
  the posterior actually MOVES, so check that first and do not repeat WP031's B arm.
* **`xg_shape` prior**, **AR(1) `rho` Beta(29,1)**, **`sigma_att/def` = 0.008**.

## STRUCTURAL LEADS (2026-10-03) — the only untouched structural ideas

Found after WP031 emptied this backlog. Everything tried in WP011-WP032 has been a
COVARIATE or a MEASUREMENT CHANNEL; these two change the LATENTS' OWN STRUCTURE, which
is the reason to expect a different outcome from the ~20 nulls. Neither needs new data.

1. **Match-level common shock. `match_effect_prior` is defined in
   `src/football_model/model/priors.py` and is NEVER CALLED.** Goals are currently
   conditionally independent Poissons given theta. A per-match shared random effect
   scaling both sides' rates captures within-match correlation -- open games where both
   teams score. **This is what WP017 explicitly pointed at** when it concluded the
   home/away tilt came from the bivariate-Poisson structure rather than the objective,
   and nobody built it. The prior is half-written. Best next candidate.
   *Caution:* a free per-match effect on ~4,000 matches is close to one parameter per
   observation; it needs a tight sigma and the predictive must integrate over it, or it
   will fit the training scorelines and predict nothing. Also check it does not simply
   re-introduce the overdispersion that WP028's `J_disp` NegBinomial already measured as
   WORSE -- if it does, that is a well-powered prior null, not an untested idea.

2. **Attack and defence latents are independent AR(1) walks** -- no correlation, no
   shared team-quality factor (verified: no LKJ or cholesky anywhere in `src/`). A
   correlated bivariate walk pools information across the two, which should help most
   where the model is weakest: promoted teams and early season, when each latent has
   little data of its own. Note WP025's Elo ENTRY prior worked (169% of prior SD) for
   exactly that population before being declined on external-data grounds, so there is
   independent evidence that entry-state information is worth something.

## Design cautions learned the hard way

1. **Test correlated features alone.** WP011's crux test looked positive (+0.135, CI excluding zero) purely because two features correlated at 0.76 split into cancelling opposite-signed coefficients. Alone: +0.052, CI includes zero.
2. **Add a source-control arm when swapping data sources.** WP014's npxG number would have been uninterpretable without `D_shotxg`, because Understat's match-level xG and shot feed disagree (corr 0.997 but means 1.425 vs 1.448).
3. **Power is set by windows, not matches.** 30 windows, not 1,314 matches. MDE ranges 4% of the gap (DC) to 52% (xG weight) depending on the arm — check before calling anything dead.
4. **Estimate sweep cost from parameter count, not a previous sweep's wall-clock.** WP016 was predicted at ~50 min and took 702 — splitting defence added ~50% more latents.
5. **When adding a parameter, add it to the harness's `scalar_names` at build time.** Twice now (WP014 per-season `kappa`, WP016 `sigma_stop`) a mechanism was undiagnosable after a long run because the parameter wasn't recorded.
6. **Noise-correct before quoting a range as evidence of a defect.** WP015 overstated the home-advantage flaw 4x by quoting a raw range across season estimates each carrying ±0.062 of sampling noise.
7. **An in-sample R^2 gain from a NESTED model is not evidence.** Adding a regressor
   can never reduce in-sample R^2, so bootstrapping that gain gives a CI whose lower bound
   is >=0 by construction — "excludes zero" is vacuous. Three WP019 premise checks produced
   lower bounds of exactly +0.0000 this way. Use out-of-sample (leave-one-season-out) R^2,
   which can and does go negative. Coefficient-difference CIs are NOT affected and remain valid.
8. **The 240-team-season premise check cannot certify anything.** Out-of-sample CI half-widths
   are +-0.03 R^2 — wider than every effect of interest, including the xG/shots premise itself
   (+0.0244 [-0.0056, +0.0565], null). Use it to RANK candidates and to kill redundant ones,
   then let the 30-window CV harness decide. Don't quote its intervals as verdicts.
9. **Every decomposition of the shot list correlates ~0.93 with shots or with total xG.**
   They are all sums over the same 116k shots, so (count, total) nearly spans the feed. Checked
   and killed: big/small chances (log small xG vs log shots r=0.932), open/dead ball, on-target
   xG (r=0.932 with xG faced). A new channel must NOT be a function of the shot list.
10. **Match the premise check's aggregation level to what the model actually does.** A
   measurement channel infers a team's rate from ONE match. WP019's premise check used
   half-season aggregates (n=240), where shot counts average into a stable quantity and
   volume outranks quality. Per match (n=7,680) xG is the better measurement and shot count
   adds only +0.0066 [-0.0019,+0.0168] on top of it, against +0.0440 [+0.0102,+0.0895] for
   xG over goals — ~15%, predicting ~-0.0004 RPS, which is the null that was measured.
   Run the per-match version FIRST: it is 32x the sample and it is the right question.
11. **Size a run by total test matches and window variance, not window count.** WP021
   predicted MDE +-0.00080 from 115 windows via 1/sqrt(n) and achieved +-0.00095, because
   TEST_LEN dropped 4->3 and the added early seasons are noisier. The scaling assumes
   equally informative windows.
12. **A point estimate that survives more data is not the same as a robust one.** WP020's
   deep effect held 86% of its size at 3x the power, which looked like confirmation — but
   dropping ONE season of ten halved it. Always run leave-one-season-out before believing
   a sub-threshold effect; WP013's bar for the xG channel was 6/6 folds.
13. **The per-match premise check over-promises — treat it as an upper bound.** Three for
   three now (WP019 shots, WP020 deep, WP022 decay) it predicted a gain larger than the
   model delivered, and for decay it got the OPTIMUM'S LOCATION wrong. It optimises a
   static weighted mean; the model has latent walks that already extract time structure
   and precision to lose. Use it to rule candidates OUT via redundancy, not to size gains.
14. **Stop a sweep once the ordering is decided.** WP022 was halted at 88/115 windows; the
   remaining 27 would have bought precision on a settled question. A partial sweep is a
   valid paired comparison as long as every arm covers the same windows.
15. **A promising coefficient on EARLY CV windows is not evidence.** WP029's
   `rest_effort` sat at 17.6% of its prior SD after 12 windows and looked like the
   architectural story working; it decayed monotonically to 2.7% over the last 30 --
   back to where the ADDITIVE version sat. Early windows have the least training
   history, so the latents are weakest and an auxiliary covariate has the most room.
   Three partial reads also called the arm "worse" with CIs excluding zero; all
   resolved to a precise zero at 115 windows. Wait for the full grid.
16. **HOW a covariate enters matters more than WHAT it measures.** Four ADDITIVE
   covariates all fitted at ~2% of a prior SD and died: congestion (WP027), absence
   (WP012), lineups (WP011), keeper. An additive term enters `theta` exactly as the
   intercept and the 29,000 latents do -- a crowded, near-degenerate direction they can
   absorb. Non-additive forms get USED: WP028's multiplicative effort term fitted at 91%
   of its prior SD, its dispersion term at 60%, WP025's Elo latent-prior at 169%. A
   product with the latent is a direction the latents CANNOT reproduce alone, because
   the covariate varies within a team.
   **But it is not a universal fix.** WP029 retested congestion multiplicatively and it
   still failed (rest_effort 17.6% of prior SD, RPS null/worse). Multiplicative helps
   only where the mechanism is genuinely quality-proportional -- measured for stakes:
   the effect rises monotonically with mismatch (r = +0.011 even / +0.040 middle /
   **+0.065 mismatched, CI excluding zero**), and the RPS gain is confined to matches
   where the FAVOURITE has high stakes (-0.00035, CI excluding zero).
   **And the form is not sufficient either.** WP030's intactness was ADDITIVE and still
   fitted at 72-104% of prior SD (sign-consistent 114/115) -- the first additive
   covariate the latents did NOT crush. Because it varies within a team and the latents
   genuinely cannot reproduce it, the ~2% rule is really about whether the covariate is
   a direction the latents can absorb, not about additivity per se. It still made RPS
   WORSE by +0.00090. A well-identified coefficient is not a better forecast.
17. **Don't stack two representations of the same thing.** A time-varying intercept on top of soft centering gives two ways to express level drift; a poorly identified posterior follows.

18. **Never clean the latents of a persistent nuisance you cannot observe at prediction
   time.** WP030 corrected training rows for squad depletion so the attack latent would
   not own results it should not. It cost +0.00090, worse in 10/10 LOSO folds. The
   reason: depletion AUTOCORRELATES (lag-1 r=+0.091, lag-2 +0.147; after a match below
   0.85 intact, 24% are still below 0.9 next match vs a 13% base rate). The
   "contaminated" latent was doing double duty as a forecast of next-match availability.
   Removing it deleted signal and put back zero, because the XI is unobservable at the
   opening price. **Before building any "correct the training rows" arm, measure the
   autocorrelation of the thing being corrected for.** If it persists and you cannot
   observe it at prediction time, expect a loss. This closes the class, not just WP030.

19. **Uniform damage across slices diagnoses a weaker latent; concentrated damage
   diagnoses a bias.** WP030's loss was +0.00122/+0.00090/+0.00079 across depleted /
   mild / full-strength test matches -- essentially flat. That ruled out the targeted
   explanation (over-rating teams that turn out depleted) and pointed at the latent
   itself being systematically less informative. I also wrongly blamed a mis-centring
   bug: centring on 1.0 predicts every match at FULL strength when the training mean is
   0.989, a constant +1.1% rate inflation. Real, but fixing it moved ΔRPS from +0.00090
   to +0.00093. **A correctly identified bug is not automatically the cause.**

20. **Check whether the LATENT WALK already absorbs a nuisance before adding a covariate
   for it.** Twice now an explicit term for something the walk was quietly handling made
   the forecast WORSE, not null: WP030 depletion (+0.00090) and WP031 crowdless football
   (+0.00021 on the very matches it targeted). Both coefficients looked excellent --
   WP030's at 72-104% of prior SD, WP031's at 66% and sign-consistent 91/115, both
   correctly signed. The walk adapts immediately and per team; a pooled coefficient needs
   many observations of the regime before it is identified, and for most test matches
   those observations are not yet in the training set. Swapping an adaptive local
   absorber for a slowly-identified global constant is a downgrade. **A well-identified,
   correctly-signed coefficient is not evidence the arm will help.**

21. **"Posterior far from the prior centre" does NOT mean the prior is distorting the
   fit.** WP031 found home_mu sitting 2.67 prior SDs from its prior centre and treated it
   as an actionable defect. With ~3,000 matches the prior is simply overwhelmed: widening
   it from 0.03 to 0.15 moved the posterior only 0.2100 -> 0.2227 and RPS did not move
   (-0.00005). Before calling a prior binding, widen it and check whether the posterior
   actually MOVES.
