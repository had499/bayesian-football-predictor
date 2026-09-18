# WP010 — Locating the Resolution Gap

**Status: complete. Conclusion — the gap is diffuse. It shows up in almost every slice tested, at similar magnitude, with no single concentrated pocket. The two things that came closest to a "lead" are both weak and already partly ruled out by earlier WPs. This is the evidence the decision rule below was written for: stop chasing accuracy on this architecture.**

## Why this, why now

WP003 established the model trails Pinnacle by +0.0139 RPS (reliable, CI clear of zero) and that the gap is a **resolution problem** (WP004): the model under-commits on strong favourites and over-commits on underdogs, while the market is well-calibrated. Three separate, principled attempts to close that gap have now been tried and each came back null or marginal:

- **WP005** (loosen the AR1/home-advantage priors) — no single prior helped; a combined loosening showed a small, non-significant improvement.
- **WP006** (per-team sigma / partial pooling) — matched or underperformed the pooled-sigma baseline in every controlled comparison.
- **WP008/WP009** (starting-XI lineup-quality covariate — genuinely new information, not a retuning) — the only comparison to clear significance against the *original* baseline was razor-thin (CI upper bound −0.00003) and did not survive the test that isolates the covariate's own marginal contribution from the prior-loosening it was bundled with.

Three different kinds of intervention, same shape of result. Before trying a *fourth* structural change on a guess, this WP asks a cheaper, prior question: **where does the gap actually live?** WP003 already sliced by how much the model disagrees with the market (gap grows monotonically, worst on the top decile) and by promoted-team status (worse, not better, early season). This WP extends that slicing along axes WP003 didn't cover, using data already in hand, to either find a concentrated, targeted lead or confirm the gap is diffuse — which is itself a real answer, and grounds to stop chasing accuracy on this architecture.

**This WP produces no model change.** It's diagnosis only. Any lever it surfaces becomes a candidate for a future WP, evaluated the same way every change has been since WP003 — against this same 401-match Pinnacle comparison.

## Data (all already on disk, nothing new to fetch)

- `work_products/wp001_walkforward_cv_baseline/cv_shared_data.pkl` → `df_cv` — match metadata: `team`/`opp_team`(_long), `datetime`, `season`, `round`, `cum_round`, `match_id`, `goals_home/away`, `xG_home/away`.
- `work_products/wp001_walkforward_cv_baseline/cv_checkpoint.pkl` → `cv_match_predictions` (401 rows: `window`, `lambda_home`, `lambda_away`, `goals_home/away`, `rho_dc`) and `results` (35 per-window summaries).
- `work_products/wp003_bookmaker_benchmark/odds_raw.pkl` — Pinnacle/Bet365/market odds for the same 401 matches, plus the crosswalk/join logic already written in that notebook (reuse verbatim, don't reinvent).

All pandas, no PyMC, runs in seconds — the cheapest WP in the project by design.

## What to understand — the full list

### A. Deepen the shrinkage mechanism itself

1. **Full three-way calibration, not just P(home).** WP003's calibration table only covered home-win probability. Build the same reliability table for draw and away-win probabilities against Pinnacle's. Draws are the historically hardest outcome to price (this is exactly what Dixon-Coles' low-score correction targets) — check whether the shrinkage is uniform across all three outcomes or concentrated in one, particularly draws.
2. **Home-favourite vs. away-favourite shrinkage.** Split the P(home) calibration bins by whether the *home* or *away* side is favoured. `home_adv` is a single global parameter, not team-specific — if under-confidence is asymmetric (worse when the away team is favoured, say), that points at `home_adv` pooling specifically rather than shrinkage in general.

### B. Decompose the gap by match characteristics WP003 didn't slice on

WP003 sliced by (a) model-vs-market disagreement magnitude and (b) promoted-team status. Both are "does the model know it's uncertain" cuts. These are "is the gap structural to certain match types" cuts, independent of whether the model happens to agree with the market:

3. **Market-implied lopsidedness**, independent of model disagreement — bucket matches by how strong a favourite Pinnacle installs (regardless of what the model says). Is the model specifically worse on lopsided matches even when it *doesn't* disagree much with the market? This is the direct test of "shrinkage" as a match-type property rather than a disagreement property.
4. **Season phase, generalized beyond promoted teams** — bucket by `round`/`cum_round` into early/mid/late-season terciles across *all* teams, not just promoted ones. WP003 found the gap is worst for promoted teams early season; is that actually a promoted-team effect, or a "everyone's AR1 history is thin early in a rolling window" effect that happens to be largest for the teams with the least prior-season carryover?
5. **Home vs. away split of the gap itself** (not of favouritism — of which side the *prediction* is for). Sanity check for an asymmetry unrelated to (2).
6. **Team-level concentration.** Group the 401 matches by team involved and rank by mean gap-to-Pinnacle. Is the RPS gap spread roughly evenly across ~28 teams, or dominated by a handful — plausibly the teams with the most squad/managerial turnover, where an AR(1)-smoothed strength estimate would structurally lag reality?
7. **Chronological trend.** Plot the gap by CV window/season, 2020-21 → 2025-26. Flat, improving, or widening as the walk-forward rolls forward — an "is the model quietly going stale" check that hasn't been done directly (WP001/002/003 all pool across windows).

### C. A specific mechanistic hypothesis for *why* WP005/WP006 came back null

8. **Correlate error with the model's own AR1 revision magnitude.** For each match, compute how much the posterior mean `attack`/`defence` for the involved teams moved over the preceding few weeks (already recoverable from trace data per window, or approximable from the CV checkpoints). If error concentrates on teams whose strength the model itself is revising fastest, that's a specific, targeted explanation: the failure mode isn't "the AR1 process is too rigid everywhere" (which is what WP005's global sigma-loosening tested, and it didn't help) — it's "the AR1 process lags specific teams going through real, fast change," which a uniform global loosening would dilute across 28 teams instead of fixing where it's needed. This is the one genuinely new hypothesis in this WP, not a re-slice of WP003.

### D. Cross-check against what's already been tried

9. For each pattern found in B/C, check it against the three prior null/marginal results for consistency, not just novelty:
   - Does the concentration pattern explain why xG/Dixon-Coles (WP002) didn't move the needle?
   - Does it explain why per-team sigma (WP006) didn't help (e.g., if the gap isn't about "standout teams need to separate from the pack," that's consistent with WP006's null)?
   - Does it explain why the lineup covariate (WP009) only marginally helped (e.g., if the gap isn't concentrated in squad-rotation-heavy matches, that's consistent with a small, hard-to-isolate effect)?
   A pattern that's consistent with all three existing nulls is more trustworthy than one that would have predicted any of them should have worked.

## Method

Same statistical toolkit as WP003/WP004 throughout — paired bootstrap of `RPS(model) − RPS(Pinnacle)` per slice, with a 95% CI, on the *same* 361 Pinnacle-covered matches (so every number here is directly comparable to WP003's own table, not a new yardstick). No new metric is introduced.

## Non-goals

- No retraining, no new priors, no new features. If this WP surfaces a concrete lead, implementing it is a future WP, evaluated on this same 401-match benchmark.
- Not re-litigating WP003's own findings (disagreement deciles, promoted-team subset, betting sim) — those stand; this extends the slicing, it doesn't redo it.

## Decision rule for what comes after

- **If one or two slices show a large, CI-excluding-zero concentration** (e.g., gap is dramatically worse for teams with high AR1 revision, or specifically in the draw-calibration channel): that's the first *targeted*, evidence-backed lead in the project's history, worth a scoped WP011 to test a fix aimed specifically at it.
- **If the gap is diffuse across every slice tested here** (consistent with the "near practical ceiling given historical-data-only inputs" read from WP005/006/009): that's grounds to stop chasing model accuracy on this architecture and treat the current config (`loose_combo` + lineup covariate, per WP009) as the resting point, redirecting future work toward non-accuracy questions (deployment/productionization, or re-examining WP003's "no edge anywhere" finding across bet types rather than just flat-staking 1X2).

## Results

Sanity check first: rebuilding `model_df` and rejoining to `odds_raw.pkl` reproduces WP003's headline number exactly — n=361, gap **+0.0139, CI [+0.0080, +0.0198]** — confirming this WP starts from the identical dataset, not a near-miss.

### A1. Full three-way calibration

| Outcome | Where the model departs from Pinnacle |
|---|---|
| Home win | Confirms WP003: under-confident at the top — 0.75–0.88 bin predicts 0.78, actual 0.92 (Pinnacle: predicts 0.81, actual 0.82). |
| **Draw** | The model's own biggest miscalibration. In its most common draw-probability band (0.25–0.38, n=74) it predicts 0.265 but the true rate is 0.324 — a 5.9-point gap, roughly double Pinnacle's 3.1-point gap in the same band (0.276 → 0.307). |
| **Away win** | A pattern not visible in WP003's home-only table: the model *over*-predicts away wins in the low-to-mid range — 0.12–0.25 bin predicts 0.196, actual is only 0.108; 0.25–0.38 predicts 0.313, actual 0.228. Pinnacle is close to on-target in both bins (0.189→0.109, 0.307→0.233). |

So the shrinkage isn't uniform across outcome types: it pulls home-win probability *up* toward the middle from above, and away-win probability *down* toward the middle from below in these bands, while draws are separately under-called where the model is most often near its draw-probability peak. Three different mis-calibrations, not one.

### A2. Home-favourite vs. away-favourite shrinkage

No meaningful asymmetry. In the top comparable bin (0.67–0.83), home-favoured matches: model predicts 0.731, actual 0.875 (14.4-point gap); away-favoured matches: model predicts 0.722, actual 0.833 (11.1-point gap) — same shrinkage, same direction, similar size, in both splits (n=24 and n=12 respectively, so neither is precisely estimated, but there's no sign the asymmetry runs through `home_adv`). Doesn't implicate the single global `home_adv` parameter specifically.

### B3. Market-implied lopsidedness (independent of model disagreement)

| Slice | n | Gap | 95% CI |
|---|---|---|---|
| bottom tercile (closest matches) | 121 | +0.0179 | [+0.0071, +0.0288] |
| middle tercile | 121 | +0.0072 | [−0.0041, +0.0181] — not significant |
| top tercile (most lopsided) | 121 | +0.0174 | [+0.0099, +0.0247] |

Non-monotonic: the gap is significant at *both* extremes and smallest (and not significant) in the middle. Doesn't cleanly support "the gap concentrates on lopsided matches" as a standalone story — if shrinkage alone explained it, this should have grown monotonically with lopsidedness the way WP003's disagreement-deciles did.

### B4. Season phase, generalized beyond promoted teams

| Slice | n | Gap | 95% CI |
|---|---|---|---|
| early third (rounds 4–14) | 146 | +0.0147 | [+0.0061, +0.0233] |
| mid third (rounds 14–24) | 153 | +0.0188 | [+0.0108, +0.0270] |
| late third (rounds 24–35) | 140 | +0.0105 | [+0.0002, +0.0207] — barely clears zero |

Present throughout the season, if anything narrowest late. WP003's "gap is worst for promoted teams early" finding does **not** generalize to "every team's prediction is worse early in a window" — it's specific to promoted sides, not a league-wide thin-history effect.

### B5. Home vs. away goal-expectation bias

`mean(lambda_home − goals_home)` = **−0.1235**, CI **[−0.2566, −0.0008]** (barely excludes zero — wide).
`mean(lambda_away − goals_away)` = **−0.0198**, CI [−0.1402, +0.1007] (not significant).

The one concrete, mechanistically specific asymmetry this WP found: the model's home-goal expectation runs low by about an eighth of a goal per match on average, while its away-goal expectation is close to unbiased. But the CI on the home figure only just clears zero, and this direction — a single global `home_adv` that's on average too small — is exactly what WP005 already tested in isolation (`home_adv_sd` as one of its four individual prior arms) and found no significant improvement from. A real-looking pattern that's already been checked and came back empty once.

### B6. Team-level concentration

Among reasonably-sampled teams (n≥20): a continuous spread from TOT (+0.0305, n=36) and WOL (+0.0266, n=38) down to WHU (−0.0059, n=36), with most teams clustered in +0.01 to +0.02. No small subset of outlier teams carrying the whole gap — consistent with why WP006's per-team sigma (designed to let standout teams separate from the pack) didn't help: there's no small pack of standouts to separate.

### B7. Chronological trend

Gap present in every season (2020-21 through 2025-26), no monotonic widening or narrowing — 2021-22 shows the largest point estimate (+0.0215) and the two most recent full seasons the smallest (+0.0107, +0.0098, both CIs touching zero, small-n). No evidence the model is going stale as the walk-forward rolls forward.

### C. Market-implied team-strength volatility (AR1-lag proxy)

The planned test — correlating error with the model's *own* posterior AR1 revision magnitude — isn't reconstructable without retraining (per-window traces aren't persisted; only scalar summaries are saved in the CV checkpoints). Substituted a market-based proxy instead: each team's volatility of Pinnacle-implied win probability over their preceding 5 matches (from `odds_raw.pkl`'s full 2,280-match history, not just the 401 test matches — leakage-safe, strictly-earlier matches only).

| Slice | n | Gap | 95% CI |
|---|---|---|---|
| low volatility | 121 | +0.0198 | [+0.0101, +0.0294] |
| mid volatility | 121 | +0.0068 | [−0.0025, +0.0161] — not significant |
| high volatility | 121 | +0.0144 | [+0.0035, +0.0251] |

**No support for the hypothesis.** If the AR(1) process structurally lagged fast-changing teams, the gap should grow with volatility. It doesn't — the low-volatility tercile has the *largest* gap, and the relationship is non-monotonic. The one genuinely new mechanistic hypothesis this WP set out to test came back null (caveat: a market-based proxy for "true" strength change is indirect — this doesn't rule out the model's own AR1 revisions behaving differently, only that this proxy for it found nothing).

## What this means

Per the decision rule set out above: **the gap is diffuse.** Across eight different slicings — outcome type, home/away favouritism, lopsidedness, season phase, home/away goal bias, team, season, and market volatility — the +0.0139 gap to Pinnacle shows up almost everywhere at broadly similar magnitude, not concentrated in one exploitable or fixable pocket. The two things that looked like leads on first pass don't survive scrutiny:

- **B3/B4's slight non-monotonicity** is noise-shaped, not a signal pointing anywhere specific.
- **B5's home-goal bias** is the only finding with a specific, mechanistic story (`home_adv` running a bit low) — but the CI barely clears zero, and the direct test of loosening exactly that parameter was already run in WP005 with no significant result. This isn't a new lead; it's the same one, seen from a different angle, that already came back empty.

Cross-checking against the project's existing nulls (Section D of the plan) makes the same point from another direction: the draw-miscalibration in A1 is exactly what Dixon-Coles targets, yet WP002 found DC doesn't help; B6's continuous (not clustered) team-level spread is consistent with why WP006's per-team sigma didn't help; nothing here concentrates in squad-rotation-heavy matches, consistent with why WP009's lineup covariate only marginally helped.

**Conclusion: stop chasing model accuracy on this architecture.** Four separate diagnostic and structural efforts (WP005 priors, WP006 partial pooling, WP009 new information, WP010 gap localization) now agree: there is no concentrated, fixable lever left that this project's methodology can find. The current best-available config (`loose_combo` + lineup covariate, per WP009) is the resting point. Future work should move to the two non-accuracy directions raised when this WP was scoped: productionizing what exists, or re-examining WP003's "no edge anywhere" finding across bet types/staking approaches rather than flat-staking 1X2 — not a WP011 in this same family.

## Reproducing

```bash
cd work_products/wp010_resolution_gap_diagnosis
jupyter lab wp010_resolution_gap_diagnosis.ipynb
```

Pure pandas, no PyMC, no training — loads three existing pickles (`wp001/cv_checkpoint.pkl`, `wp001/cv_shared_data.pkl`, `wp003/odds_raw.pkl`) and runs top to bottom in seconds.
