# WP030 — Squad intactness as a training-row correction

**Verdict: WORSE, decisively. Both arms. The direction is closed.**

| arm | RPS | ΔRPS vs champion | LOSO | seasons |
|---|---|---|---|---|
| `I_effort` (champion) | 0.19721 | — | — | — |
| `M_intact` (centred on 1.0) | 0.19811 | **+0.00090 [+0.00053,+0.00128]** | worse 10/10 | worse 9/10 |
| `N_intact_centred` (centred on training mean) | 0.19813 | **+0.00093 [+0.00055,+0.00131]** | worse 10/10 | worse 9/10 |

For scale: the xG channel is −0.00258 and the whole remaining gap to Pinnacle's
**opening** line is +0.00344. This handed back a quarter of the gap. 115 windows,
3,780 matches, 0 divergences.

## The idea, and why it was worth trying

Every absence test so far (WP011 lineups, WP012 absence covariate, WP016 keepers)
asked: *does knowing who is missing help PREDICT this match?* All null. WP030 asked the
opposite: *does knowing who was missing help the model LEARN correctly from the past?*
When a depleted side loses, the attack latent absorbs the whole result as evidence of
poor quality. It should not.

That framing also satisfies the hard constraint that killed the lineup direction: the
term is `b·(intact − c)`, which is zero at the centring point, so **prediction consumes
no team-sheet information about the priced match** and the model stays usable against
opening prices. `predict.py` deliberately does not implement the term at all; a test
guards that.

**Intactness** = minutes-weighted player importance available, divided by the team's own
recent norm. Importance is an EWMA of each player's minutes share, half-life 15 matches
— chosen on the *downstream* objective, because the obvious criterion ("predict this
match's minutes") picks half-life 3, which retains 31% after five missed matches, i.e.
it defines an injured player as unimportant. Non-appearances expand to zero minutes.
Self-normalising against the team's own norm is essential: summing whole-squad
importance measures squad *depth* (≈11 for everyone by construction), which is exactly
WP011's `absent_share` error. Result: mean 0.989, p5 0.82, 15.4% below 0.9.

**The premise was the strongest in the project** — residual correlation r = **+0.1469**
[+0.1093, +0.1875], concentrating where depletion occurs (+0.201 vs +0.128). For
comparison: Elo +0.047, stakes +0.040, congestion −0.020 (null).

## Why it failed — and it is NOT the usual null

The coefficients fitted at **72% and 104% of prior SD, sign-consistent 114/115 windows**.
The four previous additive covariates all collapsed to ~2% of prior SD. So the model
genuinely wants this term. **This is a real, large, stable effect that makes forecasting
worse** — a different and more instructive failure than a null.

**Depletion PERSISTS, so the latents' "contamination" was itself a forecast.**

    corr(intact_t, intact_t+1) = +0.091      t+2 = +0.147      t+3 = +0.114
    after a match below 0.85: 24% still below 0.9 next match
    after a match above 0.95: 13%            (base rate 15%)

A team that was depleted tends to still be depleted. So the champion's latents, by
quietly absorbing past depletion, carry a weak but genuine prediction of *next* match
availability. Explaining that away leaves the latents cleaner but **strictly less
informative about the match being priced**, because at prediction we cannot observe the
XI and have nothing to put back. We removed signal and replaced it with zero.

The premise correlation was real and still misled us: it measured that intactness
explains *contemporaneous* residuals, which says nothing about whether removing it from
the latents helps a forecast that cannot see it.

## What I got wrong on the way

- **I predicted the damage would concentrate on depleted test matches.** It did not. The
  loss is near-uniform: +0.00122 / +0.00090 / +0.00079 across depleted / mild / full
  test matches, and +0.00104 / +0.00078 / −0.00070 by depleted run-up. Uniform damage is
  the signature of a systematically less-informative latent, not of a targeted bias.
- **I diagnosed a mis-centring bug as a material cause and it was not.** Centring on 1.0
  predicts every match at *full* strength while the training mean is 0.989 — a constant
  +1.1% inflation of both rates. Real, but worth nothing: fixing it moved ΔRPS from
  +0.00090 to +0.00093. The `intactness_center_mean` flag is kept (it is the correct
  specification) but it rescues nothing.
- **`intact_att` / `intact_def` are misnamed.** They are PER-SIDE coefficients: `b_ia`
  multiplies home intactness (+θ_home, −θ_away), `b_id` away intactness. Each constrains
  a team's attack and defence response to be *equal*; the fitted 0.37 vs 0.52 is a
  home/away asymmetry, not attack/defence. A per-role split is a different, untested
  model. Documented in `model.py` rather than silently renamed.

## Checks that came back clean

- **Collinearity with `stakes_effort`** — a real worry, since the deepest depletion is
  end-of-season rotation (the most depleted rows are all mid-May dead rubbers and Man
  Utd resting before a Europa final), exactly when `stakes_effort` fires. Clean:
  `stakes_effort` *grew* 15% (−0.272 → −0.322) and shrank in only 10/115 windows.
  Intactness is not stealing its variance. (WP011's false positive was this trap.)
- **Paired baseline** — data file verified bit-identical to WP028's apart from the
  intact/rest/load columns, and rest/load are inert with `use_congestion=False`.

## Known construction weakness

The high tail is a transfer-window artefact: the top values are all Chelsea in Aug–Sep
2023, after their huge squad turnover, whose "own norm" was computed over a squad that
no longer existed, so new signings register as *above* full strength. Small share,
two-sided noise rather than a leak, but it is caused by the self-normalisation that
fixes the WP011 error.

## The transferable lesson

**Do not clean the latents of a persistent nuisance you cannot observe at prediction
time.** If the nuisance autocorrelates, the "contaminated" latent is doing double duty
as a forecast of the nuisance, and removing it is a strict loss. This closes the whole
class of "correct the training rows" ideas, not just this one — and it closes the cheap
version of the player-level architecture. The only way to win from availability is to
know the XI for the match being priced, which is ruled out by the opening-price
constraint.

    python work_products/wp030_squad_intactness/run_cv.py --workers 6
    python work_products/wp030_squad_intactness/analyse.py
