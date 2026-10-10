# WP046 — The last two constants: the audit is closed

**Verdict: both NULL. `clip_theta` is provably inert at zero compute cost;
`season_start_window` = 3 is directionally better than 5 but by −0.00008, below the
inclusion bar. Champion unchanged. The constant-audit category is now exhausted.**

## `clip_theta` — audited for free

No sweep needed. `theta = log(lambda)` spans **−0.969 to +1.372** across all 7,560
predicted rates, so a soft clip at 5.0 never activates (0.000000 of rates even reach
4.0). Provably inert on this data. Saved ~1.5h of compute by checking the arithmetic
instead of running the arm.

## `season_start_window`

    SSW_3  3 rounds   0.19684   -0.00008 [-0.00018,+0.00002]   LOSO 10/10, seasons 9/10
    SSW_8  8 rounds   0.19696   +0.00004 [-0.00009,+0.00016]   LOSO  0/10
    (base) 5 rounds   0.19692

Directionally very consistent — better in every fold and 9 of 10 seasons — but −0.00008
is **1/12th of the MDE** and 2.5% of the remaining gap to Pinnacle's opening line.
P(improves) ≈ 94%, below the ~96–97% bar that admitted the xG weight (WP032) and `deep`.

**Not shipped.** It costs nothing to take (the field is already wired, no runtime or
convergence penalty) and the project has admitted smaller point estimates before. But at
94% on an effect twelve times below the detection threshold, taking it is fitting the CV
grid. A consistency pattern this clean is what you would see from a tiny real effect OR a
tiny systematic artefact, and those are indistinguishable at this magnitude.

Note it was a DEAD config field until WP035 wired it, so this is its first real test.

## The constant audit, closed

| constant | result |
|---|---|
| **`xg_channel_weight` 0.5 → 0.75** | **WIN, −0.00029, 10/10 LOSO (WP032)** |
| `deep_channel_weight` | already optimal at 0.5; the sweep also ESTABLISHED the deep channel (WP033) |
| `sigma_att/def` | confirmed at its optimum — symmetric degradation either side (WP034) |
| `home_adv_sd` | null (WP034) |
| `rho` Beta(29,1) | non-binding — the screen killed it for 15 min instead of 3h (WP034) |
| `season_start_sigma_mult` | null, flat in 2.0–3.0 (WP035) |
| `init_scale` | null, interior valley at 0.2 (WP035) |
| `xg_channel_shape_alpha` | non-binding (WP035 screen) |
| `clip_theta` | **provably inert** (this) |
| `season_start_window` | null, −0.00008, below bar (this) |

**Final tally: 1 win (−0.00029), 2 confirmations, 7 nulls, 1 dead config field found and
fixed.** The one win came from the constant nobody had ever measured — 0.5 was a hedge
against double-counting, tested once by WP007 on a grid whose MDE was 4x the effect, then
cited as settled by WP013. **Re-measuring an assumption beat every new mechanism tried in
this project except the xG channel itself.**

    python work_products/wp046_final_constants/run_cv.py --workers 6
    python work_products/wp046_final_constants/analyse.py
