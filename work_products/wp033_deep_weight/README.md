# WP033 — Tuning the deep channel's weight

**Verdict: NO CHAMPION CHANGE. 0.5 is already the optimum. But the sweep delivered
something more useful — it finally established that the deep channel earns its place.**

| weight | RPS | ΔRPS vs 0.5 | LOSO better | seasons |
|---|---|---|---|---|
| **0.00** (off) | 0.19762 | **+0.00070** [−0.00017,+0.00158] | 0/10 | 2/10 |
| 0.25 | 0.19699 | +0.00007 [−0.00029,+0.00043] | 2/10 | 4/10 |
| **0.50** | **0.19692** | — (champion, best) | — | — |
| 0.75 | 0.19712 | +0.00020 [−0.00008,+0.00048] | 0/10 | 3/10 |
| 1.00 | 0.19755 | **+0.00063** [+0.00009,+0.00116] worse | 0/10 | 2/10 |

115 windows, 3,780 matches, 460 fits, 0 divergences, 195 min.

## Why this was worth running even though nothing changed

WP032 had just shown that `xg_channel_weight = 0.5` was an unexamined hedge that cost
−0.00029. `deep_channel_weight = 0.5` was the **identical hedge** on the identical kind
of channel, chosen for the same reason (goals and deep completions share
possession-level information, so full trust double-counts) and never measured at any
value — WP020 selected the channel and WP021 re-tested it, both at 0.5 only.

It had to be checked. It came back correct.

## The finding that matters: deep earns its place

Deep was the weakest component of the champion. It went in on **expected value, not
proof**: −0.00083 [−0.00177, +0.00013], better in only 6/10 seasons, the effect halved
when 2016 was dropped, and **WP021 could not confirm it at 115 windows.**

`D000` tests it directly against the *current* champion: turning deep off costs
**+0.00070, worse in 0/10 leave-one-season-out folds.** The headline CI still spans zero
(+0.00070 [−0.00017, +0.00158]), so this is not "CI excludes zero" — but a unanimous
10/10 fold result is the robustness criterion this project actually uses, and deep has
never cleared it before. **This is stronger evidence for the deep channel than WP020 or
WP021 produced.**

Why it is stronger now than in WP021: with xG upweighted to 0.75, the two channels were
expected to compete and deep was the plausible casualty. It survived that test instead.

## The channels genuinely interact

`xg_kappa` — the xG channel's own calibration constant, which should sit near 1.0 —
tracks deep's weight:

    deep w0.00  xg_kappa 1.003    w0.25  1.005    w0.75  1.013    w1.00  1.017

So the two channels are not independent evidence; they trade off against the same
latents. That is what made joint tuning a legitimate worry, even though it resolved to
"no change". It also explains why deep's curve is symmetric around 0.5 while xG's peak
sat above its hedge: xG is the stronger signal and can absorb more weight before
distorting, deep cannot.

## What is NOT established

This is a 1D slice through a 2D (xG, deep) surface at xG = 0.75. Both 1D optima are now
measured — xG at 0.75, deep at 0.5 — but a joint optimum could sit elsewhere, and a
proper 2D sweep is 16+ arms and ~13 hours. Not justified for effects this size. **Do not
call (0.75, 0.5) "the optimum"; call it the best point on two orthogonal slices.**

## Predictions on record vs outcome

* **Pre-smoke:** "deep's optimum has probably moved DOWN, since xG now takes more of the
  available constraint." **Wrong** — it is exactly 0.5, unmoved.
* **Pre-smoke:** "`D000` lands null, which simplifies the champion." **Wrong, and in the
  more useful direction** — deep is paying for itself.
* **Post-smoke revision:** "optimum at or above 0.5." **Right.**
* **Post-smoke:** "`D000` slightly positive." **Right** (+0.00070).
* The one-window smoke again misled on fine detail: it put 0.75 marginally AHEAD of 0.5,
  where the full sweep puts it +0.00020 behind. The smoke is reliable for killing an arm
  that is catastrophically wrong (WP032's w1.75 at 13x MDE) and unreliable for ranking
  neighbouring points. Same lesson as WP032's partial-sweep misread.

    python work_products/wp033_deep_weight/run_cv.py --workers 6
    python work_products/wp033_deep_weight/analyse.py
