# WP045 — Ligue 1: the venue thesis, tested and dead

**Verdict: the "find a weaker market" strategy is DEAD. The softest closing line of the
five Understat leagues produced the WORST model-vs-market gap, and the model adds
literally nothing to Ligue 1's opening price (blend weight w = 0.000 in all 9 seasons).**

## The test

WP043 (clean) and the EPL gave two points where the market beat us by a near-identical
margin despite very different market quality, suggesting the gap might be structural.
WP044's odds-only scan found Ligue 1 has the softest closing line of the five
(+0.02920 vs the EPL's +0.03877), making it the decisive third point. Unlike RFPL it also
has Pinnacle OPENING prices.

## Result

    league      market skill   our skill   market ahead by
    EPL             +0.03877    +0.03576       +0.00301
    RFPL            +0.02954    +0.02669       +0.00285
    Ligue_1         +0.03058    +0.02514       +0.00543   <- softest line, WORST gap

    Ligue 1 vs OPEN   gap +0.00472 [+0.00340,+0.00602]
    Ligue 1 vs CLOSE  gap +0.00543 [+0.00400,+0.00686]
    blend into the OPEN: w = 0.000 in 9/9 seasons, logloss +0.00000

**The softest market gave the worst result.** Not a constant gap — an actively worse one.

## Why: our model degrades faster than the market does

Leaving the EPL, our skill falls 0.0358 → 0.0251 while the market's only falls
0.0388 → 0.0306. And our one working mechanism weakens in lockstep:

    xG channel:  EPL -0.00258  |  RFPL -0.00177  |  Ligue_1 -0.00130

The EPL is where we have the most matches, the longest xG history and the strongest
channel effect. It is also where the market is sharpest — and the gap is still smallest
there.

**So the thesis was backwards.** A market is soft because the football is harder to
predict, and that penalises a purely statistical model more than it penalises a market
that also prices team news and money flow. Low market skill is not an exploitable
inefficiency; it is a signal that the league is noisy for everyone, and we are hurt more.

## Corrections made along the way

* **I claimed the open→close spread was worth 0.0107** and that RFPL's deficit might
  vanish at the open. WP044 measured it properly (`PSH` vs `PSCH`): it is **~0.001** in
  every league. My figure came from WP010's "pre-close" mirror column, whose timing was
  never verified. Confirmed here: Ligue 1's open→close gap is 0.00071.
* Three data faults, all caught by assertions rather than by reading output:
  **TLS** — the proxy intercepts intermittently and `certifi` has no corporate root;
  fixed with a macOS-keychain + certifi bundle (356 certs) at `.certs/bundle.pem`.
  **Unplayed fixtures** — Ligue 1 2019-20 was abandoned and Understat still lists 103
  cancelled matches stamped 2020-07-09 with `xG` exactly 0.0; they would have trained as
  real 0-0 draws.
  **Short-code collisions** — `NAN` is BOTH Nancy and Nantes, `AJA` BOTH Ajaccio and GFC
  Ajaccio (32 codes, 34 clubs). The codes would have merged two clubs into one latent
  team; the tell was a `NAN vs NAN` fixture. Now keyed on full names. RFPL is unaffected
  (32 codes, 32 clubs) but any further league needs this check.
* Odds are read from WP044's cache: football-data returned 200 for every F1 season during
  the scan and HTTPError for all of them an hour later.

    python work_products/wp045_ligue1/prepare_data.py
    python work_products/wp045_ligue1/run_cv.py --workers 6
    python work_products/wp045_ligue1/analyse.py
