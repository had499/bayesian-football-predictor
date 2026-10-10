# WP049 — Championship-derived entry strength: NULL, and it resolves the Elo mechanism

**Verdict: NULL. −0.00004 [−0.00012, +0.00003]. The model DECLINES the covariate
(1–6% of prior SD, sign-consistent 60/115 and 67/115 — coin flips). Elo's benefit is NOT
Championship entry information.**

## The idea

WP025's Elo anchor is worth −0.00021 overall and −0.00049 on promoted-team matches (CI
excluding zero) vs −0.00010 on established teams. The only reason clubelo knows anything
about a newly promoted club is that it watched a Championship season — so compute that
ourselves and drop the external dependency. That dependency is now permanent: clubelo's
API is **deactivated at source** (`/Fixtures` returns 200 with "Fixtures API
deactivated"; `/Arsenal` and `/2024-05-01` are 502 across all user-agents), the main site
is a JS shell with no historical ratings, and 538's SPI — the per-match alternative — is
gone with the company. The cached Elo CSV can never be refreshed.

## What was built

* **Data:** openfootball England, Championship + League One, every season 2013-14 to
  2024-25, 13,113 matches. Two different file formats across seasons (the 2024-25 shape
  is `home v away SCORE` with year-bearing date lines) — parsing only the older shape
  silently yielded ZERO matches for that season.
* **Championship model:** goals-only (no xG exists below the top five leagues) —
  ridge-regularised, opponent-adjusted attack/defence with a fitted home advantage,
  closed form per season. Justified because clubelo has no xG either, our form is richer
  than its single scalar, and the fitted entry coefficient ATTENUATES a noisy input
  rather than being biased by it.
* **Season-long, not recency-weighted** — the premise falsified the "promoted teams
  improve late" story: half-lives of 60d and 30d scored +0.0277/+0.0268 against flat's
  +0.0419.
* **Centred across promotions** — raw z is positive for every promotion by construction
  (they are the best Championship sides), so uncentred it would fit a near-constant
  promoted-level shift, and WP007 measured a global promoted offset as WORSE (+0.00285).
  The premise correlation is location-invariant, so the evidence is for the SPREAD only.
* **All 33 promotions matched**, every name mapping verified by eye, entry-z sd 0.509.
* Fed into the existing `entry_scale` slot, so the fitted coefficient absorbs the
  league-level offset and no cup cross-links are needed.

## Result

    champion  0.19692
    CH_ENTRY  0.19688    -0.00004 [-0.00012,+0.00003]

    elo_entry_att  median +0.0014 =  1% of prior SD, sign-consistent  60/115
    elo_entry_def  median -0.0147 =  6% of prior SD, sign-consistent  67/115

    has a PROMOTED team      1074   -0.00017 [-0.00037,+0.00004]
    established only         2706   +0.00001 [-0.00004,+0.00006]
    promoted AND rounds 1-5   138   -0.00044 [-0.00134,+0.00039]

The promoted slice points the right way at a third of Elo's effect, and the established
slice is correctly zero — no spurious effect, just one too weak to clear noise.

## What it settles: Elo's mechanism is TRACKING SPEED, not entry information

The 169%-of-prior-SD coefficient that made Elo look compelling belonged to the **anchor**,
which acts as a reversion target in EVERY round. Its `elo_entry` sibling was abandoned at
24/115 windows and never measured. This work package supplies the missing comparison: a
Championship rating built from the same results clubelo uses carries essentially nothing
at entry. So the anchor's value is that Elo **updates after every match, including cups**,
while our walk crawls at rho ≈ 0.993 — it tracks teams our latents have not yet pinned
down, which is why the gain concentrated on promoted sides and rounds 1–5.

**Implication:** the Championship route cannot replace Elo. The live idea is a
**promoted-team innovation variance** — letting a team with no history move faster until
its latent is identified. That needs NO external data. Note `season_start_sigma_mult`
inflates variance for ALL teams at season start and WP035 found it flat across 2.0–3.0;
nobody has tried inflating it only for teams the model has never seen.

## Premise vs outcome

The premise was +0.0419 [−0.0211, +0.1012] — the right sign and Elo's magnitude (+0.047)
but CI spanning zero, with 33 club-seasons as the power ceiling. I ran it anyway arguing
the premise test was the weaker instrument, since Elo's premise was the same size and its
CV arm still cleared zero. **That argument was wrong here**: the CV arm agreed with the
premise. The smoke fit warned correctly (coefficients small and wrong-signed) and I put
it below even odds before the sweep finished.

    python work_products/wp049_championship/fetch.py
    python work_products/wp049_championship/premise.py    # ppg / goal-diff: null
    python work_products/wp049_championship/premise2.py   # opponent-adjusted: +0.042, CI spans 0
    python work_products/wp049_championship/prepare_data.py
    python work_products/wp049_championship/run_cv.py --workers 6
