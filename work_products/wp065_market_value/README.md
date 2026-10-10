# WP065 — Transfermarkt squad market values: NULL (but the leak audit came out clean)

7,139 player-season values scraped across 10 seasons (league total rising EUR 5.9bn -> 13.7bn,
0.3% of panel players without a value). Row-scoped parsing, validated before the scrape:
Liverpool 2019 totals EUR 1,003m with Salah/Mane EUR 120m, Alexander-Arnold EUR 99m, van Dijk
EUR 80m, correct ordering.

## Why this was the chosen direction

Market value is the only quality signal NOT derived from this league's match results, so it is
the one thing the latents cannot already have absorbed -- which is why ~22 prior/structure arms
and 228 engineered features all came back null. It also targets a specific, known blind spot:
WP059's working injury feature weights absent players by SHARE OF RECENT MINUTES, so a
GBP100m signing injured in his first month is weighted ~0.

## Result: NULL

Coefficients chosen on windows <=200, evaluated on 201-343 (n=1,305).

| arm | params | HELD-OUT vs CLOSE | vs OUTCOME |
|---|---|---|---|
| **`out_min` (incumbent)** | 0.20 | **−0.000113 [−0.000170,−0.000059]** | −0.00025 |
| `out_val` (value-weighted) | 0.25 | −0.000115 [−0.000195,−0.000035] | −0.00029 |
| `out_min` + `out_val` | 0.05, 0.20 | −0.000125 [−0.000201,−0.000050] | −0.00030 |
| `relval` (squad value) | 0.10 | −0.000112 [−0.000257,**+0.000025**] | −0.00021 |

**Value weighting does not beat minutes weighting** (−0.000115 vs −0.000113), despite
`corr(out_min, out_val) = 0.776` confirming they carry genuinely different information. The
new-signing blind spot is real in principle and worth nothing in practice. Combining them
gains 0.000012, deep inside the noise. **Squad value alone is null.** No change to the champion.

## The leak audit — clean, and this is the methodologically useful part

Transfermarkt historical values could be START-of-season snapshots (legal) or REVISED
end-of-season values (look-ahead: a player who broke out mid-season would carry a value partly
determined by the results being predicted). Audited by splitting the gain across the season:

| | `relval` gain |
|---|---|
| EARLY (rounds 1-12) | **−0.000222 [−0.000403,−0.000059]** |
| MID (13-26) | +0.000075 |
| LATE (27+) | −0.000265 [−0.000735,+0.000166] |

**This is the LEGAL signature.** End-of-season revision would concentrate the gain LATE;
instead it sits EARLY, exactly where a genuine start-of-season snapshot should help -- before
the latents have results to work with. Same audit shape as WP059c's borderline/unambiguous
split, which is what rescued the injury result from a false leak alarm.

## A hypothesis, explicitly NOT a finding

The early-season cell (−0.000222, CI excluding zero, n=511) suggests squad value may carry
information in the first dozen rounds before the latents converge. It is **one of three
subgroups examined AFTER seeing the eval data**, the headline is null, and many close-metric
tests have now been run. Testing it properly means pre-registering an early-season-only arm
with k selected on early-season windows alone.

It is also in mild tension with WP049, which concluded Elo's advantage was in-season TRACKING
SPEED rather than entry information. Unresolved.
