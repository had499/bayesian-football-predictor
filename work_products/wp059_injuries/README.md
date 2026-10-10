# WP059 — published injury data WORKS as a post-hoc multiplier

**First new external data source to pay since the xG and deep channels**, and it breaks a
0-for-5 record on squad/absence information.

## Data

Transfermarkt dated injury spells. 200 squad pages (20 clubs x 10 seasons) -> **2,989
distinct players**, then one injury page each -> **16,808 spells** for 2,308 players.

Identity matching was the whole risk (WP043: fuzzy club matching collapsed three clubs into
one and silently dropped 25% of rows while reporting no mismatches). Progress of the fix,
measured as **share of EPL minutes unaccounted for**:

| stage | minutes missed |
|---|---|
| anchor-text squad regex | 15.4% |
| href-based extraction | 5.1% |
| transliteration + token matching | 1.2% |
| 13 hand-verified overrides | **0.20%** |

The anchor-text regex was the dangerous one: it found 50 players for Liverpool 2019 and
silently omitted **van Dijk, Alexander-Arnold, Alisson and Shaqiri**. It was caught only by
checking misses by MINUTES rather than headcount — 15% of headcount being 15.4% of minutes
proved the misses were not fringe players. Three candidate matches were DELIBERATELY left
unmatched (`Louis Beyer`->`jordan-beyer` etc.) because they are probably different people.

677 injury pages "failed to parse". Investigated rather than assumed: 12/12 of a random
sample explicitly say `<span class="empty">No entries available</span>`, and the no-spell
players hold **1.7% of all EPL minutes** at a median of 27 minutes each. Genuine fringe
players with no injury record, not a defect.

## Result — held out, coefficient locked on a disjoint half

`lambda_home' = lambda_home * (1 - k*out_home) * (1 + k*out_away)` and symmetrically; a
team's own absences depress its scoring AND lift its opponent's. **Nothing is fitted** —
which is the point: four arms have died because ~29,000 latents crush an estimated
coefficient during fitting (additive terms land at ~2% of prior SD; WP054's in-model GBM
covariate fitted at 5% and was null while the same signal as a post-hoc wrapper gave
-0.00062). A hand-set k has no coefficient to crush.

k chosen on windows <=200 (n=2,107), evaluated on 201-343 (n=1,305):

| signal | k* | HELD-OUT vs CLOSE | HELD-OUT vs OUTCOME |
|---|---|---|---|
| `out_w_new` (<=14 days) | 0.15 | −0.000082 [−0.000117,−0.000047] | −0.00016 [−0.00042,+0.00011] |
| **`out_tot`** (all absences) | 0.20 | **−0.000113 [−0.000169,−0.000059]** | −0.00024 [−0.00063,+0.00015] |

For scale, the GBM wrapper is −0.00010 on the close — so availability is a **comparable**
gain. The held-out figure came out LARGER than the in-sample-optimised −0.000057, so there
is no selection inflation.

## MY MECHANISM STORY WAS WRONG, three ways

I predicted only UNPRICED CHANGE could help, because the latents have already tracked a
long-term absence. Built `out_w_old` in as the falsification arm. All three predictions failed:

1. **`ret_w_new` falsified outright** — returning players are worse at every k. A returning
   key player is NOT unpriced good news.
2. **`out_w_old` is NOT inert** — it helps, with an optimal k of ~0.25 against new absence's
   ~0.20. Per unit of unavailable minutes the two carry the SAME weight.
3. **So duration does not matter at all**, and `out_tot` (new + old) is strictly the better
   feature — which is how following the falsification produced the stronger result.

The signal is simply "how much of the side that recently played is unavailable", linearly.
Not a story about what the latents have or have not absorbed.

## Why this broke the 0-for-5 squad/absence record

Five previous attempts failed (WP011 instability, WP012 absence covariate, WP030 intactness,
WP047 reds, WP056 lineup features). Two differences: it is a **published, dated, severity-
bearing** absence list rather than a proxy we derived ourselves, and it is applied
**post hoc** rather than as an in-model covariate — the documented failure mode.

## Caveats

- **Outcome-RPS CI includes zero.** Established on the close metric, which WP057 validated as
  4-5x more powerful. The same was true of the GBM wrapper before its close screen. Project
  standard already rejects CI-excludes-zero as the inclusion bar, but this should be stated.
- **External dependency on Transfermarkt** — the same objection that kept the Elo anchor out
  of the champion. An operational decision, not a modelling one.
- 95.1% of matches have availability on both sides; the rest fall back to no adjustment.
- Legality is arithmetic: a spell counts only if it STARTED >=2 days before kickoff, so it
  was public before the opening line posted. No confirmed-lineup information is used.

---

# WP059c — LOOK-AHEAD AUDIT of the `until` dates (prompted by the Elo-feed point)

The Elo anchor was declined because the FEED was unreliable (15-day snapshots), not because
it was external. The equivalent reliability question here is whether Transfermarkt back-fills
a spell's `until` from the player's comeback appearance. If it did, "this spell covers match
D" would encode "he did not play in D" — confirmed-lineup information, excluded by the
open-price rule. My legality filter constrained only the spell START, not the end.

## Diagnostic 1: do end dates coincide with the return match?

Of 9,396 spells with a later appearance, `until` falls on the EXACT date of the next
appearance **2.4%** of the time (15.5% within a day). A systematic back-fill would spike hard
at 0. Reassuring but not clean on its own.

## Diagnostic 2: a build that NEVER reads the end date — and it FAILED

`build_strict.py` counts a player out iff a spell started >=2 days before D AND he has made no
appearance in any PRIOR match since that spell began (past team sheets only). Held out:

| feature | k* | vs CLOSE |
|---|---|---|
| `out_tot` (uses `until`) | 0.20 | −0.000113 [−0.000169,−0.000059] |
| `out_strict` (no `until`) | 0.15 | −0.000026 [−0.000061,+0.000007] **INCL 0** |

**77% of the signal vanished.** Taken alone this looks like the original was look-ahead.

## Diagnostic 3: the decomposition that settles it

If the gain were look-ahead it MUST live where "was he back in time for D?" was genuinely
uncertain — i.e. where the spell ENDS near D. Splitting the original feature:

| cases | k* | HELD-OUT vs CLOSE |
|---|---|---|
| **UNAMBIGUOUS** (spell ends >=14d AFTER D) | 0.25 | **−0.000095 [−0.000157,−0.000035]** |
| **BORDERLINE** (spell ends within 14d of D) | 0.15 | −0.000022 [−0.000053,+0.000008] INCL 0 |

**The signal is entirely in the UNAMBIGUOUS cases and absent from the BORDERLINE ones —
the exact opposite of a leak signature.** A player whose injury ran two weeks past the match
was publicly, obviously out; there is no look-ahead to extract. The borderline cases, the
only place an `until` leak could operate, carry nothing.

**So the result stands, and this also explains why `out_strict` failed:** requiring a MISSED
MATCH before counting a player out discards fresh, obvious, long-term injuries — precisely
the cases that carry the signal. That rule was over-conservative in the wrong direction, not
a cleaner measurement.

## Deployment consequence

`out_unambig` uses the ACTUAL end date, so it is a diagnostic, not a deployable feature. But
the cases it isolates are ones where the ANNOUNCED expected return would classify the player
out too (a star with a three-month injury is unambiguous either way). A live implementation
should use Transfermarkt's published *expected* return date; our scrape captured only
`from`/`until`, so adding `expected return` is the one change needed before live use. That is
the real feed-reliability question for this source — not availability of the site, but whether
the expected-return field is timely and revised sensibly.
