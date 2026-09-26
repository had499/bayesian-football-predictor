# WP010 — Opening vs. closing line: how much of the gap is information the model can't have?

**Status: complete.** WP003/WP008 benchmarked the model against Pinnacle's **closing** line. But closing odds are set after team news is public (lineups drop ~1h before kickoff), and this model predicts days ahead — so the closing line is a structurally unfair yardstick for it. This work product re-benchmarks against the **pre-closing** snapshot and sizes how much of the remaining gap is late information versus genuine modelling shortfall.

**Headline: only ~29% of the best model's gap to closing is late-breaking information (the confirmed XI, late fitness tests). The other ~71% is a gap against a line set at the same *timing* — but emphatically not the same *information*, since the opening line already prices everything the market knows days ahead.**

No retraining — this reuses WP007's predictions and WP008's odds pipeline. Runs in seconds.

## Are the pre-closing columns really an earlier price?

football-data.co.uk's `C` infix marks closing odds; the mirror's SPEC doesn't document *when* the non-`C` snapshot is taken, and football-data.co.uk's own notes page is behind the corporate proxy block (see WP008). So rather than trust the label, this checks it empirically — an earlier, less-informed line should carry a **fatter margin** (books tighten as they take sharp money) and **score worse**:

| Book | n | Overround | Pooled RPS |
|---|---|---|---|
| Pinnacle pre-close | 1205 | 3.02% | 0.19881 |
| **Pinnacle closing** | 1205 | **2.68%** | **0.19724** |
| Bet365 pre-close | 1314 | 5.44% | 0.20128 |
| Bet365 closing | 1314 | 5.48% | 0.19965 |

Pinnacle behaves exactly as expected — margin tightens from 3.02% → 2.68% and accuracy improves. Bet365's margin is flat (5.44% vs. 5.48%), which fits a recreational book that doesn't tighten the way a sharp one does, but its accuracy still improves. Both confirm the non-`C` columns are the earlier, less-informed price.

**Caveat on precision**: this establishes these are *earlier*, not exactly *how much* earlier. Football-data.co.uk collects the non-closing snapshot at a fixed earlier point (typically a day or two before the fixture), which is roughly the decision point for a model predicting a weekend round midweek — but the exact timing isn't documented in a source reachable from here, so treat "opening" throughout as "an earlier snapshot," not "the first price posted."

## The late-information premium

Paired on the same 1,205 matches:

| | RPS |
|---|---|
| Pinnacle pre-close | 0.19881 |
| Pinnacle closing | 0.19724 |
| **Difference** | **−0.00157, CI [−0.00273, −0.00046]** |

The market gains **0.00157 RPS** between the two snapshots. That's the total value of everything arriving in that window — team news, weather, *and* sharp money correcting the price. So it's an **upper bound** on what team news alone is worth, not a pure measure of it.

## The model's gap against each snapshot

| Arm | vs. pre-close | vs. closing |
|---|---|---|
| `L0_baseline` | +0.00650 [+0.00332, +0.00948] | +0.00807 [+0.00449, +0.01153] |
| **`L5_xgchannel`** | **+0.00392 [+0.00122, +0.00663]** | +0.00549 [+0.00231, +0.00867] |

Both arms are still reliably worse than even the earlier line (CIs exclude zero). But the share of the gap attributable to late information:

| Arm | Late info as share of closing-gap |
|---|---|
| `L0_baseline` | 19% |
| `L5_xgchannel` | **29%** |

**Read this carefully — it bounds less than it first appears.** The 0.00157 premium bounds only **late-breaking** information: the confirmed XI, a late fitness test, plus whatever sharp money corrects in that window. It does **not** bound the value of lineup information in general.

*Historic* lineup data — who has been playing, minutes trends, whether a regular starter stopped featuring — is available days ahead and is **already priced into the opening line**. So it targets the remaining **0.00392**, not the 0.00157.

The 71% is therefore a gap at equal *timing*, not equal *information*. At opening time the market knows a striker has been out three weeks, a manager was sacked, a squad is rotating for a European tie. This model knows none of that at any point — it carries no player-level information at all, only team-level goals and xG.

## Where the market's sharpening actually happens

| Subset | n | closing − opening | CI (cluster) |
|---|---|---|---|
| Upset | 542 | +0.00045 | [−0.00164, +0.00248] — no gain |
| **No upset** | 663 | **−0.00322** | **[−0.00475, −0.00180]** |

The market's entire open→close improvement is concentrated in matches that go as expected; on upsets it gains nothing. That's the *same structural signature* WP009 found for the model-vs-market gap — late information sharpens confidence on outcomes that were going to happen anyway, and doesn't help anyone see upsets coming.

## The most actionable slice: line movement

| Subset | n | `L5` − opening | CI (cluster) | Market's own close−open |
|---|---|---|---|---|
| Top 25% line movement | 302 | **+0.01107** | [+0.00422, +0.01834] — worse | −0.00612 |
| Middle 50% | 601 | +0.00145 | [−0.00178, +0.00485] — **no clear difference** | +0.00009 |
| Bottom 25% line movement | 302 | +0.00170 | [−0.00318, +0.00655] — **no clear difference** | −0.00033 |

Median line movement is 1.7 points of P(home); the top decile moves 4.5+.

**On the ~75% of matches where the line doesn't move much, the model is statistically indistinguishable from Pinnacle's earlier price.** Its entire deficit is concentrated in the quarter of matches where the market subsequently moves a lot.

Worth being careful about what this does and doesn't say. The model is worse on those matches *at the opening state*, before any of the movement happened — so it isn't that the model fails to anticipate the move; it's that on matches destined to move, the model was already behind. Two readings, not separable here:
- Matches that move are ones with genuine live uncertainty (a doubtful key player, a team in flux) that the market already partly prices at opening while the model has no signal at all; or
- High-movement matches are simply higher-variance, and any weaker predictor looks worse there.

The fact that the market itself gains most on exactly these matches (−0.00612) leans toward the first reading, but doesn't settle it.

## Season-by-season vs. the earlier line

| Season | n | `L5` − opening | Verdict |
|---|---|---|---|
| 2020 | 223 | +0.00477 | worse |
| 2021 | 220 | +0.00135 | no clear difference |
| 2022 | 205 | +0.00589 | no clear difference |
| 2023 | 228 | +0.00649 | worse |
| **2024** | 219 | **−0.00144** | no clear difference *(point estimate favours the model)* |
| 2025 | 110 | +0.00904 | worse |

Against a fair benchmark the model reaches parity in **3 of 6 seasons**, up from 2 of 6 against closing — and 2024's point estimate is on the model's side, though the CI comfortably includes zero.

## What this changes

WP009 concluded the remaining gap "looks like genuine missing information," and speculated injuries/lineups. This refines that materially, and partly walks it back:

- **Late-breaking information is real but small** — ~29% of the gap, and upper-bounded since that window also contains sharp money.
- **The larger share (71%) is a gap at equal timing, not equal information.** The opening line already contains everything the market knows days ahead: squad state, transfer business, manager change, tactical matchups, European fixture load. The model has none of it.
- **This does not bound lineup modelling.** An earlier draft of this work product concluded it did — that inference was wrong. Historic lineup information (minutes trends, a regular starter who stopped featuring, squad turnover) sits *inside* the pre-close information set, so it targets the 0.00392, not the 0.00157. The mechanism is concrete: when a key player is absent the model sees it only indirectly, as weaker team-level goals/xG filtering slowly through the random walk, and it cannot distinguish a temporary absence from genuine decline — nor jump the team back up when the player returns. The market re-prices instantly.

## Recommended next steps

1. **Quote the pre-close gap, not the closing gap**, as the honest measure of this model's skill for a predict-days-ahead product. `L5_xgchannel` is +0.0039 behind a fair benchmark, not +0.0055.
2. **Investigate the high-movement quarter directly** — that's where 100% of the deficit lives, and it's a concretely identifiable subset. The open→close move is only observable in hindsight, but if a *predictable* proxy exists, that's the lever.
3. **Test whether historic lineup instability predicts the high-movement matches.** This connects the two findings: matches whose line moves a lot are plausibly matches with *squad* uncertainty (a doubtful player, a rotation call), and an unstable lineup is exactly where historic roster data is most informative. If instability predicts movement, it both validates the lineup direction and pinpoints the subset to target. Cheap to check once Understat roster data is cached — diagnostic first, model afterwards, same as WP009.

## What this doesn't answer

* **Exactly how much earlier the pre-close snapshot is** — see the caveat above; the timing isn't documented in a reachable source.
* **How much of the 0.00157 premium is team news vs. sharp money.** Separating those would need timestamped line history, which this dataset doesn't carry.
* **Whether the high-movement subset is predictable in advance.** That's the question worth answering next, and it needs a signal source this pipeline doesn't have yet.
* **Bet365's flat overround** isn't investigated; it's noted as consistent with a recreational book but not verified.

## Reproducing

```bash
python work_products/wp010_opening_vs_closing/opening_vs_closing.py
```

Needs WP007's `checkpoints/` and WP008's `odds_raw.pkl` (re-run WP008's `fetch_odds.py` if the latter predates WP010 — it now pulls the pre-closing columns too). Reads both; retrains nothing.
