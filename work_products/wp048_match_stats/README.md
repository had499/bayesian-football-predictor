# WP048 — football-data's match stats: nothing the champion lacks

**Verdict: DEAD, ~20 minutes, no sweeps. Shots on target, corners, fouls and half-time
goals all carry nothing once the champion's xG and deep channels are present. `deep`
already captures the territory signal.**

## What was new to test

WP038 closed the Understat shot-level feed, but football-data's main-league files carry
per-match counts never touched here, all at 100% coverage and already cached by WP044:
`HST/AST` (shots on target), `HC/AC` (corners), `HF/AF` (fouls), `HTHG/HTAG` (half-time
goals). Join: 4,154 matches, 100% score agreement.

## Round 1 — on top of xG alone

    specification                     OOS R^2    vs xG alone
    xG only (incumbent)              +0.07450          —
    + goals     [CONTROL, real]      +0.07581     +0.00130
    + TOTAL shots [ANCHOR: FAILED]   +0.07812     +0.00362
    + shots on target                +0.07728     +0.00277
    + corners                        +0.07864     +0.00414
    + fouls                          +0.07460     +0.00010
    + half-time goals                +0.07473     +0.00023

**The anchor is built in.** Total shots is included deliberately: WP019 BUILT it as a
channel and it was null, actively cancelling deep (+0.00082 worse). So +0.00362 is the
level at which a feature is KNOWN to fail. Shots on target (+0.00277) is below it. Fouls
and half-time goals are ~zero. Only corners (+0.00414) cleared it, by 14%.

## Round 2 — the decisive test, on top of xG AND deep

Corners and `deep` are both territory proxies, and WP021 measured shots and deep
actively cancelling. So the real question is whether corners adds anything to what the
champion already has:

    xG + deep (the champion)         +0.08741          —
    + goals     [CONTROL, real]      +0.08806     +0.00065
    + corners                        +0.08787     +0.00046
    + shots on target                +0.08786     +0.00045
    + corners & sot                  +0.08816     +0.00075

**Corners collapses from +0.00414 to +0.00046 once deep is present** — an order of
magnitude below the known-failure anchor. It is the WP021 cancellation again: two
measures of the same underlying signal, and deep is the better one.

## A note on this metric's inflation

`deep` is worth **+0.01291** on this test — more than anything else measured — while its
actual RPS contribution is only −0.00083. The premise metric over-promises by roughly an
order of magnitude, which is consistent with BACKLOG caution 13 and is why the
known-failure ANCHOR matters more than the absolute number.

## Where accessible data now stands

* **Understat match-aggregate** — exhausted (xG, npxG, shots, deep).
* **Understat shot-level** — exhausted (WP038: state-free xG, set-piece split, big
  chances, shots-in-box all below the anchor).
* **football-data match stats** — exhausted (this).
* **Absence / suspensions** — closed on five tests (WP047).
* **FBref (post-shot xG, i.e. directly observable shot-stopping)** — **403, Cloudflare-
  blocked.** This is the one genuinely promising unreached source: WP016's split-defence
  failed because shot-stopping was only identifiable from the noisy goals-minus-xG
  residual, and PSxG would make it an observation instead. Needs a mirror or a paid feed.
* **Player-level contribution data** (StatsBomb / Stats Perform / Wyscout) — the only
  remaining lever that addresses a measured gap, and it is a five-figure purchase.

    python work_products/wp048_match_stats/premise.py
