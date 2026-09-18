# WP007 — Understat Player Data Exploration

**Status: complete. The data needed for the lineup-quality covariate (attacking xG/xA, appearance/minutes, and a real answer to the defensive-attribution problem) is all there, via the `understatapi` package already used by this project. No new data source needed.**

Deliberately small and non-modelling: understand exactly what's available before building anything on it. Everything below is reproduced live in `wp007_player_data_exploration.ipynb` — no claim here isn't backed by a real API call in that notebook.

## What's available

`understatapi.UnderstatClient` (already a dependency — `get_data.py` uses its `league` endpoint for team-level match data) also exposes `team` and `player` endpoints that weren't being used:

### `team.get_player_data(team, season)` — squad discovery
Season-aggregate totals for every player on a squad, and critically, the **player ID** needed for the per-match endpoint below. Fields: `id`, `player_name`, `games`, `time` (minutes), `goals`, `xG`, `assists`, `xA`, `shots`, `key_passes`, `yellow_cards`, `red_cards`, `position`, `team_title`, `npg`, `npxG`, `xGChain`, `xGBuildup`.

### `player.get_match_data(player_id)` — the actual data source needed
Per-match log for one player. **One call per player returns their entire Understat-tracked history — every season, every competition, no season parameter, no pagination.** Same fields as above but per-match: `time` (minutes *that match*), `xG`, `xA`, `npg`, `npxG`, `xGChain`, `xGBuildup`, plus `h_team`/`a_team`/`h_goals`/`a_goals`/`date`/`season`/`id` (Understat's own match ID) to place it in context.

Bruno Fernandes: 329 matches returned in one call, spanning 2014–2026 across every club and competition he's played in on Understat's radar.

## The three open questions from planning, answered

**1. Does this reach defensive players, or is it attacker-only data?** Better than expected. `xG` is ~0 for a keeper (Ederson: `xG=0`, sensible — keepers don't shoot), but **`xGChain`/`xGBuildup` are real and substantial even for a keeper** (Ederson: `xGChain=8.03`, `xGBuildup=8.03` over a season) — these credit *any* involvement in a possession that leads to a shot, including pure distribution, so they reach non-attackers through a different mechanism than xG/xA. This doesn't replace the on/off xGA-differential design (WP007's planning conversation) for measuring defensive *quality* — `xGChain`/`xGBuildup` still measure attacking contribution, just more broadly attributed — but it's a genuinely useful, already-available "overall involvement" signal for players an xG/xA-only view would show as contributing ~nothing.

**2. Appearance weighting** — solved for free. `time` is plain per-match minutes, present on every row. No separate data source needed; this was never really a distinct acquisition task, just a field already sitting in the same payload as everything else.

**3. Can this feed the on/off defensive-differential idea?** Yes — `h_goals`/`a_goals`/`date`/`h_team`/`a_team` on every row are enough to identify, for any player, which matches their team played *with* them and which they didn't (their own logged matches vs. that team's full fixture list for the period). The actual xGA-conceded number for the "without" side needs the existing team-level match data (already pulled by `get_data.py`), joined by team + date — not a new fetch, just a join.

## Two integration findings worth flagging before building anything

1. **The existing pipeline drops Understat's own match ID.** `match_transformer.py`'s `_clean_data` selects columns without `id` (`league.get_match_data()`'s raw payload has it; `df_clean` never keeps it). So there's no id-to-id join available between the player-match log and the existing `df_cv` — has to go through `(date, team, opponent)` matching instead, the same style of join WP003 built against football-data.co.uk.
2. **That join needs zero name crosswalk, unlike WP003's.** `df_cv['team_long']` and the player log's `h_team`/`a_team` are both sourced from Understat itself, so the same string ("Manchester City", "Nottingham Forest", "Wolverhampton Wanderers") appears on both sides — confirmed by direct spot-check in the notebook. WP003 needed a 6-entry crosswalk against football-data.co.uk's different naming convention; this join needs none.

## Practical notes for the acquisition script (WP008+)

- **Every numeric field comes back as a string** — `"31.65399668365717"`, not a float. Explicit casting needed everywhere; this already bit the existing team-level pipeline historically, so it's a known pattern, not a new risk.
- **Invalid player ID raises `understatapi.exceptions.InvalidPlayer` cleanly** — straightforward to catch and skip/log during a bulk pull.
- **No built-in rate limiting.** `understatapi`'s `base.py` has no delay between requests — a real pull (hundreds of unique players across 28 teams × 6 seasons) needs its own politeness delay added; this API isn't designed to be hammered with hundreds of rapid requests.
- **One call per unique player covers their whole history** — the acquisition script should discover the full set of unique player IDs first (via `team.get_player_data` for every team/season in the existing dataset, unioned), then fetch each unique ID exactly once, rather than naively calling once per team-season (which would re-fetch the same long-tenured player's full history repeatedly).

## Next

This unblocks WP008 (the actual acquisition script + leakage-free rolling aggregation, piece 0/1/2 from the earlier planning breakdown) with no remaining data-availability unknowns. Nothing here needs a second data source or a fallback plan — Understat alone covers attacking quality, appearance weighting, and the defensive on/off design.

## Reproducing

```bash
cd work_products/wp007_player_data_exploration
jupyter lab wp007_player_data_exploration.ipynb
```

Every cell is a live API call — no cached/mocked data, no CV, runs in seconds.
