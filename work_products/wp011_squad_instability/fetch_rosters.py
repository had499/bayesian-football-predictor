"""Fetch per-match player rosters from Understat and cache them.

WP011 needs player-level minutes/xG history to build leak-safe squad features
(expected XI, squad instability, absent-contributor share). Understat already
serves this via the match roster endpoint — same provider the project already
uses, just a deeper endpoint, so no new data vendor.

Scope is 2019-2025, not the full 2014-2025 the model trains on: the diagnostic
only runs on the 2020+ test matches, and one prior season is enough lead-in to
compute each team's trailing player history. That halves the fetch.

Resumable — each season is cached separately and completed matches are skipped,
so a rate-limit or a dropped connection costs only the current season's tail.

    python work_products/wp011_squad_instability/fetch_rosters.py [--seasons 2019 2020 ...]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "rosters"
SEASONS = [str(y) for y in range(2019, 2026)]
SLEEP = 0.4  # polite delay between match requests


def fetch_season(client, season, sleep=SLEEP):
    """-> {match_id: {'meta': {...}, 'h': {...}, 'a': {...}}}, resuming from cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{season}.pkl"
    out = pickle.loads(path.read_bytes()) if path.exists() else {}

    matches = client.league(league="EPL").get_match_data(season)
    todo = [m for m in matches if m.get("isResult") and m["id"] not in out]
    print(f"[{season}] {len(matches)} matches, {len(out)} cached, {len(todo)} to fetch", flush=True)

    for i, m in enumerate(todo, 1):
        mid = m["id"]
        try:
            roster = client.match(match=str(mid)).get_roster_data()
        except Exception as e:  # noqa: BLE001 — one bad match must not kill the season
            print(f"  [{season}] match {mid} FAILED: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        out[mid] = {
            "meta": {
                "match_id": mid, "datetime": m["datetime"], "season": season,
                "home_title": m["h"]["title"], "away_title": m["a"]["title"],
                "home_short": m["h"]["short_title"], "away_short": m["a"]["short_title"],
            },
            "h": roster.get("h", {}), "a": roster.get("a", {}),
        }
        if i % 25 == 0 or i == len(todo):
            path.write_bytes(pickle.dumps(out))
            print(f"  [{season}] {i}/{len(todo)} fetched", flush=True)
        time.sleep(sleep)

    path.write_bytes(pickle.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=SEASONS)
    ap.add_argument("--sleep", type=float, default=SLEEP)
    args = ap.parse_args()

    from understatapi import UnderstatClient

    client = UnderstatClient()
    total = 0
    for s in args.seasons:
        total += len(fetch_season(client, s, sleep=args.sleep))
    print(f"done: {total} matches cached across {len(args.seasons)} seasons -> {CACHE}")


if __name__ == "__main__":
    main()
