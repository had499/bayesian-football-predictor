"""Walk-forward windows for confirming a lead on fresh matches (WP014).

WP001's 35 windows each train on rounds 1..T and test ONE round (T+1), with T
stepping by 5, so 401 matches were ever held out. When an idea is found by
looking at those 401 matches, re-testing it on them proves nothing. This
builds a second set of windows that test rounds none of the original windows
tested, so the outcomes being scored are ones the idea was never selected on.

The new windows still train on everything before their own test rounds
(including the rounds the original windows tested, which is ordinary
walk-forward use of past data, not leakage of the rounds being scored).
"""
from __future__ import annotations


def tested_rounds(windows) -> set:
    """Every round some window in `windows` scores."""
    return {r for w in windows for r in range(int(w["test_start"]), int(w["test_end"]) + 1)}


def interleaved_windows(windows, offset: int, test_span: int, last_round: int) -> list:
    """One new window per original window: train through `train_end + offset`,
    then test the next `test_span` rounds (clipped to `last_round`; a window
    left with nothing to test is dropped).

    Raises if any new test round was already tested by an original window, or
    is tested by two new windows — the whole point of this function is that
    the new held-out matches are disjoint from the old ones, and that must
    fail loudly rather than quietly re-use them."""
    old = tested_rounds(windows)
    new, seen = [], set()
    for w in windows:
        train_end = int(w["train_end"]) + offset
        test_start = train_end + 1
        if test_start > last_round:
            continue
        test_end = min(test_start + test_span - 1, last_round)
        rounds = set(range(test_start, test_end + 1))
        if rounds & old:
            raise ValueError(f"rounds {sorted(rounds & old)} were already tested by an original window")
        if rounds & seen:
            raise ValueError(f"rounds {sorted(rounds & seen)} are tested by more than one new window")
        seen |= rounds
        new.append({"train_start": int(w["train_start"]), "train_end": train_end,
                    "test_start": test_start, "test_end": test_end})
    return new


def contiguous_windows(first_train_end: int, span: int, last_round: int, train_start: int = 1) -> list:
    """Windows that together test every round from `first_train_end + 1` to
    `last_round` exactly once (WP017).

    Each window trains on rounds `train_start`..T and tests the next `span`
    rounds; the next window's T advances by `span`, so test blocks tile the
    timeline with no gaps or overlaps. Compared with WP001's one-round test every
    five rounds this scores several times as many matches per fit, at the price
    of predicting up to `span` rounds past the last trained round (the model
    holds attack/defence at their final trained value), which affects every arm
    equally. The final window is clipped to `last_round`."""
    if span < 1:
        raise ValueError("span must be at least 1")
    windows, train_end = [], first_train_end
    while train_end < last_round:
        windows.append({"train_start": int(train_start), "train_end": int(train_end),
                        "test_start": int(train_end + 1), "test_end": int(min(train_end + span, last_round))})
        train_end += span
    return windows

