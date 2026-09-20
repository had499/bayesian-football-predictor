import pytest

from football_model.evaluation import windows as wmod

interleaved_windows = wmod.interleaved_windows
rounds_tested = wmod.tested_rounds   # not named test*: pytest would collect it

# The real WP001 layout: train through T = 36, 41, ..., 206, test the single round T + 1.
WP001 = [{"train_start": 1, "train_end": t, "test_start": t + 1, "test_end": t + 1} for t in range(36, 207, 5)]


def test_wp001_fixture_matches_the_real_layout():
    assert len(WP001) == 35 and rounds_tested(WP001) == {t + 1 for t in range(36, 207, 5)}


def test_new_test_rounds_are_disjoint_from_the_original_ones():
    new = interleaved_windows(WP001, offset=2, test_span=3, last_round=208)
    assert rounds_tested(new).isdisjoint(rounds_tested(WP001))


def test_layout_offset_2_span_3():
    new = interleaved_windows(WP001, offset=2, test_span=3, last_round=208)
    assert new[0] == {"train_start": 1, "train_end": 38, "test_start": 39, "test_end": 41}
    assert new[1]["test_start"] == 44 and new[1]["test_end"] == 46
    # the last original window (T=206) would need rounds 209+, which do not exist
    assert len(new) == 34 and new[-1]["test_end"] == 206


def test_test_rounds_never_overlap_their_own_training_rounds():
    for w in interleaved_windows(WP001, offset=2, test_span=3, last_round=208):
        assert w["test_start"] == w["train_end"] + 1 and w["test_end"] >= w["test_start"]


def test_last_window_is_clipped_not_dropped_when_it_partly_fits():
    new = interleaved_windows(WP001, offset=2, test_span=3, last_round=205)
    assert new[-1]["test_end"] == 205 and rounds_tested(new).isdisjoint(rounds_tested(WP001))


def test_reusing_an_original_test_round_raises():
    with pytest.raises(ValueError, match="already tested"):
        interleaved_windows(WP001, offset=0, test_span=1, last_round=208)      # the same rounds again
    with pytest.raises(ValueError, match="already tested"):
        interleaved_windows(WP001, offset=2, test_span=5, last_round=208)      # spans into the next original test round


def test_two_new_windows_testing_the_same_round_raises():
    # the originals test rounds 1 and 2 (irrelevant here); the new windows' test rounds overlap each other
    old = [{"train_start": 1, "train_end": 10, "test_start": 1, "test_end": 1},
           {"train_start": 1, "train_end": 11, "test_start": 2, "test_end": 2}]
    with pytest.raises(ValueError, match="more than one new window"):
        interleaved_windows(old, offset=1, test_span=3, last_round=50)   # tests 12-14 and 13-15


# --- WP017: contiguous windows that test every round once ---

def test_contiguous_windows_test_every_round_exactly_once():
    ws = wmod.contiguous_windows(first_train_end=74, span=5, last_round=416)
    tested = [r for w in ws for r in range(w["test_start"], w["test_end"] + 1)]
    assert tested == list(range(75, 417))                        # every round, in order, no repeats
    assert ws[0] == {"train_start": 1, "train_end": 74, "test_start": 75, "test_end": 79}
    assert ws[-1]["test_end"] == 416                             # last window clipped, not dropped


def test_contiguous_windows_never_train_on_their_own_test_rounds():
    for w in wmod.contiguous_windows(74, 5, 416):
        assert w["test_start"] == w["train_end"] + 1 and w["test_start"] <= w["test_end"]


def test_contiguous_windows_span_one_is_a_plain_walk_forward():
    ws = wmod.contiguous_windows(10, 1, 14)
    assert [(w["train_end"], w["test_start"], w["test_end"]) for w in ws] == [(10, 11, 11), (11, 12, 12), (12, 13, 13), (13, 14, 14)]


def test_contiguous_windows_reject_a_zero_span_and_handle_nothing_to_test():
    with pytest.raises(ValueError):
        wmod.contiguous_windows(10, 0, 20)
    assert wmod.contiguous_windows(20, 5, 20) == []

