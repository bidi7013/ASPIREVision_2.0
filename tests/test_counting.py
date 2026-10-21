"""Unit tests for crossing semantics (docs/week1_definitions.md §2).

Each test is a small scripted trajectory, so the expected count is known exactly.
Run: pytest -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aspire.counting import LineCounter, Zone, anchor_points, signed_distance  # noqa: E402

# vertical line at x=100 from y=200 (p1, bottom) to y=0 (p2, top): positive side = screen right
P1, P2 = (100, 200), (100, 0)


def make(**kw):
    base = dict(line_id="L", p1=P1, p2=P2, positive_label="L2R", negative_label="R2L",
                band_px=5, segment_margin_px=0, min_track_hits=3, cooldown_frames=10)
    base.update(kw)
    return LineCounter(**base)


def feed(counter, tracks):
    """tracks: {frame: [(track_id, x, y), ...]} -> list of events"""
    events = []
    for f in sorted(tracks):
        ids = [t[0] for t in tracks[f]]
        pts = [(t[1], t[2]) for t in tracks[f]]
        events += counter.update(f, ids, pts)
    return events


def walk(tid, xs, y=100, start=0):
    return {start + i: [(tid, x, y)] for i, x in enumerate(xs)}


def test_sign_convention_positive_is_screen_right():
    assert signed_distance(P1, P2, (150, 100)) > 0
    assert signed_distance(P1, P2, (50, 100)) < 0


def test_simple_left_to_right_crossing_counts_once():
    ev = feed(make(), walk(1, range(60, 141, 10)))
    assert [(e.track_id, e.direction) for e in ev] == [(1, "L2R")]


def test_right_to_left_label():
    ev = feed(make(), walk(1, range(140, 59, -10)))
    assert [e.direction for e in ev] == ["R2L"]


def test_jitter_inside_band_does_not_count():
    # person stands on the line, box jitters ±4 px (band is 5 px)
    xs = [80, 90, 96, 104, 97, 103, 99, 101, 96, 104]
    assert feed(make(), walk(1, xs)) == []


def test_jitter_across_band_suppressed_by_cooldown():
    # crosses, then wobbles back over the band within the cooldown window
    xs = [80, 90, 110, 120, 94, 108, 92, 110]
    ev = feed(make(cooldown_frames=10), walk(1, xs))
    assert len(ev) == 1


def test_legitimate_return_after_cooldown_counts_twice():
    # walks in, waits, walks back out: two real events (one person, two events)
    xs = list(range(60, 141, 10)) + [140] * 15 + list(range(140, 59, -10))
    ev = feed(make(cooldown_frames=10), walk(1, xs))
    assert [e.direction for e in ev] == ["L2R", "R2L"]


def test_short_lived_track_is_ignored():
    # appears 1 frame before the line and crosses on its 2nd observation
    ev = feed(make(min_track_hits=3), walk(1, [90, 110]))
    assert ev == []


def test_crossing_outside_segment_extent_is_ignored():
    ev = feed(make(), walk(1, range(60, 141, 10), y=260))  # below p1 (segment ends at y=200)
    assert ev == []


def test_track_spawned_on_far_side_does_not_count():
    # detector first finds the person already past the line
    ev = feed(make(), walk(1, range(110, 161, 10)))
    assert ev == []


def test_id_switch_near_line_can_duplicate_a_count():
    """Week-2 question: how can an ID switch near the line create a duplicate?

    Track 1 crosses L->R and is counted. Just after, the tracker loses it and
    re-acquires the SAME person as track 2 whose first confirmed position is
    (because of a jittery box) back on the left side. Track 2 then 'crosses'
    again -> a second L2R event for one physical crossing. The per-track
    cooldown cannot stop this because it is a different track ID.
    """
    tracks = {}
    tracks.update(walk(1, [70, 80, 90, 110, 115]))          # frames 0-4: counted
    tracks.update(walk(2, [93, 94, 96, 108, 115], start=5))  # frames 5-9: ID switch, box jumps back
    ev = feed(make(min_track_hits=3), tracks)
    assert [(e.track_id, e.direction) for e in ev] == [(1, "L2R"), (2, "L2R")]  # documented failure mode


def test_wider_band_mitigates_that_duplicate():
    tracks = {}
    tracks.update(walk(1, [70, 80, 90, 120, 125]))
    tracks.update(walk(2, [93, 94, 96, 108, 125], start=5))
    ev = feed(make(band_px=10, min_track_hits=3), tracks)
    # track 2's early positions (93-96, 108) all fall in the ±10 band, so its
    # first CONFIRMED side is already the right side -> no duplicate
    assert [(e.track_id, e.direction) for e in ev] == [(1, "L2R")]


def test_two_people_counted_independently():
    tracks = {}
    for f, x in enumerate(range(60, 141, 10)):
        tracks[f] = [(1, x, 50), (2, 200 - x, 150)]
    ev = feed(make(), tracks)
    assert sorted((e.track_id, e.direction) for e in ev) == [(1, "L2R"), (2, "R2L")]


def test_zone_occupancy_counts_points_inside():
    z = Zone("Z", [(0, 0), (100, 0), (100, 100), (0, 100)])
    assert z.count([(50, 50), (10, 90), (150, 50), (-1, 5)]) == 2


def test_bottom_center_anchor():
    assert anchor_points([[10, 20, 30, 80]]).tolist() == [[20.0, 80.0]]
