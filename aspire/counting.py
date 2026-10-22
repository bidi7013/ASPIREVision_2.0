"""Directional line-crossing counter and region occupancy.

All geometry is in IMAGE SPACE (pixels). No physical units are implied.

Crossing semantics (see docs/week1_definitions.md, section 2):
  * A crossing EVENT is emitted when a track's anchor point moves from one
    *confirmed* side of the counting line to the other confirmed side.
  * A side is only "confirmed" when the anchor is more than `band_px` pixels
    away from the line. Inside the band the previous side is kept. This
    hysteresis stops box jitter on the line from producing repeated events.
  * The crossing must happen within the extent of the line segment (with
    `segment_margin_px` tolerance), not anywhere along its infinite extension.
  * A track must have been observed for `min_track_hits` frames before it may
    emit an event (suppresses short-lived spurious tracks).
  * After an event, the same track cannot emit another event on the same line
    for `cooldown_frames` frames. Legitimate repeat crossings (walk in, later
    walk out) are still counted once the cooldown has passed.
  * A track that is FIRST SEEN already on a side has that side recorded as its
    start side; it emits nothing until it actually changes side.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- geometry
def anchor_points(xyxy: np.ndarray, mode: str = "bottom_center") -> np.ndarray:
    """Return an (N, 2) array of anchor points for (N, 4) boxes."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    if mode == "bottom_center":
        return np.stack([cx, xyxy[:, 3]], axis=1)
    if mode == "center":
        return np.stack([cx, (xyxy[:, 1] + xyxy[:, 3]) / 2.0], axis=1)
    raise ValueError(f"unknown anchor mode: {mode}")


def signed_distance(p1, p2, pt) -> float:
    """Signed perpendicular distance (px) of `pt` from the line p1->p2.

    Positive = to the RIGHT of the direction p1->p2 as seen on screen
    (image coordinates, y grows downward). Example: p1 at the bottom, p2 at
    the top of a vertical line -> positive side is the screen-right side.
    """
    p1, p2, pt = (np.asarray(v, dtype=float) for v in (p1, p2, pt))
    d = p2 - p1
    length = float(np.hypot(*d))
    if length == 0:
        raise ValueError("counting line has zero length")
    # 2D cross product: its sign gives the side; dividing by the line length gives pixels
    cross = d[0] * (pt[1] - p1[1]) - d[1] * (pt[0] - p1[0])
    return cross / length


def projection_t(p1, p2, pt) -> float:
    """Position of pt's projection along the segment (0 at p1, 1 at p2)."""
    p1, p2, pt = (np.asarray(v, dtype=float) for v in (p1, p2, pt))
    d = p2 - p1
    return float(np.dot(pt - p1, d) / np.dot(d, d))


# ----------------------------------------------------------------- counting
@dataclass
class CrossingEvent:
    frame: int
    track_id: int
    line_id: str
    direction: str  # the configured label, e.g. "eastbound"
    sign: int  # +1 or -1 (negative side -> positive side is +1)
    anchor_x: float
    anchor_y: float


@dataclass
class _TrackState:
    hits: int = 0
    side: int = 0  # last confirmed side: -1, +1, or 0 (never confirmed)
    last_event_frame: int | None = None


@dataclass
class LineCounter:
    line_id: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    positive_label: str = "positive"  # label for a -1 -> +1 crossing
    negative_label: str = "negative"  # label for a +1 -> -1 crossing
    band_px: float = 8.0
    segment_margin_px: float = 10.0
    min_track_hits: int = 3
    cooldown_frames: int = 10
    _tracks: dict[int, _TrackState] = field(default_factory=dict, repr=False)

    def _within_segment(self, pt) -> bool:
        length = float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))
        margin = self.segment_margin_px / length  # px -> fraction of the segment, to compare with t
        t = projection_t(self.p1, self.p2, pt)
        return -margin <= t <= 1.0 + margin

    def _side(self, pt) -> int:
        d = signed_distance(self.p1, self.p2, pt)
        if d > self.band_px:
            return 1
        if d < -self.band_px:
            return -1
        return 0  # inside the hysteresis band

    def update(self, frame: int, track_ids, anchors) -> list[CrossingEvent]:
        """Feed one frame of tracked anchors; return any new events."""
        events: list[CrossingEvent] = []
        for tid, pt in zip(track_ids, anchors):
            tid = int(tid)
            st = self._tracks.setdefault(tid, _TrackState())
            st.hits += 1
            side = self._side(pt)
            if side == 0:
                continue  # in band: keep previous confirmed side
            if st.side == 0:
                st.side = side  # first confirmed side: start side, no event
                continue
            if side == st.side:
                continue
            # confirmed side change
            eligible = (
                st.hits >= self.min_track_hits
                and self._within_segment(pt)
                and (
                    st.last_event_frame is None
                    or frame - st.last_event_frame >= self.cooldown_frames
                )
            )
            # update the side even when the event is suppressed, so a blocked crossing
            # (too young, off-segment, in cooldown) can never fire later
            st.side = side
            if eligible:
                st.last_event_frame = frame
                events.append(
                    CrossingEvent(
                        frame=frame,
                        track_id=tid,
                        line_id=self.line_id,
                        direction=self.positive_label if side == 1 else self.negative_label,
                        sign=side,
                        anchor_x=float(pt[0]),
                        anchor_y=float(pt[1]),
                    )
                )
        return events


# ---------------------------------------------------------------- occupancy
@dataclass
class Zone:
    zone_id: str
    polygon: list[tuple[float, float]]

    def contains(self, pt) -> bool:
        """Ray-casting point-in-polygon test (boundary counts as inside-ish)."""
        x, y = float(pt[0]), float(pt[1])
        inside = False
        poly = self.polygon
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):  # this edge straddles the horizontal ray through pt
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi  # where the edge meets that ray
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def count(self, anchors) -> int:
        """Occupancy = number of tracked anchors inside the zone in this frame."""
        return int(sum(self.contains(p) for p in anchors))
