"""Percent done and time left (Requirement 7, design: "Progress and time left").

Work is measured in seconds of footage per pass. Speed (footage seconds per wall second) is a
moving average per pass, seeded from what this PC measured last time (or a default), and after
the first few updates it may change by at most 2x per update, so the time left never jumps wildly.

The percent is time-weighted: each pass's footage counts by how long it is expected to take, so the
bar moves at a steady pace instead of racing through the quick pass and crawling through detail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

PASSES = ("quick", "detail")
DEFAULT_SPEED = {"quick": 60.0, "detail": 10.0}  # x real time, before anything is measured
DETAIL_GUESS = 0.4  # share of the unread range expected to need the detail pass
ALPHA = 0.3
WARM_UPDATES = 3


@dataclass
class PassState:
    total_s: float = 0.0
    done_s: float = 0.0
    speed: float = 1.0
    updates: int = 0

    @property
    def left_s(self) -> float:
        return max(0.0, self.total_s - self.done_s)


@dataclass
class Progress:
    quick_total_s: float
    seed: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p = {k: PassState(speed=float(self.seed.get(k) or DEFAULT_SPEED[k])) for k in PASSES}
        self.p["quick"].total_s = max(0.0, self.quick_total_s)
        self.detail_found_s = 0.0  # moving footage found by the quick pass so far (detail work)
        self.detail_known = False
        self.current = "quick"
        self._refresh_detail_total()

    def _refresh_detail_total(self) -> None:
        q, d = self.p["quick"], self.p["detail"]
        guess = 0.0 if self.detail_known else DETAIL_GUESS * q.left_s
        d.total_s = max(d.done_s, self.detail_found_s + guess)

    def set_detail_found(self, seconds: float, final: bool = False) -> None:
        """Detail work known so far (padded moving stretches); ``final`` once the quick pass ended."""
        self.detail_found_s = max(0.0, float(seconds))
        self.detail_known = final
        self._refresh_detail_total()

    def update(self, pass_name: str, footage_s: float, wall_s: float) -> None:
        """``footage_s`` more footage of ``pass_name`` done, taking ``wall_s`` seconds."""
        st = self.p[pass_name]
        self.current = pass_name
        st.done_s = min(st.total_s if st.total_s > 0 else math.inf, st.done_s + max(0.0, footage_s))
        if footage_s > 0 and wall_s > 0:
            measured = footage_s / wall_s
            new = measured if st.updates == 0 else (1 - ALPHA) * st.speed + ALPHA * measured
            if st.updates >= WARM_UPDATES:
                new = min(max(new, st.speed / 2), st.speed * 2)
            st.speed = max(new, 1e-3)
            st.updates += 1
        self._refresh_detail_total()

    def eta(self, pass_name: str) -> float:
        st = self.p[pass_name]
        return st.left_s / st.speed

    def snapshot(self) -> dict:
        q, d = self.p["quick"], self.p["detail"]
        work_total = q.total_s / q.speed + d.total_s / d.speed
        work_done = q.done_s / q.speed + d.done_s / d.speed
        percent = 100.0 if work_total <= 0 else min(100.0, max(0.0, 100.0 * work_done / work_total))
        quick_eta = self.eta("quick")
        return {
            "pass": self.current,
            "percent": round(percent, 1),
            "quick_done_s": q.done_s, "quick_total_s": q.total_s,
            "detail_done_s": d.done_s, "detail_total_s": d.total_s, "detail_known": self.detail_known,
            "quick_eta_s": round(quick_eta, 1),
            "full_eta_s": round(quick_eta + self.eta("detail"), 1),
            "speed_x": round((q if self.current == "quick" else d).speed, 1),
        }

    def speeds(self) -> dict[str, float]:
        """Measured speeds to seed the next run on this PC (only passes that ran)."""
        return {k: st.speed for k, st in self.p.items() if st.updates > 0}
