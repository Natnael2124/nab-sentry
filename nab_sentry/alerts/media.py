"""Alert snapshots (V2 Requirement 12.5) - written by the Watcher when a rule fires.

:func:`draw_snapshot` draws the firing object's box, the rule's zone (green) or line (magenta
arrow) and a caption bar on a copy of the frame; :func:`snapshot_name` is the file name under
``data/alerts/`` (``reconcile()`` deletes files with this pattern that no ``alerts`` row
references).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import cv2
import numpy as np

from nab_sentry.rules.geometry import Point

SNAPSHOT_MAX_WIDTH = 1280
ALERT_FILE_RE = re.compile(r"^a_v\d+_[A-Za-z0-9_-]{1,64}_\d+\.(jpg|mp4)$")


def snapshot_name(video_id: int, rule_id: str, epoch_ms: int) -> str:
    return f"a_v{int(video_id)}_{rule_id}_{int(epoch_ms)}.jpg"


def _px(p: Point, w: int, h: int) -> tuple[int, int]:
    return int(round(p[0] * (w - 1))), int(round(p[1] * (h - 1)))


def draw_snapshot(image: np.ndarray, box: Sequence[int] | None, caption: str, *,
                  zone: Sequence[Point] | None = None,
                  line: tuple[Point, Point] | None = None) -> np.ndarray:
    """Annotated copy of a BGR frame, downscaled to at most ``SNAPSHOT_MAX_WIDTH`` wide."""
    out = image.copy()
    h, w = out.shape[:2]
    t = max(2, w // 400)
    if zone:
        pts = np.array([_px(p, w, h) for p in zone], np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 160, 0))
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, (0, 220, 0), t)
    if line:
        cv2.arrowedLine(out, _px(line[0], w, h), _px(line[1], w, h), (255, 0, 255), t, tipLength=0.04)
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), t + 1)
    scale = max(0.5, w / 1400)
    (tw, th), base = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    cv2.rectangle(out, (0, 0), (min(w, tw + 16), th + base + 14), (0, 0, 0), -1)
    cv2.putText(out, caption, (8, th + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t, cv2.LINE_AA)
    if w > SNAPSHOT_MAX_WIDTH:
        out = cv2.resize(out, (SNAPSHOT_MAX_WIDTH, round(h * SNAPSHOT_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return out
