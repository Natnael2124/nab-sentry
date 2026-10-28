"""Camera and disk health alerts (V2 Requirements 15.1-15.2, task 16.1).

Run by the Notifier on every tick (cheap: a few file stats and one query per camera).

* **Camera offline**: no new footage for ``camera_offline_factor x segment_s`` seconds. The last
  footage time is the newer of the Recorder's heartbeat (newest Segment file's mtime) and the
  end of the camera's newest ingested video. A configured camera that never produced footage
  is measured from when the Notifier started. ``camera_offline`` is raised once; when footage
  resumes, ``camera_recovered`` is raised once. The state is kept in ``kv``, so a restart does
  not raise either again.
* **Disk low**: free space on the data volume below ``disk_low_gb``, at most once per 6 hours.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from nab_sentry.config import Config
from nab_sentry.live.recorder import Heartbeat, heartbeat_path
from nab_sentry.logging_setup import get_logger
from nab_sentry.store.db import MetadataStore, NewAlert

log = get_logger("health")

DISK_REPEAT_S = 6 * 3600
DISK_KEY = "health:disk_low_last"


def offline_key(camera_id: str) -> str:
    return f"health:offline:{camera_id}"


class HealthMonitor:
    def __init__(self, cfg: Config, db: MetadataStore, cameras: dict[str, tuple[str, float]], *,
                 started_s: float, disk_free: Callable[[Path], float] | None = None) -> None:
        """``cameras``: camera_id -> (label, segment_s) from the Camera_Config."""
        self.cfg = cfg
        self.db = db
        self.cameras = cameras
        self.started_s = started_s
        self.disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
        for cam, (label, _) in cameras.items():
            db.upsert_camera(cam, label)  # alerts reference cameras(camera_id)

    def last_footage_s(self, camera_id: str) -> float | None:
        best = None
        hb = Heartbeat.read(heartbeat_path(self.cfg.live_dir, camera_id))
        if hb is not None and hb.last_segment_mtime:
            best = float(hb.last_segment_mtime)
        v = self.db.latest_video(camera_id)
        if v is not None:
            end = v[1] / 1000 + v[2]
            best = end if best is None else max(best, end)
        return best

    def offline_after_s(self, camera_id: str) -> float:
        return self.cfg.camera_offline_factor * self.cameras.get(camera_id, ("", 10.0))[1]

    def check(self, now_s: float) -> list[str]:
        """Raise any due health alerts; returns their kinds (for logs and tests)."""
        raised = []
        for cam, (label, _seg) in sorted(self.cameras.items()):
            last = self.last_footage_s(cam)
            silent_for = now_s - (last if last is not None else self.started_s)
            offline = self.db.kv_get(offline_key(cam)) == "1"
            if not offline and silent_for > self.offline_after_s(cam):
                self.db.insert_alert(NewAlert(
                    "camera_offline", round(now_s * 1000), camera_id=cam,
                    note=f"no footage for {int(silent_for // 60)} min" if last is not None
                    else "no footage received yet"))
                self.db.kv_set(offline_key(cam), "1")
                log.warning("camera %s (%s) offline: no footage for %.0f s", cam, label, silent_for)
                raised.append("camera_offline")
            elif offline and silent_for <= self.offline_after_s(cam):
                self.db.insert_alert(NewAlert("camera_recovered", round(now_s * 1000), camera_id=cam))
                self.db.kv_set(offline_key(cam), "0")
                log.info("camera %s (%s) recovered", cam, label)
                raised.append("camera_recovered")
        free_gb = self.disk_free(self.cfg.data_dir) / 1e9
        if free_gb < self.cfg.disk_low_gb:
            last = float(self.db.kv_get(DISK_KEY) or 0)
            if now_s - last >= DISK_REPEAT_S:
                self.db.insert_alert(NewAlert("disk_low", round(now_s * 1000),
                                              note=f"{free_gb:.1f} GB free (limit {self.cfg.disk_low_gb:g} GB)"))
                self.db.kv_set(DISK_KEY, str(now_s))
                log.warning("disk low: %.1f GB free", free_gb)
                raised.append("disk_low")
        return raised
