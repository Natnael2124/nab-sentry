"""Search a case: text -> moments, visits, find again (Requirement 8, design: search.py)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from nab_find.store import KIND_CROP, KIND_FRAME, CaseStore

TOP_HITS = 400
MERGE_GAP_S = 6.0
PAD_S = 2.0
PEOPLE, VEHICLES = {"person"}, {"car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass
class Filters:
    start_ts: float | None = None
    end_ts: float | None = None
    kind: str = "any"  # any | people | vehicles
    camera: str | None = None


@dataclass
class Moment:
    video_id: int
    camera: str
    start_ts: float  # clock time (epoch s) of the moment, padded
    end_ts: float
    best_ts: float
    offset_s: float  # video offset of the best hit (where playback starts)
    score: float
    hits: int
    thumb: str
    visit_id: int | None
    cls: str | None = None
    label: str | None = None
    vector_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def moment_score(sims: list[float]) -> float:
    """Best similarity, plus a small bonus for supporting hits (more evidence, same place)."""
    s = sorted(sims, reverse=True)
    return float(s[0] + 0.01 * math.log1p(len(s) - 1))


class Searcher:
    def __init__(self, store: CaseStore, clip) -> None:
        self.store, self.clip = store, clip

    def _video_meta(self) -> dict[int, tuple[float, str]]:
        return {v.video_id: (v.start_ts, v.camera) for v in self.store.videos()}

    def _visits(self) -> dict[int, tuple[str, dict]]:
        return {r[0]: (r[1], json.loads(r[2])) for r in self.store.query("SELECT visit_id, cls, tags_json FROM visits")}

    def _mask(self, t: dict, f: Filters, vmeta: dict, visits: dict) -> np.ndarray:
        n = len(t["ids"])
        mask = np.ones(n, bool)
        if f.start_ts is not None or f.end_ts is not None or f.camera:
            start = np.array([vmeta.get(int(v), (0.0, ""))[0] for v in t["video"]]) + t["offset"]
            if f.start_ts is not None:
                mask &= start >= f.start_ts
            if f.end_ts is not None:
                mask &= start <= f.end_ts
            if f.camera:
                mask &= np.array([vmeta.get(int(v), (0, ""))[1] == f.camera for v in t["video"]], bool)
        if f.kind in ("people", "vehicles"):
            want = PEOPLE if f.kind == "people" else VEHICLES
            cls_ok = np.array([visits.get(int(vi), ("", {}))[0] in want for vi in t["visit"]], bool)
            mask &= (t["kind"] == KIND_CROP) & cls_ok
        return mask

    def search(self, query: str, f: Filters | None = None, limit: int = 60, qvec: np.ndarray | None = None,
               exclude_visit: int | None = None, crops_only: bool = False) -> list[Moment]:
        f = f or Filters()
        with self.store.lock:
            t = {k: v.copy() for k, v in self.store.table.view().items()}
        if len(t["ids"]) == 0:
            return []
        vmeta, visits = self._video_meta(), self._visits()
        q = self.clip.encode_text(query) if qvec is None else qvec
        mask = self._mask(t, f, vmeta, visits)
        if crops_only:
            mask &= t["kind"] == KIND_CROP
        if exclude_visit is not None:
            mask &= t["visit"] != exclude_visit
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        sims = t["vecs"][idx] @ q
        k = min(TOP_HITS, idx.size)
        top = idx[np.argpartition(-sims, k - 1)[:k]] if k < idx.size else idx
        sim_of = dict(zip(idx.tolist(), sims.tolist()))
        return self._group(top, t, sim_of, vmeta, visits, limit)

    def _group(self, top, t, sim_of, vmeta, visits, limit) -> list[Moment]:
        rows = {r[0]: r[1:] for r in self.store.query(
            f"SELECT vector_id, thumb, kind FROM vectors WHERE vector_id IN ({','.join(str(int(t['ids'][i])) for i in top)})")}
        order = sorted(top.tolist(), key=lambda i: (int(t["video"][i]), float(t["offset"][i])))
        groups: list[list[int]] = []
        for i in order:
            g = groups[-1] if groups else None
            if g and int(t["video"][g[-1]]) == int(t["video"][i]) and t["offset"][i] - t["offset"][g[-1]] <= MERGE_GAP_S:
                g.append(i)
            else:
                groups.append([i])
        out = []
        for g in groups:
            best = max(g, key=lambda i: sim_of[i])
            vid = int(t["video"][best])
            start_ts, camera = vmeta[vid]
            visit = int(t["visit"][best])
            cls, tags = visits.get(visit, (None, {})) if visit >= 0 else (None, {})
            offs = [float(t["offset"][i]) for i in g]
            out.append(Moment(
                video_id=vid, camera=camera, start_ts=start_ts + max(0.0, min(offs) - PAD_S),
                end_ts=start_ts + max(offs) + PAD_S, best_ts=start_ts + float(t["offset"][best]),
                offset_s=float(t["offset"][best]), score=moment_score([sim_of[i] for i in g]), hits=len(g),
                thumb=rows.get(int(t["ids"][best]), ("",))[0], visit_id=visit if visit >= 0 else None,
                cls=cls, label=tags.get("phrase") if tags else None,
                vector_ids=[int(t["ids"][i]) for i in sorted(g, key=lambda i: -sim_of[i])[:8]]))
        out.sort(key=lambda m: (-m.score, m.best_ts))
        return out[:limit]

    def similar(self, visit_id: int, f: Filters | None = None, limit: int = 30) -> list[Moment]:
        """Other visits that look like ``visit_id`` (appearance only), one moment per visit."""
        vecs = self.store.vectors_of_visit(visit_id)
        if len(vecs) == 0:
            return []
        q = vecs.mean(axis=0)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        moments = self.search("", f, limit=400, qvec=q, exclude_visit=visit_id, crops_only=True)
        best: dict[int, Moment] = {}
        for m in moments:
            if m.visit_id is not None and (m.visit_id not in best or m.score > best[m.visit_id].score):
                best[m.visit_id] = m
        return sorted(best.values(), key=lambda m: -m.score)[:limit]

    def visits(self, f: Filters | None = None, sort: str = "time", min_s: float = 0.0, limit: int = 500) -> list[dict]:
        f = f or Filters()
        vmeta = self._video_meta()
        rows = self.store.query("SELECT visit_id, video_id, cls, t0, t1, n_obs, max_speed, tags_json, thumb "
                                "FROM visits WHERE thumb IS NOT NULL")
        out = []
        for vid_, video_id, cls, t0, t1, n, speed, tags, thumb in rows:
            start_ts, camera = vmeta.get(video_id, (0.0, ""))
            a, b = start_ts + t0, start_ts + t1
            if f.start_ts is not None and b < f.start_ts or f.end_ts is not None and a > f.end_ts:
                continue
            if f.camera and camera != f.camera:
                continue
            if f.kind == "people" and cls not in PEOPLE or f.kind == "vehicles" and cls not in VEHICLES:
                continue
            if t1 - t0 < min_s:
                continue
            tg = json.loads(tags)
            out.append({"visit_id": vid_, "video_id": video_id, "camera": camera, "cls": cls, "start_ts": a,
                        "end_ts": b, "duration_s": round(t1 - t0, 1), "offset_s": t0, "max_speed": speed,
                        "label": tg.get("phrase", cls), "thumb": thumb})
        key = {"time": lambda r: r["start_ts"], "longest": lambda r: -r["duration_s"],
               "fastest": lambda r: -r["max_speed"]}.get(sort, lambda r: r["start_ts"])
        return sorted(out, key=key)[:limit]
