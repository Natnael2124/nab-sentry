"""Re-encode a store with the configured embedder (V2 Step 4).

Changing the embedder changes the vector space, so the whole Vector_Index has to be rebuilt. The
Metadata_Store keeps everything else - frames, boxes, tracks, alerts, activities - and every
``vector_id``: each stored frame and crop is decoded again from its source video at the stored
frame index (the same decoder ingest used) and box, encoded, and added under its old ID to a new
index. The detector, tracker and rules do not run again.

Order of the commit, so a crash at any point leaves a state that is either fully old, fully new,
or detected as mismatched (and fixed by running again):

1. build the new index in memory (nothing on disk changes);
2. ``VectorIndex.save`` replaces the index file atomically;
3. ``schema_meta.embed_model`` is set to the new embedder.

Videos whose source file is missing cannot be re-encoded. :func:`reembed` refuses to start
(:class:`MissingSources`) unless ``drop_missing``: then they are deleted the way retention deletes
a video (rows, thumbnails, playback copy).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import playback_name
from nab_sentry.ingest.sources import FileSource
from nab_sentry.logging_setup import get_logger
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex

log = get_logger("reembed")


class MissingSources(RuntimeError):
    def __init__(self, missing: list[tuple[int, str]]) -> None:
        self.missing = missing
        names = ", ".join(Path(s).name for _, s in missing[:5]) + (" ..." if len(missing) > 5 else "")
        super().__init__(f"{len(missing)} stored video(s) have no source file any more ({names}); "
                         f"run again with --drop-missing to delete them from the store")


@dataclass
class ReembedReport:
    videos: int = 0
    vectors: int = 0
    dropped_videos: list[int] = field(default_factory=list)
    dropped_vectors: int = 0


@dataclass(frozen=True)
class Item:
    vector_id: int
    frame_idx: int
    box: tuple[int, int, int, int] | None  # None: the whole frame


def plan(db: MetadataStore) -> list[tuple[int, str, list[Item]]]:
    """(video_id, source path, items in decode order) for every video with vectors."""
    with db._lock:  # noqa: SLF001 - read-only query over three tables
        rows = db._conn.execute(  # noqa: SLF001
            "SELECT vi.video_id, vi.src_path, f.frame_idx, v.vector_id, v.kind, v.x1, v.y1, v.x2, v.y2 "
            "FROM vectors v JOIN frames f USING (frame_id) JOIN videos vi USING (video_id) "
            "ORDER BY vi.video_id, f.frame_idx, v.vector_id").fetchall()
    out: list[tuple[int, str, list[Item]]] = []
    for vid, src, idx, vector_id, kind, x1, y1, x2, y2 in rows:
        if not out or out[-1][0] != vid:
            out.append((vid, src, []))
        box = None if kind == "frame" else (int(x1), int(y1), int(x2), int(y2))
        out[-1][2].append(Item(int(vector_id), int(idx), box))
    return out


def source_path(cfg: Config, src: str) -> Path:
    p = Path(src)
    return p if p.is_absolute() else Path(cfg.root) / p


def images(path: Path, items: Sequence[Item],
           open_source: Callable[[Path], Any] = FileSource.open) -> Iterator[tuple[int, np.ndarray]]:
    """(vector_id, image) for ``items``, decoding the video once in order."""
    wanted: dict[int, list[Item]] = {}
    for it in items:
        wanted.setdefault(it.frame_idx, []).append(it)
    last = max(wanted)
    src = open_source(path)
    try:
        for frame in src.frames():
            if frame.index > last:
                break
            for it in wanted.pop(frame.index, []):
                img = frame.image if it.box is None else frame.image[it.box[1]:it.box[3], it.box[0]:it.box[2]]
                yield it.vector_id, img
            if not wanted:
                break
    finally:
        src.close()
    if wanted:
        raise RuntimeError(f"{path.name}: frames {sorted(wanted)[:5]} could not be decoded again")


def reembed(cfg: Config, db: MetadataStore, encoder: Any, *, drop_missing: bool = False,
            progress: Callable[[int, int], None] | None = None,
            open_source: Callable[[Path], Any] = FileSource.open) -> ReembedReport:
    model_id = getattr(encoder, "model_id", None)
    dim = int(getattr(encoder, "dim", 512))
    videos = plan(db)
    missing = [(vid, src) for vid, src, _ in videos if not source_path(cfg, src).is_file()]
    if missing and not drop_missing:
        raise MissingSources(missing)
    rep = ReembedReport()
    gone = {vid for vid, _ in missing}
    keep = [(vid, src, items) for vid, src, items in videos if vid not in gone]
    total = sum(len(items) for _, _, items in keep)
    index = VectorIndex(dim=dim)
    batch_n = max(1, int(cfg.batch_size))
    done = 0
    for vid, src, items in keep:
        ids: list[int] = []
        imgs: list[np.ndarray] = []

        def flush() -> None:
            nonlocal done
            vecs = np.asarray(encoder.encode_images(imgs), dtype=np.float32)
            if vecs.shape != (len(imgs), dim) or not np.all(np.isfinite(vecs)):
                raise RuntimeError(f"encoder returned {vecs.shape} / non-finite values")
            index.add(np.asarray(ids, dtype=np.int64), vecs)
            done += len(ids)
            ids.clear()
            imgs.clear()
            if progress is not None:
                progress(done, total)

        for vector_id, img in images(source_path(cfg, src), items, open_source):
            ids.append(vector_id)
            imgs.append(img)
            if len(imgs) >= batch_n:
                flush()
        if imgs:
            flush()
        rep.videos += 1
    # Only now, with the new index complete, delete the videos that cannot be re-encoded: an
    # interruption before this point leaves the store and the old index untouched.
    for vid in sorted(gone):
        rep.dropped_vectors += len(db.vector_ids_for_video(vid))
        for name in db.delete_video(vid):
            folder = cfg.playback_dir if name == playback_name(vid) else cfg.thumbs_dir
            try:
                (folder / name).unlink()
            except OSError:
                pass
        rep.dropped_videos.append(vid)
    if gone:
        db.delete_empty_tracks()
        log.warning("dropped %d video(s) without a source file (%d vectors)", len(gone), rep.dropped_vectors)
    stored = np.unique(db.vector_ids())
    built = np.unique(np.asarray(index.ids(), dtype=np.int64))
    if not np.array_equal(stored, built):
        raise RuntimeError(f"new index has {built.size} IDs, the store {stored.size}: nothing was changed")
    index.save(cfg.index_path)
    if model_id:
        db.set_embed_model(model_id, dim)
    rep.vectors = int(built.size)
    return rep

