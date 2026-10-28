"""Embedder comparison for search (V2 Step 4).

Usage::

    venv\\Scripts\\python.exe scripts\\embed_eval.py encode MODEL [--db data\\nab_sentry.db]
        [--queries eval\\queries_compilation.yaml] [--out data\\bench\\embed] [--limit N]
    venv\\Scripts\\python.exe scripts\\embed_eval.py score [MODEL ...] [--db ...] [--queries ...]

``encode`` re-encodes exactly the images the product stored - every ``vectors`` row of the
Metadata_Store: whole frames and detector crops, decoded again from the source video at the
stored frame index and box - plus the query texts, with one candidate MODEL. It writes
``OUT/<MODEL>.npz`` (vector IDs, image vectors, query vectors) and ``OUT/<MODEL>.json``
(load time, ms per image at batch 8, ms per query, peak memory). It needs only numpy, OpenCV,
PyYAML, psutil and the model's own library, so it can run in a separate virtual environment
(EmbeddingGemma 2 needs a newer ``transformers`` than the product does).

``score`` puts each model's vectors into a :class:`~nab_sentry.store.vector_index.VectorIndex`
under the same IDs and runs every query through the product's own
:class:`~nab_sentry.search.engine.SearchEngine` (top-k, clustering into Events, padding), on a
copy of the store. Per query it records the rank of the first relevant Event (camera and
window as in ``scripts/evaluate.py``) and reports hit@1, hit@3, MRR and p@5 per model, the
per-query ranks side by side, and the speed numbers. Results: ``OUT/report.md``.

Models (``MODEL``):

* ``product`` - the product's encoder from ``models/`` (``nab_sentry.model_files.EMBEDDER_ID``)
* ``b32-laion2b`` - ViT-B-32 laion2b_s34b_b79k, the product's embedder up to Step 3 (the baseline)
* ``b32-datacomp-xl`` (ViT-B-32, datacomp_xl_s13b_b90k) and ``b32-256-datacomp`` (ViT-B-32-256,
  datacomp_s34b_b86k) - the same architecture trained on DataComp (MIT), a drop-in change
* ``mobileclip2-s0``, ``mobileclip2-s2``, ``mobileclip2-b`` - Apple MobileCLIP2 via open_clip
  (research-only licence: measured for reference, not shippable)
* ``siglip2-b16`` - SigLIP 2 ViT-B/16 at 256 px via open_clip
* ``embeddinggemma2`` - Google EmbeddingGemma 2 (text + vision, audio off) via
  sentence-transformers, 768-d, 280 image tokens; ``embeddinggemma2-512`` is the same truncated to
  512-d; ``embeddinggemma2-t70`` uses the smallest image budget (70 tokens, about 5x faster)

Non-product models are downloaded from Hugging Face on first use (into the user's HF cache,
not ``models/``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

BATCH = 8
GEMMA_QUERY_PREFIX = "task: search result | query: "
MODELS = ("product", "b32-laion2b", "b32-datacomp-xl", "b32-256-datacomp", "mobileclip2-s0", "mobileclip2-s2", "mobileclip2-b", "siglip2-b16",
          "embeddinggemma2", "embeddinggemma2-512", "embeddinggemma2-t70")


# ---------------------------------------------------------------- pure helpers


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def first_relevant_rank(flags: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant Event, or None."""
    for i, f in enumerate(flags):
        if f:
            return i + 1
    return None


def summarize(ranks: Sequence[int | None], p5: Sequence[float]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit@1": 0.0, "hit@3": 0.0, "mrr": 0.0, "p@5": 0.0}
    return {"n": n,
            "hit@1": sum(1 for r in ranks if r == 1) / n,
            "hit@3": sum(1 for r in ranks if r is not None and r <= 3) / n,
            "mrr": sum(1.0 / r for r in ranks if r is not None) / n,
            "p@5": sum(p5) / n}


def load_query_file(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [q for q in data.get("queries") or [] if isinstance(q, dict) and q.get("text")]


# ---------------------------------------------------------------- images from the store


def stored_rows(db_path: Path) -> list[tuple[int, str, int, str, tuple[int, int, int, int] | None]]:
    """(vector_id, source video path, frame index, kind, box) of every stored vector."""
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = c.execute(
            "SELECT v.vector_id, vi.src_path, f.frame_idx, v.kind, v.x1, v.y1, v.x2, v.y2 "
            "FROM vectors v JOIN frames f USING (frame_id) JOIN videos vi USING (video_id) "
            "ORDER BY vi.video_id, f.frame_idx, v.vector_id").fetchall()
    finally:
        c.close()
    return [(vid, src, int(idx), kind, None if kind == "frame" else (int(x1), int(y1), int(x2), int(y2)))
            for vid, src, idx, kind, x1, y1, x2, y2 in rows]


def stored_images(db_path: Path, limit: int | None = None) -> tuple[list[int], list[np.ndarray]]:
    """Decode each stored frame once (sequentially, by frame index) and cut the stored crops."""
    import cv2

    rows = stored_rows(db_path)[:limit] if limit else stored_rows(db_path)
    ids: list[int] = []
    images: list[np.ndarray] = []
    by_video: dict[str, dict[int, list]] = {}
    for vid, src, idx, kind, box in rows:
        by_video.setdefault(src, {}).setdefault(idx, []).append((vid, box))
    for src, wanted in by_video.items():
        path = Path(src) if Path(src).is_absolute() else _ROOT / src
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"error: cannot open {path}")
        last = max(wanted)
        idx = 0
        while idx <= last:
            if idx not in wanted:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            for vid, box in wanted[idx]:
                ids.append(vid)
                images.append(img if box is None else img[box[1]:box[3], box[0]:box[2]].copy())
            idx += 1
        cap.release()
    order = {vid: i for i, (vid, *_r) in enumerate(rows)}
    pairs = sorted(zip(ids, images), key=lambda p: order[p[0]])
    if len(pairs) != len(rows):
        raise SystemExit(f"error: decoded {len(pairs)} of {len(rows)} stored images")
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ---------------------------------------------------------------- candidate encoders


class Candidate:
    """encode_images(list of BGR arrays) -> (n, d) and encode_texts(list of str) -> (n, d)."""

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


def _pil(img: np.ndarray):
    from PIL import Image

    return Image.fromarray(img[:, :, ::-1].copy())


class ProductCLIP(Candidate):
    def __init__(self) -> None:
        from nab_sentry.config import Config
        from nab_sentry.model_files import default_encoder_factory
        from nab_sentry.startup import require_models

        cfg = Config()
        self.enc = default_encoder_factory(cfg, require_models(cfg.models_dir))

    def encode_images(self, images):
        return self.enc.encode_images(list(images))

    def encode_texts(self, texts):
        return np.stack([self.enc.encode_text(t) for t in texts])


class OpenClipCandidate(Candidate):
    def __init__(self, arch: str, pretrained: str) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval()
        try:  # MobileCLIP / MobileOne blocks: fold the training branches for inference speed
            from timm.utils.model import reparameterize_model

            self.model = reparameterize_model(self.model)
        except Exception:  # noqa: BLE001 - not every architecture has re-parameterizable blocks
            pass
        self.tokenizer = open_clip.get_tokenizer(arch)

    def encode_images(self, images):
        with self.torch.inference_mode():
            x = self.torch.stack([self.preprocess(_pil(i)) for i in images])
            return l2n(self.model.encode_image(x).float().numpy())

    def encode_texts(self, texts):
        with self.torch.inference_mode():
            return l2n(self.model.encode_text(self.tokenizer(list(texts))).float().numpy())


class GemmaCandidate(Candidate):
    def __init__(self, dim: int | None = None, image_tokens: int | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = dim
        self.model = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                                         config_kwargs={"audio_config": None},
                                         model_kwargs={"torch_dtype": "float32"})
        if image_tokens is not None:  # 70, 140, 280 (default), 560 or 1120 soft tokens per image
            self.model[0].processor.image_processor.max_soft_tokens = image_tokens

    def _cut(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return l2n(x[:, :self.dim] if self.dim else x)

    def encode_images(self, images):
        inputs = [{"image": _pil(i)} for i in images]
        return self._cut(self.model.encode(inputs, batch_size=len(inputs), normalize_embeddings=True))

    def encode_texts(self, texts):
        return self._cut(self.model.encode([GEMMA_QUERY_PREFIX + t for t in texts], normalize_embeddings=True))


def make_candidate(name: str) -> Candidate:
    if name == "product":
        return ProductCLIP()
    if name == "b32-laion2b":
        local = _ROOT / "models" / "open_clip" / "ViT-B-32-laion2b_s34b_b79k" / "open_clip_pytorch_model.bin"
        return OpenClipCandidate("ViT-B-32", str(local) if local.is_file() else "laion2b_s34b_b79k")
    if name == "b32-datacomp-xl":
        local = _ROOT / "models" / "open_clip" / "ViT-B-32-datacomp_xl_s13b_b90k" / "open_clip_model.safetensors"
        return OpenClipCandidate("ViT-B-32", str(local) if local.is_file() else "datacomp_xl_s13b_b90k")
    if name == "b32-256-datacomp":
        return OpenClipCandidate("ViT-B-32-256", "datacomp_s34b_b86k")
    if name.startswith("mobileclip2-"):
        return OpenClipCandidate({"s0": "MobileCLIP2-S0", "s2": "MobileCLIP2-S2", "b": "MobileCLIP2-B"}[name.split("-")[1]],
                                 "dfndr2b")
    if name == "siglip2-b16":
        return OpenClipCandidate("ViT-B-16-SigLIP2-256", "webli")
    if name == "embeddinggemma2":
        return GemmaCandidate()
    if name == "embeddinggemma2-512":
        return GemmaCandidate(512)
    if name == "embeddinggemma2-t70":
        return GemmaCandidate(image_tokens=70)
    raise SystemExit(f"error: unknown model {name!r}; one of {', '.join(MODELS)}")


# ---------------------------------------------------------------- encode


def cmd_encode(args) -> int:
    import psutil

    queries = load_query_file(args.queries)
    ids, images = stored_images(args.db, args.limit)
    proc = psutil.Process()
    peak = {"rss": proc.memory_info().rss}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            stop.wait(0.2)
    threading.Thread(target=sample, daemon=True).start()
    t0 = time.perf_counter()
    cand = make_candidate(args.model)
    load_s = time.perf_counter() - t0
    cand.encode_images(images[:2])  # warm-up, not timed
    cand.encode_texts(["warm up"])
    vecs, batch_ms = [], []
    for i in range(0, len(images), BATCH):
        b = images[i:i + BATCH]
        t = time.perf_counter()
        vecs.append(np.asarray(cand.encode_images(b), dtype=np.float32))
        batch_ms.append((time.perf_counter() - t) * 1000 / len(b))
        if (i // BATCH) % 10 == 0:
            print(f"  {i + len(b)}/{len(images)} images", flush=True)
    t = time.perf_counter()
    qv = np.stack([np.asarray(cand.encode_texts([q["text"]]), dtype=np.float32)[0] for q in queries])
    query_ms = (time.perf_counter() - t) * 1000 / max(1, len(queries))
    stop.set()
    iv = np.concatenate(vecs)
    if not (np.all(np.isfinite(iv)) and np.all(np.isfinite(qv))):
        raise SystemExit("error: the model returned non-finite vectors")
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{args.model}.npz", ids=np.asarray(ids, np.int64), images=iv, queries=qv,
             query_texts=np.asarray([q["text"] for q in queries]))
    info = {"model": args.model, "dim": int(iv.shape[1]), "images": len(ids), "queries": len(queries),
            "load_s": round(load_s, 1), "ms_per_image_b8_median": round(float(np.median(batch_ms)), 1),
            "ms_per_image_b8_p95": round(float(np.percentile(batch_ms, 95)), 1),
            "ms_per_query": round(query_ms, 1), "peak_rss_mb": round(peak["rss"] / 2**20)}
    (args.out / f"{args.model}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))
    return 0


# ---------------------------------------------------------------- score


class _Fixed:
    """Stands in for the text encoder: the query vectors were computed by ``encode``."""

    def __init__(self, texts: Sequence[str], vecs: np.ndarray) -> None:
        self.map = {t: v for t, v in zip(texts, vecs)}

    def encode_text(self, query: str) -> np.ndarray:
        return self.map[query]

    def encode_images(self, images):
        raise NotImplementedError


def score_model(name: str, out: Path, db_copy: Path, queries: list[dict]) -> dict:
    from nab_sentry.config import Config
    import importlib.util

    from nab_sentry.evaluation import load_queries_file, precision_at_5
    from nab_sentry.search.engine import SearchEngine, SearchFilter
    from nab_sentry.store.db import MetadataStore
    from nab_sentry.store.vector_index import VectorIndex

    spec = importlib.util.spec_from_file_location("evaluate_script", _ROOT / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)  # its relevance_flags: the same scoring as the V1 evaluator
    data = np.load(out / f"{name}.npz")
    texts = [str(t) for t in data["query_texts"]]
    if texts != [q["text"] for q in queries]:
        raise SystemExit(f"error: {name}.npz was encoded with other queries; run encode again")
    index = VectorIndex(dim=int(data["images"].shape[1]))
    index.add(data["ids"], data["images"])
    cfg = Config()
    db = MetadataStore(db_copy)
    try:
        engine = SearchEngine(cfg, db, index, _Fixed(texts, data["queries"]))
        engine.check_ready()
        evq, errors = load_queries_file(db_copy.parent / "queries.yaml")
        if errors:
            raise SystemExit(f"error: queries file: {errors[0]}")
        ranks, p5 = [], []
        for q, vec in zip(evq, data["queries"]):
            events = engine.search_vector(np.asarray(vec, np.float32), SearchFilter(), q.text)
            flags = evaluate.relevance_flags(events, db, q)
            ranks.append(first_relevant_rank(flags))
            p5.append(precision_at_5(flags))
    finally:
        db.close()
    return {"ranks": ranks, **summarize(ranks, p5)}


def cmd_score(args) -> int:
    names = args.models or sorted(p.stem for p in args.out.glob("*.npz"))
    if not names:
        print(f"error: nothing encoded in {args.out}; run encode first", file=sys.stderr)
        return 1
    queries = load_query_file(args.queries)
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "store.db"
        src = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        dst = sqlite3.connect(db_copy)
        src.backup(dst)
        src.close()
        dst.close()
        shutil.copy2(args.queries, Path(tmp) / "queries.yaml")
        for name in names:
            results[name] = score_model(name, args.out, db_copy, queries)
            info_path = args.out / f"{name}.json"
            results[name]["info"] = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    report = render(results, queries)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def render(results: dict[str, dict], queries: list[dict]) -> str:
    names = list(results)
    lines = ["| Model | dim | hit@1 | hit@3 | MRR | p@5 | ms/image (b8) | ms/query | load s | peak RSS MB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r, i = results[n], results[n].get("info", {})
        lines.append(f"| {n} | {i.get('dim', '')} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['mrr']:.3f} | "
                     f"{r['p@5']:.2f} | {i.get('ms_per_image_b8_median', '')} | {i.get('ms_per_query', '')} | "
                     f"{i.get('load_s', '')} | {i.get('peak_rss_mb', '')} |")
    lines += ["", "Rank of the first relevant Event per query (- = not in the results):", "",
              "| Query | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, q in enumerate(queries):
        cells = [str(results[n]["ranks"][k]) if results[n]["ranks"][k] is not None else "-" for n in names]
        lines.append(f"| {q['text']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "score"):
        s = sub.add_parser(name)
        if name == "encode":
            s.add_argument("model", choices=MODELS)
            s.add_argument("--limit", type=int, default=None, help="only the first N stored images (smoke test)")
        else:
            s.add_argument("models", nargs="*")
        s.add_argument("--db", type=Path, default=_ROOT / "data" / "nab_sentry.db")
        s.add_argument("--queries", type=Path, default=_ROOT / "eval" / "queries_compilation.yaml")
        s.add_argument("--out", type=Path, default=_ROOT / "data" / "bench" / "embed")
    args = p.parse_args(argv)
    return cmd_encode(args) if args.cmd == "encode" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
