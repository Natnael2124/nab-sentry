"""One embedder per store (V2 Step 4).

Vectors of two embedders live in different spaces: mixed in one index, every search would be
silently wrong. The store records the embedder that made its vectors (``schema_meta.embed_model``);
:func:`check_embedder` compares it with the encoder in use before ingest adds vectors and before
search reads them.

* same embedder: fine;
* the store has no vectors yet: it takes the encoder's embedder (a new install, or after
  retention emptied it);
* otherwise: :class:`EmbedderMismatch`, naming both and the command that converts the store
  (``scripts/reembed.py``).

Encoders without a ``model_id`` (test fakes) are not checked.
"""

from __future__ import annotations

from typing import Any

REEMBED_HINT = ("stop run_live.ps1, then run: venv\\Scripts\\python.exe scripts\\reembed.py "
                "(re-encodes the stored footage with the new embedder; about as long as embedding it took)")


class EmbedderMismatch(RuntimeError):
    def __init__(self, stored: str | None, wanted: str) -> None:
        self.stored = stored
        self.wanted = wanted
        super().__init__(f"the stored vectors were made by the embedder {stored or 'unknown'}, but "
                         f"{wanted} is configured; {REEMBED_HINT}")


def check_embedder(db: Any, encoder: Any) -> None:
    wanted = getattr(encoder, "model_id", None)
    if not wanted:
        return
    stored = db.embed_model()
    if stored == wanted:
        return
    if db.vector_ids().size == 0:
        db.set_embed_model(wanted, int(getattr(encoder, "dim", 512)))
        return
    raise EmbedderMismatch(stored, wanted)
