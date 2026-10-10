"""Deterministic SOP retrieval for Jadal QA.

What this module does
--------------------
Loads every `*.md` SOP in a directory, splits each file into `## Step ...`
passages and retrieves the single best-matching passage for a query.

Determinism rules baked in here
--------------------------------
- passages are sorted by `passage_id`; every ordering key ends in `passage_id`,
  so a tie can never be broken by dict order, hash order or call order;
- scores are rounded to 6 decimals before they are compared, so float noise
  cannot reorder results;
- no network call happens at import time or inside `search`.

Retrieval backends
------------------
`Embedder` is the fixed interface for a dense retriever:

    class MyEmbedder:
        dim = 768
        def embed(self, texts: list[str]) -> list[list[float]]: ...

Pass an instance as `SOPIndex(..., embedder=...)` or per call
(`search(query, embedder=...)`); passages are embedded once per (lang, embedder)
and cached. When no embedder is given the index falls back to a BM25 keyword
score, which needs no model, no GPU and no network — that is the path the unit
tests and the offline dashboard use. A failure in a supplied embedder is *not*
swallowed: it propagates, because a silently degraded retriever is worse than a
loud error.

What search returns
-------------------
`search()` returns dicts whose `text` is the passage **verbatim** — the exact
slice of the SOP file, never a paraphrase, summary or generated sentence. The
SOP file is the only source of text in the result dict.

NOT DONE
--------
- no reranker and no cross-encoder: hybrid score is a plain cosine (embedder
  path) or BM25 (fallback path), never a blend of the two;
- `HashEmbedder` is a feature-hashing smoke-test embedder, not a semantic
  model; `SentenceTransformersEmbedder` is untested in this environment
  (sentence-transformers is not installed);
- no relevance evaluation exists: nothing here reports an accuracy number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

SCORE_DECIMALS = 6
DEFAULT_TOP_K = 5


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase word tokens, with Indic combining marks kept inside the word.

    `[^\\W_]` alone drops Devanagari/Kannada/Tamil/Telugu matras (they are
    combining marks, not `\\w`), which would split "कैमरा" into single
    letters and make every Indic string collide. Adding the U+0900-U+0DFF
    block keeps each Indic script's words whole. The split is still a pure
    function of the input bytes, so results are stable across runs.
    """
    return tuple(_WORD_RE.findall(text.lower()))


_WORD_RE = re.compile(r"(?:[^\W_]|[\u0900-\u0dff])+", re.UNICODE)
_HEADER_RE = re.compile(r"\A\s*<!--(?P<body>.*?)-->", re.DOTALL)
_STEP_RE = re.compile(r"step\s*[-–—:]?\s*(\d+)", re.IGNORECASE)

def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} != {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _is_finite_vector(vec: Sequence[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vec)


@dataclass(frozen=True)
class Passage:
    """One retrieval unit: a single `## Step` section of a SOP markdown file."""

    passage_id: str
    doc_id: str
    sop_id: str
    lang: str | None
    title: str
    text: str
    step: int | None = None
    tokens: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.tokens:
            object.__setattr__(self, "tokens", tokenize(self.text))

    def to_dict(self) -> dict:
        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "sop_id": self.sop_id,
            "lang": self.lang,
            "title": self.title,
            "step": self.step,
            "text": self.text,
        }


@dataclass(frozen=True)
class SopDoc:
    """Parsed header of one SOP markdown file."""

    doc_id: str
    sop_id: str
    lang: str | None
    title: str
    status: str | None
    passages: tuple[Passage, ...]


def parse_sop_markdown(text: str, doc_id: str) -> SopDoc:
    """Parse one SOP markdown file into passages.

    The file may start with an HTML comment header block carrying
    `key: value` lines (`sop_id`, `lang`, `status`). Sections start at `## `
    headings; each section becomes one passage whose `text` is the verbatim
    slice of the file (leading/trailing blank lines trimmed). A file with no
    `## ` sections becomes a single passage holding the whole body.
    """
    header: dict[str, str] = {}
    body = text
    match = _HEADER_RE.match(text)
    if match:
        for line in match.group("body").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip().lower()] = value.strip()
        body = text[match.end():]

    doc_title = doc_id
    for line in body.splitlines():
        if line.startswith("# ") and line[2:].strip():
            doc_title = line[2:].strip()
            break

    sections: list[list[str]] = []
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append([line])
        elif sections:
            sections[-1].append(line)

    passages: list[Passage] = []
    if sections:
        for number, section_lines in enumerate(sections, start=1):
            section_text = "\n".join(section_lines).strip("\n")
            heading = section_lines[0][3:].strip()
            step_match = _STEP_RE.search(heading)
            step = int(step_match.group(1)) if step_match else None
            passages.append(
                Passage(
                    passage_id=f"{doc_id}/{number:03d}",
                    doc_id=doc_id,
                    sop_id=header.get("sop_id", doc_id),
                    lang=header.get("lang") or None,
                    title=heading or doc_title,
                    text=section_text,
                    step=step,
                )
            )
    else:
        whole = body.strip("\n")
        if whole.strip():
            passages.append(
                Passage(
                    passage_id=f"{doc_id}/001",
                    doc_id=doc_id,
                    sop_id=header.get("sop_id", doc_id),
                    lang=header.get("lang") or None,
                    title=doc_title,
                    text=whole,
                    step=None,
                )
            )

    return SopDoc(
        doc_id=doc_id,
        sop_id=header.get("sop_id", doc_id),
        lang=header.get("lang") or None,
        title=doc_title,
        status=header.get("status"),
        passages=tuple(passages),
    )


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75). Deterministic, dependency-free."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.docs: list[list[str]] = [list(d) for d in docs]
        self.n = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.term_freq: list[Counter[str]] = [Counter(d) for d in self.docs]
        doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))
        self.doc_freq = doc_freq

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: Sequence[str]) -> list[float]:
        if self.n == 0:
            return []
        avgdl = self.avgdl or 1.0
        out: list[float] = []
        for tf, dl in zip(self.term_freq, self.doc_len):
            score = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl or 1) / avgdl)
                score += self.idf(term) * freq * (self.k1 + 1.0) / denom
            out.append(score)
        return out


@runtime_checkable
class Embedder(Protocol):
    """Fixed interface for a dense retriever (the only contract callers must
    meet). `embed` takes texts and returns one row per text, each a finite
    float vector of length `dim`. It must not sample or randomise."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Feature-hashing embedder: deterministic, offline, no vocabulary.

    It exists so the embedding code path can be exercised without a model or a
    GPU. It is not semantic and is not what production should use.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:6], "big") % self.dim
            sign = 1.0 if digest[6] & 1 == 0 else -1.0
            vec[bucket] += sign
        return _l2_normalise(vec)


class SentenceTransformersEmbedder:
    """Dense embedder wrapper for sentence-transformers (optional dependency).

    The import happens in `__init__`, so importing this module never pulls a
    model or opens a socket. Untested here: sentence-transformers is not
    installed in the Jadal env.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: no import-time network

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        rows = self._model.encode(
            list(texts), convert_to_numpy=False, normalize_embeddings=False, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in rows]


def make_embedder(kind: str, dim: int = 512, model_name: str | None = None) -> Embedder | None:
    """`"none"` -> None (BM25 fallback); `"hash"` -> HashEmbedder;
    `"sentence-transformers"` -> SentenceTransformersEmbedder."""
    if kind == "none":
        return None
    if kind == "hash":
        return HashEmbedder(dim=dim)
    if kind == "sentence-transformers":
        if not model_name:
            raise ValueError("sentence-transformers needs --model-name")
        return SentenceTransformersEmbedder(model_name)
    raise ValueError(f"unknown embedder kind: {kind!r}")


class SOPIndex:
    """Deterministic retrieval over SOP markdown passages.

    `search(query, lang=None, top_k=5, embedder=None)`:
      - `lang` filters on the header `lang` field (exact, case-insensitive);
      - with no embedder the score is normalised BM25, which is the offline
        fallback; with one it is the cosine between query and passage vectors;
      - results are ordered by `(-score, -bm25, passage_id)` and only strictly
        positive scores are returned;
      - `text` is verbatim passage text.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        embedder: Embedder | None = None,
        docs: Sequence[SopDoc] = (),
    ) -> None:
        self.passages: tuple[Passage, ...] = tuple(
            sorted(passages, key=lambda p: p.passage_id)
        )
        self.embedder = embedder
        self.docs: tuple[SopDoc, ...] = tuple(sorted(docs, key=lambda d: d.doc_id))
        self._by_id: dict[str, Passage] = {p.passage_id: p for p in self.passages}
        self._bm25_cache: dict[str | None, BM25] = {}
        self._vector_cache: dict[tuple[str | None, int, int], tuple[Embedder, list[list[float]]]] = {}

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "SOPIndex":
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"SOP directory not found: {directory}")
        passages: list[Passage] = []
        docs: list[SopDoc] = []
        for path in sorted(directory.glob("*.md")):
            doc = parse_sop_markdown(path.read_text(encoding="utf-8"), doc_id=path.stem)
            docs.append(doc)
            passages.extend(doc.passages)
        return cls(passages, embedder=embedder, docs=docs)

    def __len__(self) -> int:
        return len(self.passages)

    def __repr__(self) -> str:
        return f"SOPIndex(passages={len(self.passages)}, embedder={type(self.embedder).__name__})"

    def languages(self) -> list[str]:
        return sorted({p.lang for p in self.passages if p.lang})

    def passage(self, passage_id: str) -> Passage | None:
        return self._by_id.get(passage_id)

    def _candidates(self, lang: str | None) -> list[Passage]:
        if lang is None:
            return list(self.passages)
        want = lang.lower()
        return [p for p in self.passages if (p.lang or "").lower() == want]

    def _bm25_for(self, lang: str | None) -> BM25:
        key = lang.lower() if lang else None
        if key not in self._bm25_cache:
            docs = self._candidates(key)
            self._bm25_cache[key] = BM25([p.tokens for p in docs])
        return self._bm25_cache[key]

    def _vectors_for(self, lang: str | None, embedder: Embedder) -> list[list[float]]:
        docs = self._candidates(lang)
        # The cache holds a strong reference to the embedder alongside its
        # vectors, so id(embedder) cannot be recycled while the entry lives.
        key = (lang.lower() if lang else None, id(embedder), embedder.dim)
        cached = self._vector_cache.get(key)
        if cached is None:
            vectors = embedder.embed([p.text for p in docs])
            if len(vectors) != len(docs):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(docs)} passages"
                )
            for vector in vectors:
                if len(vector) != embedder.dim or not _is_finite_vector(vector):
                    raise ValueError(
                        f"embedder returned a bad vector (dim {len(vector)}, expected {embedder.dim})"
                    )
            cached = (embedder, [list(v) for v in vectors])
            self._vector_cache[key] = cached
        return cached[1]

    def search(
        self,
        query: str,
        lang: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        embedder: Embedder | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidates = self._candidates(lang)
        if not candidates:
            return []

        bm25_scores = self._bm25_for(lang).scores(query_tokens)
        peak = max((abs(s) for s in bm25_scores), default=0.0)
        bm25_norm = [
            round(s / peak, SCORE_DECIMALS) if peak > 0.0 else 0.0 for s in bm25_scores
        ]

        effective = embedder if embedder is not None else self.embedder
        if effective is not None:
            query_vectors = effective.embed([query])
            if len(query_vectors) != 1:
                raise ValueError("embedder must return one vector per query")
            query_vector = query_vectors[0]
            if len(query_vector) != effective.dim or not _is_finite_vector(query_vector):
                raise ValueError(
                    f"embedder returned a bad query vector (dim {len(query_vector)},"
                    f" expected {effective.dim})"
                )
            passage_vectors = self._vectors_for(lang, effective)
            scores = [
                round(_cosine(query_vector, vec), SCORE_DECIMALS) for vec in passage_vectors
            ]
        else:
            scores = bm25_norm

        order = sorted(
            range(len(candidates)),
            key=lambda i: (-scores[i], -bm25_norm[i], candidates[i].passage_id),
        )

        results: list[dict] = []
        for index in order:
            if scores[index] <= 0.0:
                continue
            passage = candidates[index]
            result = {
                "rank": len(results) + 1,
                "passage_id": passage.passage_id,
                "doc_id": passage.doc_id,
                "sop_id": passage.sop_id,
                "lang": passage.lang,
                "title": passage.title,
                "step": passage.step,
                "score": scores[index],
                "bm25": bm25_norm[index],
                "text": passage.text,
            }
            results.append(result)
            if len(results) >= top_k:
                break
        return results


def default_sop_dir() -> Path:
    return Path(__file__).resolve().parent


def load_sop_index(directory: str | Path | None = None, embedder: Embedder | None = None) -> SOPIndex:
    return SOPIndex.load(directory if directory is not None else default_sop_dir(), embedder=embedder)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic SOP retrieval.")
    parser.add_argument("--dir", default=str(default_sop_dir()), help="directory of *.md SOPs")
    parser.add_argument("--q", default=None, help="query text")
    parser.add_argument("--lang", default=None, help="filter by SOP header lang, e.g. hi-IN")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedder", default="none", choices=["none", "hash", "sentence-transformers"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--list", action="store_true", help="list passages instead of searching")
    args = parser.parse_args(argv)

    index = load_sop_index(args.dir, embedder=make_embedder(args.embedder, model_name=args.model_name))
    if args.list:
        payload = [p.to_dict() for p in index.passages]
    elif args.q:
        payload = index.search(args.q, lang=args.lang, top_k=args.top_k)
    else:
        payload = {"passages": len(index.passages), "docs": [d.sop_id for d in index.docs]}
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
