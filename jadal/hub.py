"""Deterministic video-hub index (GitHub issue #27).

Storage: float16 matrix (N, D) + ids + metadata, saved as
``index.npz`` + ``meta.jsonl`` under ``data_runtime/hub/`` (git-ignored).
Build order is canonical (sorted by segment id); ``index_hash`` is the
sha256 over the float16 matrix bytes plus canonical metadata JSON.
Search is exact cosine via blocked float32 matmul; ties break by
segment_id. Every search returns a ``results_hash`` (sha256 of canonical
JSON of ids + scores rounded to 4 decimals) so a search is re-verifiable.

Metadata per segment: clip, start_s, end_s, worker_id, site_id, verdict,
reasons (list), step (optional).

NOT DONE:
- No ANN index (exact search only; ANN needs a MEASURED latency reason —
  see docs/reports/hub.md).
- No incremental re-sharding; rebuild from ``add()`` + ``save()``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _round4(x: float) -> float:
    return round(float(x), 4)


META_FIELDS = ("clip", "start_s", "end_s", "worker_id", "site_id",
               "verdict", "reasons", "step", "path")


class Index:
    """Exact-search hub index. Hardware-free (embed_fn injected)."""

    def __init__(self, dim: int = 256,
                 embed_fn: Callable[[str], np.ndarray] | None = None) -> None:
        self.dim = dim
        self._embed_fn = embed_fn
        self._vecs: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}

    # -- build ---------------------------------------------------------
    def add(self, segment_id: str, vector: np.ndarray, meta: dict) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"vector dim {v.shape[0]} != index dim {self.dim}")
        n = float(np.linalg.norm(v))
        if n == 0:
            raise ValueError("zero vector")
        self._vecs[segment_id] = (v / n).astype(np.float16)
        m = {k: meta.get(k) for k in META_FIELDS}
        if m["reasons"] is None:
            m["reasons"] = []
        else:
            m["reasons"] = list(m["reasons"])
        self._meta[segment_id] = m

    def __len__(self) -> int:
        return len(self._vecs)

    def ids_sorted(self) -> list[str]:
        return sorted(self._vecs.keys())

    def _ordered_matrix(self) -> tuple[list[str], np.ndarray]:
        ids = self.ids_sorted()
        if not ids:
            return ids, np.zeros((0, self.dim), dtype=np.float16)
        mat = np.stack([self._vecs[i] for i in ids]).astype(np.float16)
        return ids, mat

    def index_hash(self) -> str:
        ids, mat = self._ordered_matrix()
        h = hashlib.sha256()
        h.update(mat.tobytes())
        h.update(canonical_json([{"id": i, "meta": self._meta[i]} for i in ids]))
        return h.hexdigest()

    # -- persistence ----------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        ids, mat = self._ordered_matrix()
        np.savez_compressed(d / "index.npz",
                            ids=np.array(ids), matrix=mat, dim=self.dim)
        with open(d / "meta.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "meta": self._meta[i]},
                                   sort_keys=True) + "\n")
        with open(d / "index_hash.txt", "w") as f:
            f.write(self.index_hash())
        return d

    @classmethod
    def load(cls, directory: str | os.PathLike,
             embed_fn: Callable | None = None) -> "Index":
        d = Path(directory)
        z = np.load(d / "index.npz", allow_pickle=True)
        ids = [str(i) for i in z["ids"].tolist()]
        mat = np.asarray(z["matrix"], dtype=np.float16)
        idx = cls(dim=int(z["dim"]), embed_fn=embed_fn)
        metas: dict[str, dict] = {}
        with open(d / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    metas[rec["id"]] = rec["meta"]
        for i, row in zip(ids, mat):
            idx._vecs[i] = row.astype(np.float16)
            idx._meta[i] = metas[i]
        return idx

    # -- filters ---------------------------------------------------------
    @staticmethod
    def _passes(meta: dict, f: dict) -> bool:
        if not f:
            return True
        if f.get("site") is not None and meta.get("site_id") != f["site"]:
            return False
        if f.get("worker") is not None and meta.get("worker_id") != f["worker"]:
            return False
        if f.get("verdict") is not None and meta.get("verdict") != f["verdict"]:
            return False
        if f.get("reason") is not None and f["reason"] not in (meta.get("reasons") or []):
            return False
        if f.get("clip") is not None and meta.get("clip") != f["clip"]:
            return False
        if f.get("step") is not None and meta.get("step") != f["step"]:
            return False
        t0, t1 = f.get("t0"), f.get("t1")
        if t0 is not None or t1 is not None:
            s, e = meta.get("start_s"), meta.get("end_s")
            if s is None or e is None:
                return False
            if t0 is not None and e <= t0:
                return False
            if t1 is not None and s >= t1:
                return False
        return True

    # -- search ----------------------------------------------------------
    def search_by_vector(self, qvec: np.ndarray, k: int = 5,
                         filters: dict | None = None) -> dict:
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError("query dim mismatch")
        n = float(np.linalg.norm(q))
        if n == 0:
            raise ValueError("zero query vector")
        q = q / n
        f = filters or {}
        ids, mat = self._ordered_matrix()
        if not ids:
            return {"hits": [], "results_hash": hashlib.sha256(
                canonical_json([])).hexdigest()}
        scores = mat.astype(np.float32) @ q
        scored = [(float(scores[j]), ids[j]) for j in range(len(ids))
                  if self._passes(self._meta[ids[j]], f)]
        # Ties broken by segment_id (score rounded to 9 decimals for grouping).
        scored.sort(key=lambda t: (-round(t[0], 9), t[1]))
        hits = []
        for s, i in scored[:max(k, 0)]:
            m = self._meta[i]
            hits.append({"segment_id": i, "score": _round4(s),
                         "clip": m.get("clip"), "start_s": m.get("start_s"),
                         "end_s": m.get("end_s"), "worker_id": m.get("worker_id"),
                         "site_id": m.get("site_id"), "verdict": m.get("verdict"),
                         "reasons": m.get("reasons"), "step": m.get("step")})
        rh = hashlib.sha256(canonical_json(
            [{"id": h["segment_id"], "score": h["score"]} for h in hits])).hexdigest()
        return {"hits": hits, "results_hash": rh}

    def search_text(self, query: str, k: int = 5,
                    filters: dict | None = None) -> dict:
        if self._embed_fn is None:
            raise RuntimeError("no embed_fn: construct Index(embed_fn=...)")
        return self.search_by_vector(np.asarray(self._embed_fn(query)), k, filters)

    def search_similar(self, segment_id: str | None = None,
                       vector: np.ndarray | None = None, k: int = 5,
                       filters: dict | None = None) -> dict:
        """Find segments similar to an example. The query segment itself is
        excluded from hits (documented in docs/reports/hub.md)."""
        if vector is None:
            if segment_id is None or segment_id not in self._vecs:
                raise KeyError(f"unknown segment_id: {segment_id}")
            vector = self._vecs[segment_id].astype(np.float32)
        res = self.search_by_vector(np.asarray(vector), k=k + 1, filters=filters)
        if segment_id is not None:
            res["hits"] = [h for h in res["hits"]
                           if h["segment_id"] != segment_id][:k]
            res["results_hash"] = hashlib.sha256(canonical_json(
                [{"id": h["segment_id"], "score": h["score"]}
                 for h in res["hits"]])).hexdigest()
        else:
            res["hits"] = res["hits"][:k]
        return res

    # -- dedup ------------------------------------------------------------
    def dedup_clusters(self, threshold: float = 0.95,
                       block: int = 2048) -> dict:
        """Connected components over cosine >= threshold (exact, blocked).

        Returns {"clusters": [{"representative", "members", "size"}],
        "n_segments", "n_clusters", "dup_fraction"}. Representative = medoid
        (max in-cluster similarity sum; ties by id). No cap on cluster size.
        """
        ids, mat = self._ordered_matrix()
        n = len(ids)
        if n == 0:
            return {"clusters": [], "n_segments": 0, "n_clusters": 0,
                    "dup_fraction": 0.0}
        f32 = mat.astype(np.float32)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i0 in range(0, n, block):
            a = f32[i0:i0 + block]
            for j0 in range(0, n, block):
                sims = a @ f32[j0:j0 + block].T
                ii, jj = np.nonzero(sims >= threshold)
                for di, dj in zip(ii.tolist(), jj.tolist()):
                    gi, gj = i0 + int(di), j0 + int(dj)
                    if gj > gi:
                        union(gi, gj)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        clusters = []
        for members in groups.values():
            sub = f32[members] @ f32[members].T
            sums = sub.sum(axis=1)
            best = min(range(len(members)),
                       key=lambda r: (-round(float(sums[r]), 9), ids[members[r]]))
            rep = ids[members[best]]
            mem_ids = sorted(ids[m] for m in members)
            clusters.append({"representative": rep, "members": mem_ids,
                             "size": len(mem_ids)})
        clusters.sort(key=lambda c: c["representative"])
        dup = (n - len(clusters)) / n if n else 0.0
        return {"clusters": clusters, "n_segments": n,
                "n_clusters": len(clusters), "dup_fraction": round(dup, 4)}
