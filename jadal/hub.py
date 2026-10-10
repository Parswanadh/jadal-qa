"""Deterministic video-hub index (GitHub issue #27).

Storage: float16 matrix (N, D) + ids + metadata, saved as
``index.npz`` (plain unicode + float16 arrays, NO pickle) + ``meta.jsonl``
under ``data_runtime/hub/`` (git-ignored). ``load()`` recomputes and verifies
``index_hash`` (mismatch => error).

Performance layout: the canonical ordered ids and the CONTIGUOUS float16 /
float32 matrices are built ONCE (rebuilt only when ``add()`` dirties them) —
``search_*`` and ``index_hash`` never re-sort or re-stack per query.
Search is exact cosine over the cached float32 matrix with a vectorised
filter mask and an exact top-k (threshold + id tie-break, no full sort).
``dedup_clusters`` refuses N above ``max_clusters_n`` (service protection;
the API maps this to HTTP 413).

Canonical build order is sorted by segment id; ``index_hash`` is the sha256
over the float16 matrix bytes plus canonical metadata JSON. Every search
returns a ``results_hash`` (sha256 of canonical JSON of ids + scores rounded
to 4 decimals) so a search is re-verifiable.

Metadata per segment: clip, start_s, end_s, worker_id, site_id, verdict,
reasons (list), step (optional), path.

NOT DONE:
- No ANN index (exact search only; measured latencies in docs/reports/hub.md
  state the N each claim holds for).
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


class ClusterLimitError(ValueError):
    """Raised when dedup_clusters is asked to cluster more than
    max_clusters_n segments (protects the service from O(N^2) blowups)."""


class Index:
    """Exact-search hub index. Hardware-free (embed_fn injected)."""

    def __init__(self, dim: int = 256,
                 embed_fn: Callable[[str], np.ndarray] | None = None,
                 max_clusters_n: int = 20000) -> None:
        self.dim = dim
        self._embed_fn = embed_fn
        self.max_clusters_n = max_clusters_n
        self._vecs: dict[str, np.ndarray] = {}
        self._meta: dict[str, dict] = {}
        # Cached canonical layout (rebuilt only when dirty).
        self._dirty = True
        self._ids: list[str] = []
        self._mat16 = np.zeros((0, dim), dtype=np.float16)
        self._mat32 = np.zeros((0, dim), dtype=np.float32)

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
        self._dirty = True

    def get_meta(self, segment_id: str) -> dict | None:
        """Public metadata accessor (None when unknown)."""
        m = self._meta.get(segment_id)
        return dict(m) if m is not None else None

    def get_vector(self, segment_id: str) -> np.ndarray:
        """Stored unit vector as float32 (KeyError when unknown)."""
        self._ensure_ordered()
        return self._mat32[self._ids.index(segment_id)].copy()

    def __len__(self) -> int:
        return len(self._vecs)

    def _ensure_ordered(self) -> None:
        """Rebuild cached canonical arrays iff dirty. All mutation of
        _vecs/_meta must go through add() (which sets the flag)."""
        if not self._dirty and len(self._ids) == len(self._vecs):
            return
        ids = sorted(self._vecs.keys())
        if ids:
            mat = np.stack([self._vecs[i] for i in ids]).astype(np.float16)
        else:
            mat = np.zeros((0, self.dim), dtype=np.float16)
        self._ids = ids
        self._mat16 = np.ascontiguousarray(mat, dtype=np.float16)
        self._mat32 = np.ascontiguousarray(mat, dtype=np.float32)
        self._dirty = False

    def ids_sorted(self) -> list[str]:
        self._ensure_ordered()
        return list(self._ids)

    def index_hash(self) -> str:
        self._ensure_ordered()
        h = hashlib.sha256()
        h.update(self._mat16.tobytes())
        h.update(canonical_json(
            [{"id": i, "meta": self._meta[i]} for i in self._ids]))
        return h.hexdigest()

    # -- persistence ----------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        self._ensure_ordered()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "index.npz",
                            ids=np.array(self._ids, dtype=np.str_),
                            matrix=self._mat16,
                            dim=np.int64(self.dim))
        with open(d / "meta.jsonl", "w", encoding="utf-8") as f:
            for i in self._ids:
                f.write(json.dumps({"id": i, "meta": self._meta[i]},
                                   sort_keys=True) + "\n")
        with open(d / "index_hash.txt", "w") as f:
            f.write(self.index_hash())
        return d

    @classmethod
    def load(cls, directory: str | os.PathLike,
             embed_fn: Callable | None = None,
             max_clusters_n: int = 20000) -> "Index":
        d = Path(directory)
        # allow_pickle=False: ids are plain unicode, matrix is float16 —
        # an index shared between hosts must never be an arbitrary-code path.
        z = np.load(d / "index.npz", allow_pickle=False)
        ids = [str(i) for i in z["ids"].tolist()]
        mat = np.ascontiguousarray(z["matrix"], dtype=np.float16)
        dim = int(z["dim"])
        metas: dict[str, dict] = {}
        with open(d / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    metas[rec["id"]] = rec["meta"]
        if set(ids) != set(metas.keys()):
            raise ValueError("index.npz ids do not match meta.jsonl ids")
        idx = cls(dim=dim, embed_fn=embed_fn, max_clusters_n=max_clusters_n)
        for i, row in zip(ids, mat):
            idx._vecs[i] = np.ascontiguousarray(row, dtype=np.float16)
            idx._meta[i] = metas[i]
        idx._dirty = True
        idx._ensure_ordered()
        expected = (d / "index_hash.txt").read_text().strip()
        actual = idx.index_hash()
        if expected != actual:
            raise ValueError(
                f"index integrity check failed in {d}: stored hash "
                f"{expected[:16]}... != recomputed {actual[:16]}...")
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

    def _filter_mask(self, f: dict) -> np.ndarray:
        """Boolean mask over the canonical order. Literal equality/overlap
        only — filter values are never interpreted (no injection)."""
        if not f:
            return np.ones(len(self._ids), dtype=bool)
        return np.fromiter((self._passes(self._meta[i], f) for i in self._ids),
                           dtype=bool, count=len(self._ids))

    # -- search ----------------------------------------------------------
    def _matvec(self, mat: np.ndarray, q: np.ndarray,
                use_torch: bool = False) -> np.ndarray:
        if use_torch:
            import torch
            if torch.cuda.is_available():
                t = torch.from_numpy(mat).to("cuda")
                qq = torch.from_numpy(q).to("cuda")
                out = (t @ qq).to("cpu").numpy()
                torch.cuda.synchronize()
                return np.asarray(out, dtype=np.float32)
            t = torch.from_numpy(mat)
            return np.asarray(t @ torch.from_numpy(q), dtype=np.float32)
        return mat @ q

    def search_by_vector(self, qvec: np.ndarray, k: int = 5,
                         filters: dict | None = None,
                         use_torch: bool = False) -> dict:
        """Exact cosine top-k over the cached matrix.

        Deterministic: scores rounded to 9 decimals, order is
        (-score, segment_id). Boundary ties at rank k resolve by id, so the
        result is identical for the numpy and torch backends.
        """
        self._ensure_ordered()
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError("query dim mismatch")
        n = float(np.linalg.norm(q))
        if n == 0:
            raise ValueError("zero query vector")
        q = np.ascontiguousarray(q / n, dtype=np.float32)
        f = filters or {}
        if not self._ids:
            return {"hits": [], "results_hash": hashlib.sha256(
                canonical_json([])).hexdigest()}
        mask = self._filter_mask(f)
        pos = np.nonzero(mask)[0]
        if pos.size == 0:
            return {"hits": [], "results_hash": hashlib.sha256(
                canonical_json([])).hexdigest()}
        scores = self._matvec(self._mat32[pos], q, use_torch=use_torch)
        rscores = np.round(scores.astype(np.float64), 9)
        m = pos.size
        kk = min(max(k, 0), m)
        if m <= kk or kk == 0:
            order = sorted(range(m),
                           key=lambda r: (-rscores[r], self._ids[pos[r]]))
            sel = order[:kk]
        else:
            kth = -np.partition(-rscores, kk - 1)[kk - 1]
            gt = [r for r in range(m) if rscores[r] > kth]
            need = kk - len(gt)
            eq = sorted((r for r in range(m) if rscores[r] == kth),
                        key=lambda r: self._ids[pos[r]])[:need]
            sel = sorted(gt + eq,
                         key=lambda r: (-rscores[r], self._ids[pos[r]]))
        hits = []
        for r in sel:
            i = self._ids[pos[r]]
            md = self._meta[i]
            hits.append({"segment_id": i, "score": _round4(float(scores[r])),
                         "clip": md.get("clip"), "start_s": md.get("start_s"),
                         "end_s": md.get("end_s"), "worker_id": md.get("worker_id"),
                         "site_id": md.get("site_id"), "verdict": md.get("verdict"),
                         "reasons": md.get("reasons"), "step": md.get("step")})
        rh = hashlib.sha256(canonical_json(
            [{"id": h["segment_id"], "score": h["score"]} for h in hits])).hexdigest()
        return {"hits": hits, "results_hash": rh}

    def search_text(self, query: str, k: int = 5,
                    filters: dict | None = None,
                    use_torch: bool = False) -> dict:
        if self._embed_fn is None:
            raise RuntimeError("no embed_fn: construct Index(embed_fn=...)")
        return self.search_by_vector(np.asarray(self._embed_fn(query)), k,
                                     filters, use_torch=use_torch)

    def search_similar(self, segment_id: str | None = None,
                       vector: np.ndarray | None = None, k: int = 5,
                       filters: dict | None = None,
                       use_torch: bool = False) -> dict:
        """Find segments similar to an example. The query segment itself is
        excluded from hits (documented in docs/reports/hub.md)."""
        self._ensure_ordered()
        if vector is None:
            if segment_id is None or segment_id not in self._vecs:
                raise KeyError(f"unknown segment_id: {segment_id}")
            vector = self._mat32[self._ids.index(segment_id)]
        res = self.search_by_vector(np.asarray(vector), k=k + 1,
                                    filters=filters, use_torch=use_torch)
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

        Refuses N > max_clusters_n (ClusterLimitError) — the pairwise graph
        is O(N^2) and a giant cluster's medoid is O(m^2); the API surface
        maps this to HTTP 413. No cap on cluster SIZE below the limit.

        Returns {"clusters": [{"representative", "members", "size"}],
        "n_segments", "n_clusters", "dup_fraction"}. Representative = medoid
        (max in-cluster similarity sum, blockwise; ties by id).
        """
        self._ensure_ordered()
        ids, f32 = self._ids, self._mat32
        n = len(ids)
        if n == 0:
            return {"clusters": [], "n_segments": 0, "n_clusters": 0,
                    "dup_fraction": 0.0}
        if n > self.max_clusters_n:
            raise ClusterLimitError(
                f"dedup_clusters refuses N={n} > max_clusters_n="
                f"{self.max_clusters_n} (O(N^2) pairwise graph); "
                f"raise max_clusters_n explicitly or shard the index")
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

        # Upper triangle only (halves the blocked matmuls).
        for i0 in range(0, n, block):
            a = f32[i0:i0 + block]
            for j0 in range(i0, n, block):
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
            B = f32[members]  # (m, D)
            mlen = len(members)
            sums = np.zeros(mlen, dtype=np.float64)
            cb = 512  # blockwise: O(m*cb) memory, never O(m^2)
            for c0 in range(0, mlen, cb):
                sums += (B @ B[c0:c0 + cb].T).sum(axis=1)
            best = min(range(mlen),
                       key=lambda r: (-round(float(sums[r]), 9), ids[members[r]]))
            rep = ids[members[best]]
            mem_ids = sorted(ids[m] for m in members)
            clusters.append({"representative": rep, "members": mem_ids,
                             "size": len(mem_ids)})
        clusters.sort(key=lambda c: c["representative"])
        dup = (n - len(clusters)) / n if n else 0.0
        return {"clusters": clusters, "n_segments": n,
                "n_clusters": len(clusters), "dup_fraction": round(dup, 4)}
