"""SOP documents and deterministic retrieval for Jadal QA.

`sop_index.SOPIndex.load(dir)` reads every `*.md` file in a directory and
returns verbatim passages ranked by BM25 (offline fallback) or by any embedder
implementing the fixed `Embedder` interface. Import nothing heavy here so that
`python -m sop.sop_index` stays quiet and no network is touched at import time.
See `sop_index.py`'s docstring for what is not done.
"""
