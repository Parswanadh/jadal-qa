"""Unit tests for sop/sop_index.py — deterministic SOP retrieval.

Run from the repo root:

    /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v

No network, no GPU, no model download: the default path is BM25 and the
embedding path is exercised with stub embedders written here. Test scratch
lives under the repo's gitignored data_runtime/, never /tmp.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sop.sop_index import (  # noqa: E402
    BM25,
    HashEmbedder,
    Passage,
    SOPIndex,
    parse_sop_markdown,
    tokenize,
)

SAMPLE_SOP = REPO_ROOT / "sop" / "sop_garment_v1.md"


class ScratchTestCase(unittest.TestCase):
    """Base class giving each test a scratch directory inside the repo."""

    def setUp(self) -> None:
        base = REPO_ROOT / "data_runtime" / "test_scratch"
        try:
            base.mkdir(parents=True, exist_ok=True)
            self.tmp = Path(tempfile.mkdtemp(dir=str(base)))
        except OSError:
            self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_sop(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path


class StubEmbedder:
    """Fixed-interface test double: maps exact text -> vector."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.dim = len(next(iter(vectors.values())))
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return [list(self._vectors.get(text, [0.0] * self.dim)) for text in texts]


class TokenizeTests(unittest.TestCase):
    def test_ascii_words_are_lowercased_and_split(self) -> None:
        self.assertEqual(tokenize("Hold the camera STEADY."), ("hold", "the", "camera", "steady"))

    def test_indic_words_do_not_split_into_characters(self) -> None:
        tokens = tokenize("कैमरा को स्थिर रख")
        self.assertIn("कैमरा", tokens)
        self.assertIn("रख", tokens)
        self.assertTrue(all(len(t) > 1 for t in tokens if t.isalpha()))

    def test_telugu_tamil_kannada_tokens_are_words(self) -> None:
        for word in ("కెమెరా", "கேமரா", "ಕ್ಯಾಮೆರಾ"):
            self.assertIn(word, tokenize(word))


class ParseSopMarkdownTests(unittest.TestCase):
    def test_sample_sop_parses_five_labelled_steps(self) -> None:
        doc = parse_sop_markdown(SAMPLE_SOP.read_text(encoding="utf-8"), doc_id="sop_garment_v1")
        self.assertEqual(doc.sop_id, "sop_garment_v1")
        self.assertEqual(doc.lang, "en-IN")
        self.assertEqual(doc.status, "SAMPLE")
        self.assertEqual(len(doc.passages), 5)
        self.assertEqual([p.step for p in doc.passages], [1, 2, 3, 4, 5])
        self.assertEqual(
            [p.passage_id for p in doc.passages],
            [f"sop_garment_v1/{n:03d}" for n in range(1, 6)],
        )

    def test_passage_text_is_a_verbatim_slice_of_the_file(self) -> None:
        source = SAMPLE_SOP.read_text(encoding="utf-8")
        doc = parse_sop_markdown(source, doc_id="sop_garment_v1")
        for passage in doc.passages:
            self.assertIn(passage.text, source)
            self.assertTrue(passage.text.startswith("## Step"))

    def test_file_without_step_sections_becomes_one_passage(self) -> None:
        doc = parse_sop_markdown("# Title\n\nSome body text here.\n", doc_id="flat")
        self.assertEqual(len(doc.passages), 1)
        self.assertEqual(doc.passages[0].passage_id, "flat/001")
        self.assertIsNone(doc.passages[0].lang)
        self.assertEqual(doc.passages[0].text, "# Title\n\nSome body text here.")

    def test_header_lang_is_kept(self) -> None:
        doc = parse_sop_markdown(
            "<!--\nsop_id: x\nlang: hi-IN\n-->\n\n## Step 1 — a\n\nbody\n", doc_id="doc"
        )
        self.assertEqual(doc.lang, "hi-IN")
        self.assertEqual(doc.passages[0].lang, "hi-IN")


class SOPIndexLoadTests(ScratchTestCase):
    def test_load_sample_index(self) -> None:
        index = SOPIndex.load(REPO_ROOT / "sop")
        self.assertEqual(len(index), 5)
        self.assertEqual(index.languages(), ["en-IN"])
        self.assertEqual([d.sop_id for d in index.docs], ["sop_garment_v1"])

    def test_load_missing_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SOPIndex.load(REPO_ROOT / "sop" / "does-not-exist")

    def test_load_orders_passages_by_id_not_by_directory_order(self) -> None:
        self.write_sop("b_sop.md", "<!--\nlang: hi-IN\n-->\n## Step 1 — क\n\nकैमरा स्थिर\n")
        self.write_sop("a_sop.md", "<!--\nlang: hi-IN\n-->\n## Step 1 — क\n\nकैमरा स्थिर\n")
        index = SOPIndex.load(self.tmp)
        self.assertEqual(
            [p.passage_id for p in index.passages], ["a_sop/001", "b_sop/001"]
        )


class BM25Tests(unittest.TestCase):
    def test_rare_term_beats_common_term(self) -> None:
        docs = [
            "blur blur blur blur".split(),
            "the the the the blur".split(),
            "stitch hem".split(),
        ]
        scores = BM25(docs).scores(["blur"])
        self.assertEqual(scores[2], 0.0)
        self.assertGreater(scores[0], scores[1])

    def test_empty_corpus_returns_empty(self) -> None:
        self.assertEqual(BM25([]).scores(["blur"]), [])


class SearchTests(ScratchTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SAMPLE_SOP.read_text(encoding="utf-8")
        cls.index = SOPIndex.load(REPO_ROOT / "sop")

    def test_relevant_step_is_ranked_first_by_bm25(self) -> None:
        top = self.index.search("broken needle and belt", lang="en-IN", top_k=3)
        self.assertEqual(top[0]["step"], 1)
        self.assertEqual(top[0]["passage_id"], "sop_garment_v1/001")

    def test_results_are_ranked_and_truncated(self) -> None:
        results = self.index.search("collar seam stitch", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_text_is_verbatim_and_never_generated(self) -> None:
        for result in self.index.search("press the collar with the iron", top_k=5):
            passage = self.index.passage(result["passage_id"])
            self.assertIsNotNone(passage)
            self.assertEqual(result["text"], passage.text)
            self.assertIn(result["text"], self.source)
            self.assertNotIn("summary", result)
            self.assertNotIn("answer", result)

    def test_same_query_twice_is_identical(self) -> None:
        first = self.index.search("trim the seam allowance", top_k=5)
        second = self.index.search("trim the seam allowance", top_k=5)
        self.assertEqual(first, second)

    def test_result_order_does_not_depend_on_passage_order(self) -> None:
        shuffled = list(self.index.passages)
        random.Random(7).shuffle(shuffled)
        other = SOPIndex(shuffled)
        self.assertEqual(
            self.index.search("collar points", top_k=5),
            other.search("collar points", top_k=5),
        )

    def test_unmatched_query_returns_no_hits(self) -> None:
        self.assertEqual(self.index.search("quantum blockchain ledger"), [])
        self.assertEqual(self.index.search(""), [])
        self.assertEqual(self.index.search("   "), [])
        self.assertEqual(self.index.search("collar", top_k=0), [])

    def test_lang_filter_excludes_other_languages(self) -> None:
        self.write_sop(
            "hi.md",
            "<!--\nsop_id: hi_sop\nlang: hi-IN\n-->\n## Step 1 — सेटअप\n\nकैमरा स्थिर रख\n",
        )
        self.write_sop(
            "ta.md",
            "<!--\nsop_id: ta_sop\nlang: ta-IN\n-->\n## Step 1 — அமைப்பு\n\nகேமரா நிலையாக\n",
        )
        index = SOPIndex.load(self.tmp)
        self.assertEqual([r["sop_id"] for r in index.search("कैमरा", lang="hi-IN")], ["hi_sop"])
        self.assertEqual([r["sop_id"] for r in index.search("कैमरा", lang="ta-IN")], [])
        self.assertEqual(index.search("कैमरा", lang="kn-IN"), [])
        self.assertEqual([r["sop_id"] for r in index.search("कैमरा")], ["hi_sop"])


class TieBreakTests(unittest.TestCase):
    def _index(self) -> SOPIndex:
        passages = [
            Passage(passage_id="doc/003", doc_id="doc", sop_id="doc", lang="en-IN",
                    title="three", text="same body text here"),
            Passage(passage_id="doc/001", doc_id="doc", sop_id="doc", lang="en-IN",
                    title="one", text="same body text here"),
            Passage(passage_id="doc/002", doc_id="doc", sop_id="doc", lang="en-IN",
                    title="two", text="same body text here"),
        ]
        return SOPIndex(passages)

    def test_bm25_tie_is_broken_by_passage_id(self) -> None:
        results = self._index().search("same body text")
        self.assertEqual([r["passage_id"] for r in results], ["doc/001", "doc/002", "doc/003"])

    def test_embedding_tie_is_broken_by_passage_id(self) -> None:
        vector = [1.0, 0.0, 0.0, 0.0]
        embedder = StubEmbedder({"same body text here": vector, "same body text": vector})
        results = self._index().search("same body text", embedder=embedder)
        self.assertEqual([r["passage_id"] for r in results], ["doc/001", "doc/002", "doc/003"])
        self.assertTrue(all(r["score"] == 1.0 for r in results))


class EmbedderPathTests(unittest.TestCase):
    def _index(self) -> SOPIndex:
        passages = [
            Passage(passage_id="doc/001", doc_id="doc", sop_id="doc", lang="en-IN",
                    title="keyword match", text="blur blur blur blur"),
            Passage(passage_id="doc/002", doc_id="doc", sop_id="doc", lang="en-IN",
                    title="embedding match", text="stitch the hem neatly"),
        ]
        return SOPIndex(passages)

    def test_embedding_score_outranks_bm25_keyword_match(self) -> None:
        embedder = StubEmbedder(
            {
                "blur": [1.0, 0.0, 0.0, 0.0],
                "blur blur blur blur": [0.0, 1.0, 0.0, 0.0],
                "stitch the hem neatly": [1.0, 0.0, 0.0, 0.0],
            }
        )
        results = self._index().search("blur", embedder=embedder)
        self.assertEqual([r["passage_id"] for r in results], ["doc/002"])
        self.assertEqual(results[0]["score"], 1.0)

    def test_embedding_tie_falls_back_to_bm25_then_id(self) -> None:
        embedder = StubEmbedder(
            {
                "blur": [1.0, 0.0, 0.0, 0.0],
                "blur blur blur blur": [1.0, 0.0, 0.0, 0.0],
                "stitch the hem neatly": [1.0, 0.0, 0.0, 0.0],
            }
        )
        results = self._index().search("blur", embedder=embedder)
        self.assertEqual([r["passage_id"] for r in results], ["doc/001", "doc/002"])
        self.assertEqual(results[0]["bm25"], 1.0)
        self.assertEqual(results[1]["bm25"], 0.0)

    def test_passages_are_embedded_once_per_index(self) -> None:
        embedder = StubEmbedder(
            {
                "blur": [1.0, 0.0, 0.0, 0.0],
                "blur blur blur blur": [1.0, 0.0, 0.0, 0.0],
                "stitch the hem neatly": [0.0, 1.0, 0.0, 0.0],
            }
        )
        index = self._index()
        index.search("blur", embedder=embedder)
        index.search("blur", embedder=embedder)
        passage_calls = [call for call in embedder.calls if len(call) == 2]
        self.assertEqual(len(passage_calls), 1)

    def test_zero_vector_query_returns_nothing(self) -> None:
        embedder = StubEmbedder(
            {
                "blur blur blur blur": [1.0, 0.0, 0.0, 0.0],
                "stitch the hem neatly": [1.0, 0.0, 0.0, 0.0],
            }
        )
        self.assertEqual(self._index().search("blur", embedder=embedder), [])

    def test_embedder_with_wrong_dim_raises(self) -> None:
        class BadDim:
            dim = 3

            def embed(self, texts):
                return [[1.0, 0.0] for _ in texts]

        with self.assertRaises(ValueError):
            self._index().search("blur", embedder=BadDim())


class HashEmbedderTests(unittest.TestCase):
    def test_hash_embedder_is_deterministic_and_offline(self) -> None:
        self.assertEqual(HashEmbedder().embed(["collar seam"]), HashEmbedder().embed(["collar seam"]))
        self.assertEqual(HashEmbedder().dim, 512)

    def test_hash_embedder_query_hits_the_matching_step(self) -> None:
        index = SOPIndex.load(REPO_ROOT / "sop", embedder=HashEmbedder(dim=4096))
        results = index.search("cut the collar pieces with the pattern", top_k=1)
        self.assertEqual(results[0]["step"], 2)

    def test_hash_embedder_path_is_repeatable(self) -> None:
        index = SOPIndex.load(REPO_ROOT / "sop", embedder=HashEmbedder(dim=256))
        self.assertEqual(
            index.search("press the collar flat", top_k=3),
            index.search("press the collar flat", top_k=3),
        )


class ResultShapeTests(unittest.TestCase):
    def test_search_result_is_json_serialisable_with_fixed_fields(self) -> None:
        index = SOPIndex.load(REPO_ROOT / "sop")
        result = index.search("pack the folded garment", top_k=1)[0]
        self.assertEqual(
            sorted(result),
            sorted(
                [
                    "rank", "passage_id", "doc_id", "sop_id", "lang", "title",
                    "step", "score", "bm25", "text",
                ]
            ),
        )
        json.dumps(result, ensure_ascii=False)

    def test_scores_are_rounded_to_six_decimals(self) -> None:
        index = SOPIndex.load(REPO_ROOT / "sop")
        for result in index.search("collar stitch seam", top_k=5):
            self.assertEqual(result["score"], round(result["score"], 6))
            self.assertLessEqual(result["score"], 1.0)
            self.assertGreater(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
