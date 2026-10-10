"""Tests for the eval harness: make_splits helpers and calibration_report.

Run from the repo root:
    /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v

All data here is synthetic (tiny PNGs written to a temp dir); nothing depends
on the 30k-row eval parquet files.
"""

from __future__ import annotations

import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from eval.make_splits import (
    assign_splits,
    build_manifest,
    dedup,
    hamming,
    hash_parquet,
    scan_metadata,
    split_distribution,
)
from eval import calibration_report as cr


def _png_bytes(seed: int, size: tuple[int, int] = (32, 32), tweak: int = 0) -> bytes:
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    if tweak:
        px[0, 0] = ((px[0, 0][0] + tweak) % 256, px[0, 0][1], px[0, 0][2])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.table(
        {
            "frame_id": [r["frame_id"] for r in rows],
            "image": [{"bytes": r["png"], "path": r["frame_id"] + ".png"} for r in rows],
            "source_dataset": [r["source"] for r in rows],
            "hand_count": pa.array([r["hand"] for r in rows], type=pa.int32()),
            "active_labor": [r["label"] for r in rows],
        }
    )
    pq.write_table(table, path)


def _synthetic_rows() -> list[dict]:
    rows: list[dict] = []
    specs = [
        ("ego", 0, "no", 4),
        ("ego", 1, "yes", 6),
        ("ego", 2, "yes", 6),
        ("kit", 0, "no", 4),
        ("kit", 1, "yes", 6),
        ("kit", 2, "no", 6),
    ]
    i = 0
    for source, hand, label, count in specs:
        for k in range(count):
            rows.append(
                {
                    "frame_id": f"{source}_{hand}_{label}_{k:02d}",
                    "png": _png_bytes(1000 + i),
                    "source": source,
                    "hand": hand,
                    "label": label,
                }
            )
            i += 1
    rows.append({"frame_id": "ego_dup_a", "png": _png_bytes(4242), "source": "ego", "hand": 1, "label": "yes"})
    rows.append({"frame_id": "ego_dup_b", "png": _png_bytes(4242), "source": "ego", "hand": 1, "label": "yes"})
    return rows


class TestHamming(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(hamming("0" * 16, "0" * 16), 0)
        self.assertEqual(hamming("0" * 16, "f" * 16), 64)
        self.assertEqual(hamming("0" * 15 + "0", "0" * 15 + "1"), 1)
        self.assertEqual(hamming("0" * 16, "0" * 15 + "f"), 4)


class TestDedup(unittest.TestCase):
    def _rec(self, fid: str, ph: str) -> dict:
        return {"frame_id": fid, "source_dataset": "s", "hand_count": 1, "active_labor": "yes", "phash": ph}

    def test_exact_and_boundary(self) -> None:
        base = "0" * 16
        recs = [
            self._rec("a", base),
            self._rec("b", base),  # exact dup
            self._rec("c", "0" * 15 + "f"),  # 4 bits -> grouped
            self._rec("d", "0" * 13 + "1ff"),  # 5+ bits from a and c -> kept
            self._rec("e", "f" * 16),  # far -> kept
        ]
        kept, report = dedup(recs, threshold=4)
        kept_ids = sorted(r["frame_id"] for r in kept)
        self.assertEqual(kept_ids, ["a", "d", "e"])
        self.assertEqual(report["removed_rows"], 2)
        self.assertEqual(report["duplicate_groups"], 1)

    def test_representative_is_min_frame_id(self) -> None:
        recs = [self._rec("zebra", "0" * 16), self._rec("apple", "0" * 16)]
        kept, _ = dedup(recs, threshold=4)
        self.assertEqual([r["frame_id"] for r in kept], ["apple"])

    def test_real_images_exact_dup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.parquet"
            _write_parquet(path, _synthetic_rows())
            recs = hash_parquet(path, workers=1, batch_size=16)
        self.assertEqual(len(recs), 34)
        kept, report = dedup(recs, threshold=4)
        kept_ids = {r["frame_id"] for r in kept}
        self.assertEqual(len(kept_ids & {"ego_dup_a", "ego_dup_b"}), 1)
        self.assertGreaterEqual(report["removed_rows"], 1)


class TestMetadata(unittest.TestCase):
    def test_scan_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.parquet"
            _write_parquet(path, _synthetic_rows())
            report = scan_metadata(Path(tmp), files=["t.parquet"])
        f = report["files"]["t.parquet"]
        self.assertEqual(f["rows"], 34)
        self.assertEqual(report["total_rows"], 34)
        self.assertEqual(report["distinct_frame_ids"], 34)
        self.assertEqual(f["distinct_source_dataset"], {"ego": 18, "kit": 16})
        self.assertEqual(sum(int(v) for v in f["hand_count"].values()), 34)


class TestSplits(unittest.TestCase):
    def _recs(self, n: int = 300) -> list[dict]:
        rng = random.Random(7)
        recs: list[dict] = []
        for i in range(n):
            recs.append(
                {
                    "frame_id": f"f{i:05d}",
                    "source_dataset": rng.choice(["ego4d", "egocentric_10k", "epic_kitchens"]),
                    "hand_count": rng.choice([0, 1, 2]),
                    "active_labor": rng.choice(["yes", "no"]),
                    "phash": f"{i:016x}",
                }
            )
        return recs

    def test_deterministic_no_overlap_full_coverage(self) -> None:
        recs = self._recs()
        a = assign_splits(recs, seed=1234)
        b = assign_splits(recs, seed=1234)
        self.assertEqual(a["assignment"], b["assignment"])
        self.assertEqual(set(a["assignment"]), {r["frame_id"] for r in recs})

    def test_ratios_approx(self) -> None:
        recs = self._recs(2000)
        a = assign_splits(recs, seed=1234)
        dist = split_distribution(recs, a["assignment"])
        total = sum(dist[s]["n"] for s in ("train", "calibration", "test"))
        self.assertEqual(total, len(recs))
        self.assertAlmostEqual(dist["train"]["n"] / total, 0.70, delta=0.03)
        self.assertAlmostEqual(dist["calibration"]["n"] / total, 0.15, delta=0.03)
        self.assertAlmostEqual(dist["test"]["n"] / total, 0.15, delta=0.03)

    def test_stratified_by_source(self) -> None:
        recs = self._recs(2000)
        a = assign_splits(recs, seed=99)
        dist = split_distribution(recs, a["assignment"])
        for src in ("ego4d", "egocentric_10k", "epic_kitchens"):
            n_src = sum(dist[s]["by_source"].get(src, 0) for s in dist)
            frac_train = dist["train"]["by_source"].get(src, 0) / n_src
            self.assertAlmostEqual(frac_train, 0.70, delta=0.06)

    def test_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.parquet"
            _write_parquet(path, _synthetic_rows())
            meta = scan_metadata(Path(tmp), files=["t.parquet"])
            recs = hash_parquet(path, workers=1, batch_size=16)
            kept, dup = dedup(recs, threshold=4)
            split = assign_splits(kept, seed=1234)
            manifest = build_manifest(
                kept, split["assignment"], split["strata"], seed=1234, ratios=(0.7, 0.15, 0.15),
                dup_report=dup, metadata_report=meta,
                data_dir=Path(tmp), files=["t.parquet"], dup_threshold=4,
            )
        self.assertEqual(set(manifest["splits"]), {"train", "calibration", "test"})
        total = sum(manifest["splits"][s]["count"] for s in manifest["splits"])
        self.assertEqual(total, len(kept))
        for name in manifest["splits"]:
            self.assertIsInstance(manifest["splits"][name]["fraction"], float)


class TestCalibrationParsing(unittest.TestCase):
    def test_p_yes_formats(self) -> None:
        self.assertEqual(cr._as_p_yes(0.8), 0.8)
        self.assertEqual(cr._as_p_yes({"yes": 0.8, "no": 0.2}), 0.8)
        self.assertEqual(cr._as_p_yes([0.2, 0.8]), 0.8)
        self.assertEqual(cr._as_p_yes({"0": 0.2, "1": 0.8}), 0.8)
        self.assertIsNone(cr._as_p_yes(None))
        self.assertIsNone(cr._as_p_yes(float("nan")))

    def test_label_formats(self) -> None:
        self.assertEqual(cr._as_label("yes"), 1)
        self.assertEqual(cr._as_label("no"), 0)
        self.assertEqual(cr._as_label(1), 1)
        self.assertEqual(cr._as_label(0), 0)
        self.assertEqual(cr._as_label(True), 1)
        self.assertIsNone(cr._as_label("maybe"))

    def test_load_predictions_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.json"
            path.write_text(
                json.dumps(
                    [
                        {"frame_id": "a", "probs": {"yes": 0.9, "no": 0.1}, "label": "yes"},
                        {"frame_id": "b", "probs": 0.2, "label": "no"},
                        {"frame_id": "c", "probs": [0.4, 0.6], "label": 1},
                        {"frame_id": "d"},
                        {"frame_id": "e", "probs": 0.5, "label": "huh"},
                    ]
                )
            )
            recs, skipped = cr.load_predictions(path)
        self.assertEqual(len(recs), 3)
        self.assertEqual(skipped, 2)


class TestCalibrationReport(unittest.TestCase):
    def _preds(self, n: int = 200) -> list[dict]:
        rng = random.Random(3)
        out = []
        for i in range(n):
            p = 0.5 + 0.5 * rng.random()  # predicted class always "yes"
            y = 1 if rng.random() < p else 0  # calibrated by construction
            out.append({"frame_id": f"f{i:04d}", "probs": p, "label": y})
        return out

    def test_ece_small_for_calibrated(self) -> None:
        preds = self._preds()
        conf = [p["probs"] for p in preds]
        y = [p["label"] for p in preds]
        self.assertLess(cr.local_ece(conf, y, bins=15), 0.10)

    def test_risk_coverage_shape(self) -> None:
        preds = self._preds()
        conf = [p["probs"] for p in preds]
        correct = [int(p["probs"] >= 0.5) == p["label"] for p in preds]
        pts = cr.local_risk_coverage(conf, correct)
        self.assertGreater(len(pts), 5)
        covs = [p["coverage"] for p in pts]
        self.assertEqual(covs, sorted(covs))  # thresholds descend, coverage ascends
        for p in pts:
            self.assertGreaterEqual(p["coverage"], 0.0)
            self.assertLessEqual(p["coverage"], 1.0)
            self.assertGreaterEqual(p["error"], 0.0)
            self.assertLessEqual(p["error"], 1.0)

    def test_end_to_end_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "preds.json").write_text(json.dumps(self._preds()))
            rc = cr.main(
                [
                    "--predictions", str(t / "preds.json"),
                    "--json-out", str(t / "cal.json"),
                    "--png-out", str(t / "cal.png"),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads((t / "cal.json").read_text())
            self.assertEqual(report["n"], 200)
            self.assertIn("ece", report)
            self.assertIn("risk_coverage", report)
            self.assertEqual(report["ece_source"], "local_fallback")
            self.assertTrue((t / "cal.png").exists())
            self.assertGreater((t / "cal.png").stat().st_size, 1000)

    def test_restrict_to_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "preds.json").write_text(json.dumps(self._preds(50)))
            (t / "splits.json").write_text(
                json.dumps({"splits": {"test": {"frame_ids": [f"f{i:04d}" for i in range(10)]}}})
            )
            recs, _ = cr.load_predictions(t / "preds.json")
            kept, dropped = cr.restrict_to_split(recs, t / "splits.json", "test")
            self.assertEqual(len(kept), 10)
            self.assertEqual(dropped, 40)


if __name__ == "__main__":
    unittest.main()
