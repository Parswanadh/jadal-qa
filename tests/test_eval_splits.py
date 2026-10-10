"""Tests for the eval harness: make_splits components/splits, check_leakage,
calibration_report (select-on-calibration / report-on-test).

Run: PYTHONNOUSERSITE=1 python3 -m unittest discover -s tests
(from the repo root). All fixtures are synthetic; real parquet inputs untouched.
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
    META_COLUMNS,
    assign_splits,
    build_manifest,
    component_records,
    component_stratum,
    find_components,
    hamming,
    hash_parquet,
    largest_remainder,
    scan_metadata,
    split_distribution,
)
from eval import calibration_report as cr
from eval.check_leakage import count_cross_split_pairs


def _png_bytes(seed: int, size: tuple[int, int] = (32, 32)) -> bytes:
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict], with_image: bool = True) -> None:
    fields = {
        "frame_id": [r["frame_id"] for r in rows],
        "source_dataset": [r["source"] for r in rows],
        "hand_count": pa.array([r["hand"] for r in rows], type=pa.int32()),
        "active_labor": [r["label"] for r in rows],
    }
    if with_image:
        fields["image"] = [{"bytes": r["png"], "path": r["frame_id"] + ".png"} for r in rows]
    pq.write_table(pa.table(fields), path)


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


class TestComponents(unittest.TestCase):
    def _rec(self, fid: str, ph: str, source: str = "s", hand: int = 1, label: str = "yes") -> dict:
        return {"frame_id": fid, "source_dataset": source, "hand_count": hand,
                "active_labor": label, "phash": ph}

    def test_exact_and_boundary(self) -> None:
        base = "0" * 16
        recs = [
            self._rec("a", base),
            self._rec("b", base),  # exact dup
            self._rec("c", "0" * 15 + "f"),  # 4 bits -> same component
            self._rec("d", "1f" + "0" * 14),  # 5 bits from base, 9 from c -> own component
            self._rec("e", "f" * 16),  # far -> own component
        ]
        comps, report = find_components(recs, threshold=4)
        self.assertEqual(report["bucket_cutoff"], None)
        self.assertEqual([[recs[i]["frame_id"] for i in c] for c in comps], [["a", "b", "c"], ["d"], ["e"]])
        self.assertEqual(report["removed_rows"], 2)
        self.assertEqual(report["multiframe_components"], 1)
        kept, removed = component_records(recs, comps)
        self.assertEqual([r["frame_id"] for r in kept], ["a", "d", "e"])
        self.assertEqual(len(removed), 2)

    def test_large_bucket_collapses(self) -> None:
        # >400 identical hashes (the old code skipped such buckets): one component.
        recs = [self._rec(f"f{i:04d}", "0" * 16) for i in range(500)]
        recs.append(self._rec("near", "0" * 15 + "1"))  # 1 bit off -> joins
        recs.append(self._rec("far", "f" * 16))  # stays alone
        comps, report = find_components(recs, threshold=4)
        self.assertEqual(len(comps), 2)
        self.assertEqual(len(comps[0]), 501)
        self.assertEqual(comps[0][0], 0)  # representative = smallest frame_id f0000
        self.assertEqual(report["removed_rows"], 500)
        self.assertIsNone(report["bucket_cutoff"])
        # exact-collapse did the work: only the near/far distinct hashes were compared
        self.assertLessEqual(report["hash_pairs_checked"], 3)

    def test_stratum_majority(self) -> None:
        recs = [
            self._rec("a", "0" * 16, source="ego4d", hand=1, label="yes"),
            self._rec("b", "0" * 16, source="ego4d", hand=2, label="yes"),
            self._rec("c", "0" * 16, source="kit", hand=1, label="no"),
        ]
        self.assertEqual(component_stratum(recs, [0, 1, 2]), ("ego4d", 1, "yes"))

    def test_real_images_exact_dup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.parquet"
            _write_parquet(path, _synthetic_rows())
            recs = hash_parquet(path, workers=1, batch_size=16)
        self.assertEqual(len(recs), 34)
        comps, report = find_components(recs, threshold=4)
        kept, _ = component_records(recs, comps)
        kept_ids = {r["frame_id"] for r in kept}
        self.assertEqual(len(kept_ids & {"ego_dup_a", "ego_dup_b"}), 1)
        self.assertGreaterEqual(report["removed_rows"], 1)


class TestLargestRemainder(unittest.TestCase):
    def test_exact_totals(self) -> None:
        for n in (1, 2, 3, 7, 10, 100, 4471):
            counts = largest_remainder(n, (0.70, 0.15, 0.15))
            self.assertEqual(sum(counts.values()), n, msg=f"n={n}")
            self.assertEqual(set(counts), {"train", "calibration", "test"})

    def test_close_to_ratios(self) -> None:
        counts = largest_remainder(1000, (0.70, 0.15, 0.15))
        self.assertEqual(counts, {"train": 700, "calibration": 150, "test": 150})


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

    def test_scan_without_image_column(self) -> None:
        # proves scan_metadata never requests image bytes (review I3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.parquet"
            _write_parquet(path, _synthetic_rows(), with_image=False)
            report = scan_metadata(Path(tmp), files=["t.parquet"])
        self.assertEqual(report["total_rows"], 34)
        self.assertNotIn("image", META_COLUMNS)


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
                    # random hashes: overwhelmingly singleton components
                    "phash": f"{rng.getrandbits(64):016x}",
                }
            )
        return recs

    def _comps(self, recs: list[dict]) -> list[list[int]]:
        comps, _ = find_components(recs, threshold=4)
        return comps

    def test_deterministic_no_overlap_full_coverage(self) -> None:
        recs = self._recs()
        comps = self._comps(recs)
        a = assign_splits(recs, comps, seed=1234)
        b = assign_splits(recs, comps, seed=1234)
        self.assertEqual(a["assignment"], b["assignment"])
        self.assertEqual(set(a["assignment"]), {r["frame_id"] for r in recs})

    def test_component_atomicity(self) -> None:
        # near-dup pair with disagreeing strata must still land in ONE split
        recs = [
            {"frame_id": "x1", "source_dataset": "ego4d", "hand_count": 1, "active_labor": "yes",
             "phash": "0" * 16},
            {"frame_id": "x2", "source_dataset": "ego4d", "hand_count": 2, "active_labor": "no",
             "phash": "0" * 15 + "3"},  # 2 bits -> same component
        ]
        comps, _ = find_components(recs, threshold=4)
        self.assertEqual(len(comps), 1)
        got = assign_splits(recs, comps, seed=5)
        self.assertEqual(got["assignment"]["x1"], got["assignment"]["x2"])

    def test_ratios_approx(self) -> None:
        recs = self._recs(2000)
        comps = self._comps(recs)
        a = assign_splits(recs, comps, seed=1234)
        dist = split_distribution(recs, a["assignment"])
        total = sum(dist[s]["n"] for s in ("train", "calibration", "test"))
        self.assertEqual(total, len(recs))
        self.assertAlmostEqual(dist["train"]["n"] / total, 0.70, delta=0.03)
        self.assertAlmostEqual(dist["calibration"]["n"] / total, 0.15, delta=0.03)
        self.assertAlmostEqual(dist["test"]["n"] / total, 0.15, delta=0.03)

    def test_per_stratum_near_exact(self) -> None:
        # largest-remainder targets hold up to the biggest component in the stratum
        recs = self._recs(2000)
        comps = self._comps(recs)
        a = assign_splits(recs, comps, seed=1234)
        by_stratum: dict[tuple, list[list[int]]] = {}
        for members in comps:
            from eval.make_splits import component_stratum as cs
            by_stratum.setdefault(cs(recs, members), []).append(members)
        for stratum, members in by_stratum.items():
            n = sum(len(m) for m in members)
            biggest = max(len(m) for m in members)
            targets = largest_remainder(n, (0.70, 0.15, 0.15))
            got = {s: 0 for s in targets}
            for m in members:
                got[a["assignment"][recs[m[0]]["frame_id"]]] += len(m)
            for s in targets:
                self.assertLessEqual(abs(got[s] - targets[s]), biggest, msg=f"{stratum} {s}")

    def test_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.parquet"
            _write_parquet(path, _synthetic_rows())
            meta = scan_metadata(Path(tmp), files=["t.parquet"])
            recs = hash_parquet(path, workers=1, batch_size=16)
            comps, comp_report = find_components(recs, threshold=4)
            kept, removed = component_records(recs, comps)
            split = assign_splits(recs, comps, seed=1234)
            manifest = build_manifest(
                recs, split["assignment"], split["strata"], seed=1234, ratios=(0.7, 0.15, 0.15),
                dup_report={**comp_report, "removed": removed}, metadata_report=meta,
                data_dir=Path(tmp), files=["t.parquet"], dup_threshold=4,
            )
        self.assertEqual(set(manifest["splits"]), {"train", "calibration", "test"})
        total = sum(manifest["splits"][s]["count"] for s in manifest["splits"])
        self.assertEqual(total, len(recs))
        self.assertIn("peak_rss_mb", manifest)

    def test_cli_determinism(self) -> None:
        # two full CLI runs give byte-identical manifests apart from created_utc
        from eval import make_splits as ms

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            src = t / "data"
            src.mkdir()
            _write_parquet(src / "a.parquet", _synthetic_rows())
            outs = []
            for run in ("r1", "r2"):
                out = t / run
                rc = ms.main(["--data-dir", str(src), "--files", "a.parquet",
                              "--out-dir", str(out),
                              "--no-cache", "--workers", "1", "--batch-size", "16"])
                self.assertEqual(rc, 0)
                outs.append(json.loads((out / "splits.json").read_text()))
            for m in outs:
                m.pop("created_utc")
                m.pop("peak_rss_mb")  # resource usage is not part of determinism
            self.assertEqual(outs[0], outs[1])


class TestLeakage(unittest.TestCase):
    def test_no_cross_split_pairs(self) -> None:
        rng = random.Random(11)
        recs = []
        for i in range(120):
            recs.append({"frame_id": f"g{i:03d}", "source_dataset": "ego4d",
                         "hand_count": i % 3, "active_labor": "yes" if i % 2 else "no",
                         "phash": f"{rng.getrandbits(64):016x}"})
        # a near-dup pair (2 bits) that must stay together
        recs.append({"frame_id": "pair_a", "source_dataset": "ego4d", "hand_count": 1,
                     "active_labor": "yes", "phash": "abcd" + "0" * 12})
        recs.append({"frame_id": "pair_b", "source_dataset": "ego4d", "hand_count": 1,
                     "active_labor": "yes", "phash": "abce" + "0" * 12})
        comps, _ = find_components(recs, threshold=4)
        got = assign_splits(recs, comps, seed=1234)
        split_ids = {s: sorted(fid for fid, sp in got["assignment"].items() if sp == s)
                     for s in ("train", "calibration", "test")}
        hashes = {r["frame_id"]: r["phash"] for r in recs}
        cross, _ = count_cross_split_pairs(split_ids, hashes, threshold=4)
        self.assertEqual(cross, 0)
        self.assertEqual(got["assignment"]["pair_a"], got["assignment"]["pair_b"])

    def test_detector_finds_planted_leak(self) -> None:
        split_ids = {"train": ["a"], "calibration": ["b"], "test": []}
        hashes = {"a": "0" * 16, "b": "0" * 15 + "1"}
        cross, examples = count_cross_split_pairs(split_ids, hashes, threshold=4)
        self.assertEqual(cross, 1)
        self.assertEqual(len(examples), 1)


class TestCalibrationParsing(unittest.TestCase):
    def test_p_yes_formats(self) -> None:
        self.assertEqual(cr._as_p_yes(0.8), 0.8)
        self.assertEqual(cr._as_p_yes({"yes": 0.8, "no": 0.2}), 0.8)
        self.assertEqual(cr._as_p_yes([0.2, 0.8]), 0.8)
        self.assertEqual(cr._as_p_yes({"0": 0.2, "1": 0.8}), 0.8)
        self.assertIsNone(cr._as_p_yes(None))
        self.assertIsNone(cr._as_p_yes(float("nan")))
        self.assertIsNone(cr._as_p_yes({"a": 0.3, "b": 0.7}))  # ambiguous order refused

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
    def _preds(self, n: int, start: int = 0, seed: int = 3) -> list[dict]:
        rng = random.Random(seed)
        out = []
        for i in range(n):
            p = rng.random()
            y = 1 if rng.random() < p else 0  # calibrated by construction
            out.append({"frame_id": f"f{start + i:04d}", "probs": p, "label": y})
        return out

    def test_ece_small_for_calibrated(self) -> None:
        preds = self._preds(200)
        conf = [p["probs"] for p in preds]
        y = [p["label"] for p in preds]
        self.assertLess(cr.local_ece(conf, y, bins=15), 0.10)

    def test_risk_coverage_shape(self) -> None:
        preds = self._preds(200)
        conf = [max(p["probs"], 1 - p["probs"]) for p in preds]
        correct = [int(p["probs"] >= 0.5) == p["label"] for p in preds]
        pts = cr.local_risk_coverage(conf, correct)
        self.assertGreater(len(pts), 5)
        covs = [p["coverage"] for p in pts]
        self.assertEqual(covs, sorted(covs))  # thresholds descend, coverage ascends
        for p in pts:
            self.assertGreaterEqual(p["coverage"], 0.0)
            self.assertLessEqual(p["coverage"], 1.0)

    def test_local_pick_threshold_certified(self) -> None:
        rng = random.Random(9)
        conf = [0.99] * 60 + [0.6] * 40
        correct = [1 if rng.random() < 0.99 else 0 for _ in range(60)]
        correct += [1 if rng.random() < 0.6 else 0 for _ in range(40)]
        sel = cr.local_pick_threshold(conf, correct, 0.05, 0.1, 30)
        self.assertIsNotNone(sel["threshold"])
        self.assertLessEqual(sel["upper_bound"], 0.05)
        self.assertGreaterEqual(sel["accepted"], 30)

    def test_end_to_end_select_calib_report_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "preds.json").write_text(json.dumps(self._preds(60, 0) + self._preds(60, 60, seed=5)))
            (t / "splits.json").write_text(json.dumps({"splits": {
                "train": {"frame_ids": []},
                "calibration": {"frame_ids": [f"f{i:04d}" for i in range(60)]},
                "test": {"frame_ids": [f"f{i:04d}" for i in range(60, 120)]},
            }}))
            rc = cr.main([
                "--predictions", str(t / "preds.json"),
                "--splits", str(t / "splits.json"),
                "--select-split", "calibration", "--report-split", "test",
                "--json-out", str(t / "cal.json"), "--png-out", str(t / "cal.png"),
            ])
            self.assertEqual(rc, 0)
            report = json.loads((t / "cal.json").read_text())
            self.assertEqual(report["n"], 60)
            self.assertEqual(report["n_select"], 60)
            self.assertIn("ece", report)
            self.assertIn("applied_to_report", report["threshold"])
            self.assertTrue((t / "cal.png").exists())
            self.assertGreater((t / "cal.png").stat().st_size, 1000)

    def test_overlap_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "preds.json").write_text(json.dumps(self._preds(30)))
            (t / "splits.json").write_text(json.dumps({"splits": {
                "train": {"frame_ids": []},
                "calibration": {"frame_ids": [f"f{i:04d}" for i in range(30)]},
                "test": {"frame_ids": [f"f{i:04d}" for i in range(30)]},
            }}))
            with self.assertRaises(SystemExit):
                cr.main([
                    "--predictions", str(t / "preds.json"),
                    "--splits", str(t / "splits.json"),
                    "--select-split", "calibration", "--report-split", "calibration",
                    "--json-out", str(t / "cal.json"), "--png-out", str(t / "cal.png"),
                ])

    def test_restrict_to_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "splits.json").write_text(
                json.dumps({"splits": {
                    "train": {"frame_ids": ["a", "b"]},
                    "calibration": {"frame_ids": ["c"]},
                    "test": {"frame_ids": ["d"]},
                }})
            )
            recs = [{"frame_id": f, "p_yes": 0.5, "y": 1} for f in "abcd"]
            kept, dropped = cr.restrict_to_split(recs, t / "splits.json", "test")
            self.assertEqual([r["frame_id"] for r in kept], ["d"])
            self.assertEqual(dropped, 3)


if __name__ == "__main__":
    unittest.main()
