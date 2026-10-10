"""Generate the dashboard mock fixtures in web/mock/*.json.

Run:
    /home/parshu/envs/jadal/bin/python web/mock/make_mock.py

The numbers here are ILLUSTRATIVE. Nothing in this directory is a measured
result; the fixtures exist so the dashboard can be built and screenshotted
before `jadal/service.py` exists. Field names follow docs/SPEC.md exactly, plus
one documented enrichment (`segments` on a batch) that carries worker/site and
per-segment latency, which SPEC's Batch shape does not define.

Receipt hashes are computed with the SPEC rule:
    sha256 of canonical JSON (sorted keys, separators (",", ":"), UTF-8) of the
    receipt without receipt_sha256 / receipt_id / created_utc.
`receipt_id` = first 16 hex chars of that hash.

NOT DONE:
- No real clips, models or calibration data behind these numbers.
- `risk_coverage` / `sop_gaps` are hand-shaped, not produced by
  `jadal/certify.py`; the error bound below is a real Clopper-Pearson upper
  bound on the fabricated (errors, n) pair, nothing more.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent

LANGS = ["hi-IN", "kn-IN", "ta-IN", "te-IN", "en-IN"]

SITES = ["site_tirupur_02", "site_ludhiana_07", "site_surat_01"]
WORKERS = ["w_arun", "w_meena", "w_kavya", "w_rahul", "w_deepa", "w_sanjay"]

STEPS = ["align_hem", "attach_collar", "press_seam", "inspect_label", "pack"]


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def receipt_id_of(body: dict) -> tuple[str, str]:
    h = hashlib.sha256(canonical(body)).hexdigest()
    return h, h[:16]


def exact_binomial_upper(errors: int, n: int, delta: float = 0.1) -> float:
    """Clopper-Pearson upper bound on the error rate, solved by bisection.

    Pure Python so the fixture generator needs no scientific stack. Matches the
    interface documented for jadal/certify.exact_binomial_upper.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if errors < 0 or errors > n:
        raise ValueError("errors out of range")
    if errors == 0:
        return 1.0 - delta ** (1.0 / n)

    def cdf(k: int, p: float) -> float:
        # P(X <= k) for X ~ Binomial(n, p)
        total = 0.0
        for i in range(k + 1):
            total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        return total

    lo, hi = errors / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(errors - 1, mid) > delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def r4(x: float) -> float:
    return float(f"{x:.4f}")


def make_signals(rng: random.Random, phash: str, dup_of: str | None) -> dict:
    return {
        "decodable": True,
        "blur_var": r4(rng.uniform(18.0, 240.0)),
        "brightness_mean": r4(rng.uniform(28.0, 168.0)),
        "under_exposed_frac": r4(rng.uniform(0.0, 0.22)),
        "over_exposed_frac": r4(rng.uniform(0.0, 0.41)),
        "motion_mean": r4(rng.uniform(0.4, 13.5)),
        "hands_visible_frac": r4(rng.uniform(0.11, 0.98)),
        "hand_count_mode": rng.choice([0, 1, 2, 2]),
        "faces_found": rng.choice([0, 0, 1, 1, 2]),
        "faces_blurred": 0,
        "phash": phash,
        "dup_of": dup_of,
    }


def make_decisions(rng: random.Random, step_choice: str | None, conf: float) -> dict:
    probs = {s: 0.0 for s in STEPS}
    if step_choice:
        rest = 1.0 - conf
        others = [s for s in STEPS if s != step_choice]
        weights = [rng.uniform(0.2, 1.0) for _ in others]
        total = sum(weights)
        for s, w in zip(others, weights):
            probs[s] = r4(rest * w / total)
        probs[step_choice] = r4(1.0 - sum(probs[s] for s in others))
    else:
        weights = [rng.uniform(0.2, 1.0) for _ in STEPS]
        total = sum(weights)
        for s, w in zip(STEPS, weights):
            probs[s] = r4(w / total)
    return {
        "hands_ok": {"p_yes": r4(rng.uniform(0.2, 0.99))},
        "active_work": {"p_yes": r4(rng.uniform(0.15, 0.98))},
        "step": {"choice": step_choice, "probs": probs, "answer_conf": r4(conf)},
        "model_versions": {
            "hands": "embeddinggemma2-3b-heads@sha256:9f2c1d4e",
            "active_work": "embeddinggemma2-3b-heads@sha256:9f2c1d4e",
            "step": "embeddinggemma2-3b-heads@sha256:9f2c1d4e",
        },
    }


THRESHOLDS = {
    "hands_ok": 0.55,
    "active_work": 0.60,
    "step": 0.50,
    "blur_var_min": 60.0,
    "brightness_min": 40.0,
    "over_exposed_frac_max": 0.35,
    "motion_max": 12.0,
    "hands_visible_frac_min": 0.5,
}


def build() -> None:
    rng = random.Random(1234)
    base = datetime(2026, 10, 10, 9, 12, 0, tzinfo=timezone.utc)

    # (batch_id, site, decision, segment specs)
    # each spec: (verdict, reasons, step_choice, conf)
    plans = [
        (
            "B-20261010-0912-a3f1",
            SITES[0],
            "accept",
            [
                ("accept", [], "align_hem", 0.91),
                ("accept", [], "attach_collar", 0.87),
                ("accept", [], "press_seam", 0.78),
                ("accept", [], "align_hem", 0.83),
                ("review", ["LOW_CONFIDENCE"], "inspect_label", 0.44),
                ("accept", [], "pack", 0.88),
                ("accept", [], "press_seam", 0.81),
                ("accept", [], "attach_collar", 0.76),
            ],
        ),
        (
            "B-20261010-0931-7c09",
            SITES[1],
            "reject",
            [
                ("accept", [], "align_hem", 0.90),
                ("reject", ["HANDS_NOT_VISIBLE", "LOW_CONFIDENCE"], None, 0.31),
                ("reject", ["BLUR", "SHAKY"], None, 0.22),
                ("review", ["WRONG_STEP", "LOW_CONFIDENCE"], "press_seam", 0.47),
                ("reject", ["DUPLICATE"], "pack", 0.35),
            ],
        ),
        (
            "B-20261010-1004-d21b",
            SITES[2],
            "accept",
            [
                ("accept", [], "press_seam", 0.85),
                ("accept", [], "align_hem", 0.79),
                ("accept", [], "inspect_label", 0.92),
                ("reject", ["TOO_DARK"], None, 0.19),
                ("review", ["OVEREXPOSED", "LOW_CONFIDENCE"], None, 0.27),
                ("review", ["NO_ACTIVE_WORK", "LOW_CONFIDENCE"], None, 0.41),
                ("accept", [], "pack", 0.86),
            ],
        ),
    ]

    batches = []
    receipts = {}

    for bi, (batch_id, site, decision, specs) in enumerate(plans):
        created = base + timedelta(minutes=19 * bi, seconds=bi * 7)
        seg_ids, verdicts, seg_meta = [], [], []
        prev_id = None
        prev_phash = ""
        for si, (verdict, reasons, step, conf) in enumerate(specs):
            seg_id = f"seg_{created:%Y%m%d_%H%M}_{bi}{si:02d}"
            clip = f"clip_{site}_headcam_0{bi + 1}.mp4"
            dup_of = prev_id if (si == len(specs) - 1 and bi == 1) else None

            phash = prev_phash if dup_of else "".join(rng.choice("0123456789abcdef") for _ in range(16))
            signals = make_signals(rng, phash, dup_of)
            if verdict == "reject" and "DUPLICATE" in reasons:
                signals["faces_found"] = 1
                signals["faces_blurred"] = 1
            decisions = make_decisions(rng, step, conf)
            verdict_obj = {
                "segment_id": seg_id,
                "verdict": verdict,
                "reasons": reasons,
                "answer_conf": r4(conf),
                "threshold": 0.5,
            }

            body = {
                "segment_sha256": hashlib.sha256(f"{clip}:{seg_id}".encode()).hexdigest(),
                "signals": signals,
                "decisions": decisions,
                "verdict": verdict_obj,
                "config": {
                    "thresholds": THRESHOLDS,
                    "pipeline_version": "0.1.0",
                    "versions": {
                        "python": "3.12.15",
                        "torch": "2.9.0+cu130",
                        "opencv": "4.11.0",
                        "models": decisions["model_versions"],
                    },
                },
            }
            h, rid = receipt_id_of(body)
            receipt = dict(body)
            receipt["receipt_id"] = rid
            receipt["created_utc"] = (created + timedelta(seconds=si * 21)).strftime("%Y-%m-%dT%H:%M:%SZ")
            receipt["receipt_sha256"] = h
            receipts[rid] = receipt

            seg_ids.append(seg_id)
            prev_id = seg_id
            prev_phash = phash
            verdicts.append(verdict_obj)
            seg_meta.append(
                {
                    "segment_id": seg_id,
                    "receipt_id": rid,
                    "worker_id": WORKERS[(bi * 3 + si) % len(WORKERS)],
                    "site_id": site,
                    "source_clip": clip,
                    "start_s": r4(si * 20.0),
                    "end_s": r4(si * 20.0 + 20.0),
                    "latency_ms": rng.randint(1400, 5200),
                }
            )

        rejected = sum(1 for v in verdicts if v["verdict"] == "reject")
        k = 1
        derived = "reject" if rejected > k else "accept"
        assert derived == decision, f"{batch_id}: rule gives {derived}, plan said {decision}"
        batches.append(
            {
                "batch_id": batch_id,
                "site_id": site,
                "segment_ids": seg_ids,
                "segments": seg_meta,
                "verdicts": verdicts,
                "batch_decision": decision,
                "rule": "reject if rejected_count > k",
                "k": k,
                "n": len(seg_ids),
                "rejected_count": rejected,
                "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )

    # --- risk / coverage ------------------------------------------------------
    points = []
    for i in range(21):
        t = i / 20.0
        coverage = 1.0 - 0.55 * t
        error = 0.34 * (t ** 1.7) / (1.0 + 2.5 * t)
        points.append({"threshold": r4(t), "coverage": r4(coverage), "error": r4(error)})
    n, errors, delta, target = 240, 12, 0.1, 0.05
    bound = r4(exact_binomial_upper(errors, n, delta))

    risk = {
        "_note": "ILLUSTRATIVE mock curve. Not a measured calibration result.",
        "points": points,
        "n": n,
        "errors": errors,
        "target_error": target,
        "delta": delta,
        "error_bound": bound,
        "min_accepted": 30,
        "split": "val_20261009",
        "generated_utc": "2026-10-10T08:40:11Z",
    }

    # --- sop gaps -------------------------------------------------------------
    gaps = [
        {"reason_code": "HANDS_NOT_VISIBLE", "count": 14, "sop_section": "4.1 table height", "share": 0.28},
        {"reason_code": "WRONG_STEP", "count": 11, "sop_section": "3.2 hem alignment", "share": 0.22},
        {"reason_code": "BLUR", "count": 8, "sop_section": "2.5 head-camera fit", "share": 0.16},
        {"reason_code": "LOW_CONFIDENCE", "count": 7, "sop_section": "5.3 lighting", "share": 0.14},
        {"reason_code": "NO_ACTIVE_WORK", "count": 5, "sop_section": "3.1 station setup", "share": 0.10},
        {"reason_code": "TOO_DARK", "count": 3, "sop_section": "5.3 lighting", "share": 0.06},
        {"reason_code": "DUPLICATE", "count": 2, "sop_section": "1.4 upload retries", "share": 0.04},
    ]
    sop = {
        "_note": "ILLUSTRATIVE mock gap counts. Not a measured result.",
        "gaps": gaps,
        "total_segments": 50,
        "window_utc": "2026-10-09T00:00:00Z/2026-10-10T00:00:00Z",
    }

    pico = {
        "_note": "ILLUSTRATIVE mock status for GET /pico/status.",
        "connected": True,
        "led": "A",
        "led_meaning": "review pending",
        "port": "/dev/ttyACM0",
        "baud": 115200,
        "ticks_ms": 1_486_220,
        "last_heartbeat_utc": "2026-10-10T10:20:00Z",
        "firmware": "jadal-pico-0.1.0",
        "flash_pattern": None,
    }

    def dump(name: str, obj) -> None:
        (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {name} ({(OUT / name).stat().st_size} bytes)")

    dump("batches.json", batches)
    dump("receipts.json", receipts)
    dump("risk_coverage.json", risk)
    dump("sop_gaps.json", sop)
    dump("pico_status.json", pico)
    print(f"receipts: {len(receipts)}  batches: {len(batches)}")
    print(f"exact_binomial_upper(errors={errors}, n={n}, delta={delta}) = {bound}")


if __name__ == "__main__":
    build()
