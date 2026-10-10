"""B2 / E8.01 contract helpers: canonical JSON, digests, receipts, verdicts.

The PoC contract is one JSONL line per scored window with EXACTLY these keys:

    window_id, factory, worker, clip, start_s, end_s, verdict, stage,
    reasons[{code, evidence{feature, value, op, threshold}}], score, features{},
    features_digest, receipt_sha256, versions{}, latency_ms{},
    label_source

`receipt_sha256` = SHA-256 of the canonical JSON of the line without
`receipt_sha256` (content-hashed, NOT signed -- never call it signed or
unforgeable).  `features_digest` = SHA-256 of the canonical `features{}` block.

Determinism: all floats are rounded (`round_floats`) before hashing, so the
same input bytes + same device + same recipe give the same digests.
"""
from __future__ import annotations

import hashlib
import json
import math

CONTRACT_FIELDS = (
    "window_id", "factory", "worker", "clip", "start_s", "end_s", "verdict",
    "stage", "reasons", "score", "features", "features_digest",
    "receipt_sha256", "versions", "latency_ms", "label_source",
)

VERDICTS = ("accept", "reject", "review")

#: honesty note carried in the JSONL header line, the report and the registry
HONESTY_NOTE = (
    "These 1,546 Egocentric-10K windows have NO human labels: label_source is "
    "'none' and every accept/reject/review verdict here is DESCRIPTIVE, not "
    "accuracy. IndEgo-trained models did not transfer out-of-domain "
    "(0.4-0.53 AUROC on the 60 human-labelled Egocentric-10K windows, n_bad=9), "
    "so the accept/reject share below is NOT a quality measure."
)


def canonical_bytes(obj) -> bytes:
    """Canonical JSON bytes: sorted keys, compact separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def round_floats(value, decimals: int = 4):
    """Recursively round floats; -0.0 becomes 0.0; NaN/inf raise."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float: {value!r}")
        return round(value, decimals) + 0.0
    if isinstance(value, dict):
        return {k: round_floats(v, decimals) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v, decimals) for v in value]
    return value


def features_block(pixel_feats, eg2_mean, n_frames_pixel, n_frames_eg2,
                   fps_pixel, p_eg2, p_pixel, decimals: int = 4) -> dict:
    """`features{}` for one window + its digest.

    The 26 pixel features are stored explicitly; the 768-d EG2 mean is stored
    by content hash (the full vectors live in
    `/workspace/artifacts/exp_rescore/eg2_mean_l2.npy`, whose sha256 is recorded
    in the results JSON), so the digest stays small and is re-verifiable.
    """
    block = {k: round(float(v), decimals) for k, v in sorted(pixel_feats.items())}
    block["eg2_mean_sha256"] = sha256_hex(
        canonical_bytes([round(float(x), decimals) for x in eg2_mean]))
    block["eg2_dim"] = int(len(eg2_mean))
    block["n_frames_pixel"] = int(n_frames_pixel)
    block["n_frames_eg2"] = int(n_frames_eg2)
    block["fps_pixel"] = round(float(fps_pixel), 4)
    block["p_eg2"] = round(float(p_eg2), decimals)
    block["p_pixel"] = round(float(p_pixel), decimals)
    return block, sha256_hex(canonical_bytes(block))


def verdict_for(score: float, accept_thr: float, reject_thr: float) -> str:
    """Profile operating point: score >= accept -> accept; <= reject -> reject."""
    if score >= accept_thr:
        return "accept"
    if score <= reject_thr:
        return "reject"
    return "review"


def reasons_for(score, verdict, accept_thr, reject_thr, route, pixel_feats,
                decimals: int = 4) -> list:
    """Evidence list: route rule + verdict rule + documented physics heuristics."""
    out = []
    for code, ev in route["reason_evidence"]:
        out.append({"code": code, "evidence": ev})
    op = ">=" if verdict == "accept" else ("<=" if verdict == "reject" else "in")
    thr = accept_thr if verdict == "accept" else (
        reject_thr if verdict == "reject" else round(accept_thr - reject_thr, decimals))
    out.append({"code": {"accept": "SCORE_AT_OR_ABOVE_PROFILE_ACCEPT",
                         "reject": "SCORE_AT_OR_BELOW_PROFILE_REJECT",
                         "review": "LOW_MARGIN_PROFILE_REVIEW"}[verdict],
                "evidence": {"feature": "usable_prob", "value": round(float(score), decimals),
                             "op": op, "threshold": round(float(thr), decimals)}})
    # documented heuristics (NOT validated operating points)
    if float(pixel_feats.get("dark_frac", 0.0)) > 0.5:
        out.append({"code": "HEURISTIC_DARK",
                    "evidence": {"feature": "dark_frac",
                                 "value": round(float(pixel_feats["dark_frac"]), decimals),
                                 "op": ">", "threshold": 0.5}})
    if float(pixel_feats.get("frozen_frac", 0.0)) > 0.5:
        out.append({"code": "HEURISTIC_FROZEN",
                    "evidence": {"feature": "frozen_frac",
                                 "value": round(float(pixel_feats["frozen_frac"]), decimals),
                                 "op": ">", "threshold": 0.5}})
    return out


def route_from_decision(decision: dict, descriptors: dict) -> dict:
    """Router decision -> (chosen profile, margin, reason evidence list)."""
    prof = decision["profile"]
    if prof == "factory_1080p30":
        ev = [("PROFILE_ROUTE_HIRES",
               {"feature": "width", "value": round(float(descriptors["width"]), 4),
                "op": ">=", "threshold": 1280.0}),
              ("PROFILE_ROUTE_HIRES_FPS",
               {"feature": "fps", "value": round(float(descriptors["fps"]), 4),
                "op": ">=", "threshold": 25.0})]
    else:  # pragma: no cover - no other profile occurred on this corpus
        ev = [("PROFILE_ROUTE_" + prof.upper(),
               {"feature": "router_margin", "value": round(float(decision["margin"]), 4),
                "op": ">=", "threshold": 0.0})]
    if decision.get("unseen_regime"):
        ev.append(("UNSEEN_REGIME",
                   {"feature": "router_margin", "value": 0.0, "op": "==", "threshold": 0.0}))
    return {"profile": prof, "margin": round(float(decision["margin"]), 4),
            "profile_reasons": list(decision["reasons"]),
            "unseen_regime": bool(decision.get("unseen_regime")),
            "reason_evidence": ev}


def build_line(window, verdict, stage, reasons, score, features_digest,
               features, versions, latency_ms) -> dict:
    """Assemble one contract line and stamp its receipt."""
    line = {
        "window_id": window["window_id"],
        "factory": window["factory"],
        "worker": window["worker"],
        "clip": window["clip"],
        "start_s": round(float(window["start_s"]), 4),
        "end_s": round(float(window["end_s"]), 4),
        "verdict": verdict,
        "stage": stage,
        "reasons": reasons,
        "score": round(float(score), 4),
        "features": features,
        "features_digest": features_digest,
        "versions": versions,
        "latency_ms": round_floats(latency_ms, 3),
        "label_source": "none",
    }
    body = round_floats(line, 4)
    body.pop("receipt_sha256", None)
    line = body
    line["receipt_sha256"] = sha256_hex(canonical_bytes(body))
    return line


def verify_line(line: dict) -> bool:
    """Re-verify a contract line's two digests from its own content."""
    if tuple(sorted(line)) != tuple(sorted(CONTRACT_FIELDS)):
        return False
    body = {k: v for k, v in line.items() if k != "receipt_sha256"}
    if sha256_hex(canonical_bytes(body)) != line["receipt_sha256"]:
        return False
    return sha256_hex(canonical_bytes(line["features"])) == line["features_digest"]


def header_line(n, recipe, profile_dist, verdict_dist, device, git_sha,
                artifact_sha256=None) -> dict:
    return {
        "type": "header",
        "n_windows": int(n),
        "label_source": "none",
        "protocol": "descriptive re-score (no human labels on these windows)",
        "honesty_note": HONESTY_NOTE,
        "recipe": recipe,
        "profile_distribution": profile_dist,
        "verdict_distribution": verdict_dist,
        "device": device,
        "git_sha": git_sha,
        "windows_body_sha256": artifact_sha256,
        "receipt_note": "receipt_sha256 = SHA-256 of the canonical JSON line; content-hashed, NOT signed",
    }
