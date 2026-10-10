"""Deterministic physics rule-list scorer: fixed integer thresholds, no learned weights.

Design (see docs/reports/exp_phys_rules.md):
  * A fixed, a-priori candidate rule list maps named quality/semantic rules to
    physics features (the task's candidate rules: TOO_DARK, OVEREXPOSED, BLUR,
    FROZEN_STATIC, NO_ACTIVE_WORK, HANDS_NOT_VISIBLE, DUPLICATE, UNDECODABLE).
  * For every (rule, feature) we carry FOUR integer thresholds derived from the
    TRAINING clips only, at fixed low quantiles p01/p05/p10/p25 (mirrored to
    p99/p95/p90/p75 for high-is-bad features), quantised to 4 decimals.
  * score(window) = number of crossed thresholds (integer count). No weights are
    learned; every threshold contributes exactly +1.
  * Decision is deterministic: a quality rule that fires -> REJECT; otherwise a
    training-derived reject threshold on the integer score -> REJECT / REVIEW /
    ACCEPT. Every decision emits {rule, feature, value, op, threshold}.

Nothing here fits a parameter. The only quantities taken from data are quantiles
of training-clip feature values and one reject operating point chosen on
training-good windows.
"""
from __future__ import annotations

import numpy as np

# Fixed quantiles of the TRAINING distribution. Low-is-bad features get these on
# the lower tail; high-is-bad features get the mirror (1 - q) on the upper tail.
QUANTILES = (0.01, 0.05, 0.10, 0.25)
Q = 4  # decimal places for threshold quantisation

# ---------------------------------------------------------------------------
# Candidate rule list. Fields:
#   rule      : named rule (product vocabulary)
#   feature   : feature key, "<source>.<name>"
#   source    : "full" (26 GPU per-pixel features) or "phys" (16 physics feats)
#   op        : "<" or ">"  (direction that indicates a problem)
#   side      : "low" or "high" (which training tail supplies the thresholds)
#   strength  : "quality" (hard reject) or "weak" (semantic -> review band)
# ---------------------------------------------------------------------------
CANDIDATE_RULES = [
    # --- spec quality tripwires (hard) ---
    {"rule": "TOO_DARK",       "feature": "full.luma_mean",     "op": "<", "side": "low",  "strength": "quality"},
    {"rule": "TOO_DARK",       "feature": "full.dark_frac",     "op": ">", "side": "high", "strength": "quality"},
    {"rule": "OVEREXPOSED",    "feature": "full.bright_frac",   "op": ">", "side": "high", "strength": "quality"},
    {"rule": "BLUR",           "feature": "full.hf_ratio",      "op": "<", "side": "low",  "strength": "quality"},
    {"rule": "BLUR",           "feature": "full.dct_energy",    "op": "<", "side": "low",  "strength": "quality"},
    {"rule": "FROZEN_STATIC",  "feature": "full.temp_b0",       "op": "<", "side": "low",  "strength": "quality"},
    {"rule": "FROZEN_STATIC",  "feature": "full.frozen_frac",   "op": ">", "side": "high", "strength": "quality"},
    # --- weak / semantic rules (review band) ---
    {"rule": "NO_ACTIVE_WORK", "feature": "phys.flow_act",      "op": ">", "side": "high", "strength": "weak"},
    {"rule": "NO_ACTIVE_WORK", "feature": "phys.flow_med",      "op": ">", "side": "high", "strength": "weak"},
    {"rule": "NO_ACTIVE_WORK", "feature": "full.motion_energy", "op": ">", "side": "high", "strength": "weak"},
    {"rule": "HANDS_NOT_VISIBLE", "feature": "full.skin_frac",  "op": "<", "side": "low",  "strength": "weak"},
    {"rule": "DUPLICATE",      "feature": "phys.dhash_loop_min", "op": ">", "side": "high", "strength": "weak"},
    {"rule": "DUPLICATE",      "feature": "phys.mad_mean",      "op": ">", "side": "high", "strength": "weak"},
    {"rule": "CODING_NOISE",   "feature": "phys.frm_bytes_med", "op": ">", "side": "high", "strength": "weak"},
    {"rule": "CODING_NOISE",   "feature": "phys.p_size_med",    "op": ">", "side": "high", "strength": "weak"},
    {"rule": "CODING_NOISE",   "feature": "phys.snr_db",        "op": ">", "side": "high", "strength": "weak"},
]

# Core features whose non-finiteness means the window could not be decoded.
UNDECODABLE_FEATURES = ("full.luma_mean", "full.motion_energy", "full.luma_p05")


def derive_thresholds(values: np.ndarray, side: str) -> list[float]:
    """Four quantised thresholds from TRAINING values only."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 8 or len(np.unique(v)) < 2:
        return []
    if side == "low":
        qs = np.array(QUANTILES)
    else:
        qs = 1.0 - np.array(QUANTILES)
    return [round(float(t), Q) for t in np.quantile(v, qs)]


def build_rule_table(train_feats: dict[str, np.ndarray]) -> list[dict]:
    """Return the resolved rule table using only TRAINING feature vectors."""
    table = []
    for r in CANDIDATE_RULES:
        ths = derive_thresholds(train_feats[r["feature"]], r["side"])
        if not ths:
            continue
        table.append({**r, "thresholds": ths,
                      "quantiles": list(QUANTILES),
                      "source_quantile": ("p01/p05/p10/p25" if r["side"] == "low"
                                          else "p99/p95/p90/p75")})
    return table


def score_window(feats: dict[str, float], table: list[dict]):
    """Deterministic integer score + fired-rule evidence for one window.

    Crossing threshold index 0 (the p01/p99 extreme) is a *hard* fire; crossing
    the softer p10/p25 thresholds only adds to the integer score.
    """
    score = 0
    fired = []          # list of evidence dicts
    fired_rules = set()
    hard_rules = set()
    for r in table:
        v = feats.get(r["feature"])
        if v is None or not np.isfinite(v):
            continue
        crossed = 0
        best = None  # closest crossed threshold (smallest margin)
        for idx, t in enumerate(r["thresholds"]):
            hit = (v < t) if r["op"] == "<" else (v > t)
            if hit:
                crossed += 1
                if idx == 0:
                    hard_rules.add(r["rule"])
                margin = (t - v) if r["op"] == "<" else (v - t)
                if best is None or margin < best[1]:
                    best = (t, margin, idx)
        if crossed:
            score += crossed
            fired_rules.add(r["rule"])
            t, _, idx = best
            fired.append({"rule": r["rule"], "feature": r["feature"],
                          "value": round(float(v), Q), "op": r["op"],
                          "threshold": t, "strength": r["strength"],
                          "n_thresholds_crossed": crossed,
                          "hard": idx == 0})
    # UNDECODABLE tripwire (never fires in-domain; guards the decode path).
    und = [f for f in UNDECODABLE_FEATURES
           if f not in feats or not np.isfinite(feats.get(f, float("nan")))]
    if und:
        score += 10 ** 6
        fired_rules.add("UNDECODABLE")
        hard_rules.add("UNDECODABLE")
        fired.append({"rule": "UNDECODABLE", "feature": und[0], "value": None,
                      "op": "is_finite", "threshold": True,
                      "strength": "quality", "n_thresholds_crossed": 1,
                      "hard": True})
    strong = sorted(r for r in hard_rules
                    if any(e["strength"] == "quality" and e["rule"] == r for e in fired))
    weak = sorted({e["rule"] for e in fired if e["strength"] == "weak"})
    return {"score": int(score), "fired": fired, "fired_rules": sorted(fired_rules),
            "strong_fired": strong, "weak_fired": weak}


def choose_reject_threshold(train_scores: np.ndarray, train_y: np.ndarray,
                            fr_budget: float = 0.05) -> int:
    """Smallest integer s with training-good false-reject rate <= budget."""
    good = train_scores[np.asarray(train_y) == 1]
    if good.size == 0:
        return 1
    mx = int(train_scores.max())
    for s in range(1, mx + 2):
        if float((good >= s).mean()) <= fr_budget:
            return s
    return mx + 1


def choose_reject_threshold_ba(train_scores: np.ndarray, train_y: np.ndarray) -> int:
    """Integer s maximising training balanced accuracy (reject = score>=s)."""
    sc = np.asarray(train_scores)
    yy = np.asarray(train_y)
    good = yy == 1
    bad = yy == 0
    best_s, best_ba = 1, -1.0
    for s in range(1, int(sc.max()) + 2):
        tpr = float((sc[bad] >= s).mean()) if bad.any() else 0.0
        tnr = float((sc[good] < s).mean()) if good.any() else 0.0
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_s = ba, s
    return best_s


def decide(sc: dict, reject_threshold: int) -> dict:
    """Deterministic ACCEPT / REJECT / REVIEW with evidence."""
    if sc["strong_fired"]:
        ev = next(e for e in sc["fired"]
                  if e["rule"] == sc["strong_fired"][0] and e["hard"])
        return {"decision": "REJECT", "reason": "quality_rule",
                "evidence": ev, "score": sc["score"]}
    if sc["score"] >= reject_threshold:
        ev = (max(sc["fired"], key=lambda e: e["n_thresholds_crossed"])
              if sc["fired"] else None)
        return {"decision": "REJECT", "reason": "score>=reject_threshold",
                "evidence": ev, "score": sc["score"]}
    if sc["score"] >= 1:
        ev = sc["fired"][0]
        return {"decision": "REVIEW", "reason": "weak_rule",
                "evidence": ev, "score": sc["score"]}
    return {"decision": "ACCEPT", "reason": "no_rule", "evidence": None,
            "score": sc["score"]}
