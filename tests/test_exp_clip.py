"""Unit tests for Z4 (clip-level sequential decisions + active labels).

CPU-only, cached data.  Tests that need the pod bundle skip cleanly if it is absent.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "exp_clip"))
import run_clip as rc  # noqa: E402


def test_make_llr_hard_signs():
    # bad window (q_bad >= 0.5) -> positive; good window -> negative
    q = np.array([0.9, 0.1])
    llr = rc.make_llr(q, "hard", 0.15, 0.45)
    assert llr[0] > 0 > llr[1]
    assert abs(llr[0] - np.log(0.45 / 0.15)) < 1e-9
    assert abs(llr[1] - np.log(0.55 / 0.85)) < 1e-9


def test_seq_run_stops_at_reject_bound():
    clip = np.array(["desk"] * 5)
    start = np.arange(5.0)
    llr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    res = rc.seq_run(clip, start, llr, A=2.5, B=2.5)
    assert res["desk"]["decision"] == "reject"
    assert res["desk"]["k"] == 3           # 1+1+1 crosses 2.5


def test_seq_run_abstains_when_bounds_not_reached():
    clip = np.array(["desk"] * 3)
    start = np.arange(3.0)
    llr = np.array([0.1, -0.1, 0.1])
    res = rc.seq_run(clip, start, llr, A=5.0, B=5.0)
    assert res["desk"]["decision"] == "abstain"
    assert res["desk"]["k"] == 3


def test_bal_acc_perfect():
    ba, sens, spec = rc.bal_acc([1, 1, 0, 0], [1, 1, 0, 0])
    assert ba == 1.0 and sens == 1.0 and spec == 1.0


def test_rank_active_respects_exclusions_and_diversity():
    rows = []
    for i in range(40):
        rows.append({"window_id": f"w{i:03d}", "clip": f"c{i // 8}", "factory": f"f{i // 4}",
                     "score": 0.5, "verdict": "review",
                     "features": {"p_eg2": 0.5, "p_pixel": 0.5}})
    sel, cand = rc.rank_active(rows, exclude_ids={"w000"}, top=10)
    assert len(sel) == 10
    assert "w000" not in {s["window_id"] for s in sel}
    assert all(s["rank_score"] >= 0 for s in sel)


def test_z4_pipeline_matches_published_cell_if_bundle_present():
    if not (os.path.exists(rc.MANIFEST) and os.path.exists(rc.OOF)):
        return  # bundle not mounted; skip
    m, clip, start, y, p_ens, p_platt = rc.load_gold()
    _, cv = rc.clip_truth(clip, y)
    llr = rc.make_llr(1 - p_ens, "hard", 0.15, 0.45)
    res = rc.seq_run(clip, start, llr, 4.9, 3.7)
    mm = rc.seq_metrics(res, cv)
    assert mm["agreement_decided"] == 1.0
    assert abs(mm["abstain_rate"] - 2 / 7) < 1e-9
    assert abs(mm["mean_windows_processed_frac"] - 0.6653) < 0.01
