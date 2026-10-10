"""Unit tests for SKIP (Q2) section-sampling policies."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "exp_skip"))
import run_skip as sk  # noqa: E402


def test_sections_cover_all_windows():
    for n in (6, 17, 29, 48):
        ss = sk.sections(n)
        joined = np.concatenate(ss)
        assert len(joined) == n
        assert sorted(joined.tolist()) == list(range(n))
        assert len(ss) <= sk.NSEC
        # sections are contiguous, increasing blocks
        assert all(ss[i][-1] < ss[i + 1][0] for i in range(len(ss) - 1))


def test_drop_start_middle_end_count():
    n = 48
    ss = sk.sections(n)
    for rule, sec in (("start", 0), ("middle", 2), ("end", -1)):
        kept = sk.drop_sections(n, 1, rule)
        assert len(kept) == n - len(ss[sec])
        assert len(kept) >= n * 4 // 5 - 1
    assert 0 not in sk.drop_sections(n, 1, "start")
    assert (n - 1) not in sk.drop_sections(n, 1, "end")


def test_drop_evenly_is_spread_not_contiguous():
    n = 50
    kept = sk.drop_sections(n, 1, "evenly", seed=0)
    dropped = sorted(set(range(n)) - set(kept))
    assert len(dropped) == 10
    # dropped windows are spread across the whole clip (not one contiguous block)
    assert dropped[-1] - dropped[0] > n // 2


def test_strat_probe_first_of_each_section():
    n = 48
    S = sk.strat_probe(n, 0, "start")
    assert len(S) == 5
    assert S == [s[0] for s in sk.sections(n)]
    # after dropping the start section, the start window is gone
    S2 = sk.strat_probe(n, 1, "start")
    assert 0 not in S2


def test_verdict_threshold_agreement():
    bad = np.array([0, 0, 1, 1, 1])           # truth frac 0.6 -> reject
    p = np.array([0.9, 0.9, 0.4, 0.4, 0.4])
    r = sk.eval_clip(bad, p, [0, 1, 2, 3, 4])
    assert r["truth_v"] == "reject" and r["agree_L"] == 1 and r["err_L"] == 0


def test_every2_matches_e3_if_bundle_present():
    """Reproduce the E3 headline cell on cached data when available."""
    man = sk.MANIFEST
    if not (os.path.exists(man) and os.path.exists(sk.OOF)):
        return  # bundle not mounted; skip
    m, clips, y, bad, p = sk.load_gold()
    rows = []
    for c in clips:
        idx = clips[c]
        S = list(range(0, len(idx), 2))
        rows.append(sk.eval_clip(bad[idx], p[idx], S))
    assert abs(np.mean([r["frac"] for r in rows]) - 0.5146) < 0.01
    assert abs(np.mean([r["err_L"] for r in rows]) - 0.0486) < 0.005
    assert abs(np.mean([r["agree_L"] for r in rows]) - 1.0) < 1e-9
