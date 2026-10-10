"""Priority path (deadline order): stratified 1k subset of egocentric_10k.

Fixed seed 0: all 358 hand==0 frames + 300 work==0 frames + random fill to
1000. Features reused from the cached full-10k extraction (same deterministic
extractor, d=53). Trains/evaluates heads with bootstrap CIs + family ablation
+ majority baseline. Labels are AI labels (gemini-2.5-flash), NOT human.

Usage: python -m eval.frames_spectral.subset_heads
Out: eval/results/frames_spectral/subset_1k.npz, subset_heads.json,
     subset_oof_logreg.npz
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from eval.frames_spectral.frame_features import FAMILIES, FEATURES
from eval.frames_spectral.train_heads import boot_ci_safe, metrics_binary, oof_predict

IN = "eval/results/frames_spectral"
SEED = 0
N_SUB = 1000


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    z = np.load(f"{IN}/egocentric_10k.npz", allow_pickle=True)
    X = z["X"].astype(np.float64)
    hand = z["hand"].astype(int)
    work = z["work"].astype(int)
    ids = [str(s) for s in z["ids"].tolist()]
    n = len(ids)
    idx0 = np.where(hand == 0)[0]
    idx_no = np.where(work == 0)[0]
    rng2 = np.random.default_rng(SEED)
    take_no = rng2.choice(idx_no, min(300, len(idx_no)), replace=False)
    base = np.union1d(idx0, take_no)
    rest = np.setdiff1d(np.arange(n), base)
    fill = rng2.choice(rest, N_SUB - len(base), replace=False)
    sel = np.sort(np.concatenate([base, fill]))
    assert len(sel) == N_SUB
    Xs, hands, works = X[sel], hand[sel], work[sel]
    idss = [ids[i] for i in sel]
    np.savez_compressed(f"{IN}/subset_1k.npz", X=Xs.astype(np.float32),
                        hand=hands.astype(np.int8), work=works.astype(np.int8),
                        ids=np.asarray(idss), sel=sel.astype(np.int32),
                        features=np.asarray(FEATURES))
    import hashlib
    grp_rand = np.random.default_rng(SEED).integers(0, 5, N_SUB)
    grp_hash = np.asarray([int(hashlib.md5(i.encode()).hexdigest(), 16) % 5 for i in idss])
    log_fn = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                        random_state=SEED)
    hgb_fn = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                    max_leaf_nodes=31, random_state=SEED)
    out: dict = {
        "label_source": "AI (gemini-2.5-flash per dataset README), NOT human",
        "n": N_SUB, "seed": SEED,
        "composition": {"hand0_all": int((hands == 0).sum()),
                        "work_no": int((works == 0).sum()),
                        "hand_counts": {str(k): int((hands == k).sum()) for k in (0, 1, 2)}}}
    tasks = {"hands_visible": (hands > 0).astype(int), "active_labor": works}
    oof_store = {}
    for tname, y in tasks.items():
        out[tname] = {}
        maj = int(y.mean() >= 0.5)
        majpred = np.full(N_SUB, maj)
        out[tname]["majority_baseline"] = {
            "bal_acc": float(balanced_accuracy_score(y, majpred)),
            "f1": float(f1_score(y, majpred, zero_division=0))}
        for pname, grp in (("random_split", grp_rand), ("frameid_hash", grp_hash)):
            out[tname][pname] = {}
            for mname, mfn in (("logreg", log_fn), ("hgb", hgb_fn)):
                p = oof_predict(Xs, y, mfn, grp)
                out[tname][pname][mname] = metrics_binary(y, p, rng)
                if mname == "logreg" and pname == "random_split":
                    oof_store[tname] = p
    # 3-class hand_count
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    P = np.zeros((N_SUB, 3))
    for tr, te in skf.split(Xs, hands):
        sc = StandardScaler()
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                               random_state=SEED)
        m.fit(sc.fit_transform(Xs[tr]), hands[tr])
        P[te] = m.predict_proba(sc.transform(Xs[te]))
    pred = P.argmax(1)
    out["hand_count_3class"] = {
        "macro_f1": float(f1_score(hands, pred, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(hands, pred))}
    # ablation, logreg random_split
    out["ablation"] = {}
    for tname, y in tasks.items():
        base_auc = out[tname]["random_split"]["logreg"]["auroc"]
        d = {"full_logreg_auroc": base_auc, "drop_one_family": {}}
        for fam, members in FAMILIES.items():
            keep = [i for i in range(Xs.shape[1]) if FEATURES[i] not in members]
            p = oof_predict(Xs[:, keep], y, log_fn, grp_rand)
            try:
                a = float(roc_auc_score(y, p))
            except Exception:
                a = float("nan")
            d["drop_one_family"][fam] = {"auroc": a, "delta": float(a - base_auc)}
        out["ablation"][tname] = d
    out["elapsed_s"] = time.time() - t0
    np.savez_compressed(f"{IN}/subset_oof_logreg.npz",
                        hands_visible=oof_store["hands_visible"],
                        active_labor=oof_store["active_labor"],
                        hand=hands.astype(np.int8), work=works.astype(np.int8))
    with open(f"{IN}/subset_heads.json", "w") as f:
        json.dump(out, f, indent=1)
    for tname in tasks:
        for pname in ("random_split", "frameid_hash"):
            for mname in ("logreg", "hgb"):
                m = out[tname][pname][mname]
                print(f"{tname}/{pname}/{mname}: AUROC={m['auroc']:.3f} "
                      f"CI={m['auroc_ci']} balAcc={m['bal_acc']:.3f} F1={m['f1']:.3f}", flush=True)
    print("wrote subset_heads.json", flush=True)


if __name__ == "__main__":
    main()
