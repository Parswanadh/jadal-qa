"""Train final conv head on ALL activity windows (seed 0), save weights."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
CFG = os.path.normpath(os.path.join(HERE, "..", "..", "configs",
                                    "perpixel_conv.pt"))


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    z = np.load(os.path.join(RES, "series_gold.npz"), allow_pickle=True)
    smap = {str(i): np.asarray(s, dtype=np.float64)
            for i, s in zip(z["ids"], z["S"])}
    gm = json.load(open(
        "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))
    items = {it["id"]: it for it in gm["items"]}
    keep = [i for i in sorted(smap)
            if items[i]["label"] == 1
            or items[i]["reason"] in ("NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE")]
    T = 100
    Xa = np.stack([np.resize(smap[i], (T, 8)) for i in keep])
    ya = np.array([items[i]["label"] for i in keep])
    mu, sd = Xa.mean((0, 1)), Xa.std((0, 1)) + 1e-9
    Xtr = torch.tensor((Xa - mu) / sd, dtype=torch.float32).transpose(1, 2)
    ytr = torch.tensor(ya, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Conv1d(8, 16, 5, padding=2), torch.nn.ReLU(),
        torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
        torch.nn.Linear(16, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    loss = torch.nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(40):
        opt.zero_grad()
        o = net(Xtr).squeeze(1)
        l = loss(o, ytr)
        l.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(net(Xtr)).numpy()
    from sklearn.metrics import roc_auc_score
    thr = float(np.percentile(ptr[ya == 1], 5))
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "arch": "conv1d-8x16-k5x2-pool-fc", "epochs": 40, "seed": 0,
                "T": T, "threshold": thr, "n": len(keep)}, CFG)
    print("saved thr5:", round(thr, 4),
          "train_auroc:", round(float(roc_auc_score(ya, ptr)), 3), flush=True)


if __name__ == "__main__":
    main()
