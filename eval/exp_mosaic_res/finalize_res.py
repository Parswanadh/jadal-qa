"""Finalise X3 (m/mosaic-res) results -> MR.01..03.json + console table.

Reads eval/results/exp_mosaic_res/{features.npz,raw_eval.json}.
Regen: python3 eval/exp_mosaic_res/finalize_res.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "eval", "results", "exp_mosaic_res")
TILE_LADDER = [224, 160, 112, 96, 64]
DENSITY_GRIDS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
DENSITY_TILE = 112
CANVAS_DUEL = [("m_3x3_t224", 3, 3, 224), ("m_4x4_t168", 4, 4, 168)]

PROTOCOL = ("in-domain leave-one-clip-out over the 7 IndEgo clips (150 gold "
            "windows, 100 good / 50 bad), logistic C=0.01 on L2-normalised EG2 "
            "features, clip-bootstrap CI B=1000 seed0")
LABEL_SOURCE = ("human (IndEgo gold manifest_gold.json, one labeller); 150 "
                "windows; clips ecu_full/desk/organise/pallet/logistics4/"
                "ecu_late/pallet2")


def canvas_px(r, c, tp):
    return (c * tp + (c + 1) * 2) * (r * tp + (r + 1) * 2)


def load():
    raw = json.load(open(os.path.join(R, "raw_eval.json")))
    z = np.load(os.path.join(R, "features.npz"), allow_pickle=True)
    return raw, z


def main():
    raw, z = load()
    res, cost = raw["res"], raw["cost"]
    n = raw["n"]
    per = raw["per_cfg_ms"]
    git = os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip()
    head = {"n": n, "label_source": LABEL_SOURCE, "protocol": PROTOCOL,
            "device": "A40", "git_sha": git,
            "command": ("bash tools/pod/gpu_run.sh python3 "
                        "eval/exp_mosaic_res/run_res.py --no-artifacts"),
            "ms_per_win_note": ("mosaic embed ms/window = batch mosaic-encode time "
                                "attributed equally across configs; batch-shared, "
                                "so use canvas_px as the deterministic cost proxy"),
            "cost_total": {k: round(v, 1) for k, v in cost.items()},
            "baselines": {
                "eg2_window_embedding_loco": 0.848,
                "feature26_loco": 0.810,
                "source": "pod task brief / E2 cached reference (2fps EG2 window mean)"}}

    def row_mosaic(tag, r, c, tp, frames):
        rr = res[tag]
        return {"tag": tag, "frames": frames, "tile_px": tp, "grid": [r, c],
                "canvas_px": canvas_px(r, c, tp),
                "auroc": rr["auroc"], "ci95": rr["ci95"],
                "bal_acc": rr["bal_acc"],
                "mosaic_embed_ms_per_win": round(per[tag] / n, 1)}

    # ---- MR.01 tile-resolution ladder (grid 3x3 fixed) ----
    mr01_rows = {}
    for tp in TILE_LADDER:
        d = row_mosaic(f"m_3x3_t{tp}", 3, 3, tp, 9)
        d["vs_meanframes_N9"] = round(d["auroc"] - res["mean_N9"]["auroc"], 4)
        mr01_rows[f"tile{tp}"] = d
    best = max(mr01_rows.values(), key=lambda x: x["auroc"])
    within = [v for v in mr01_rows.values()
              if best["auroc"] - v["auroc"] <= 0.02]
    smallest = min(within, key=lambda x: x["tile_px"])
    mr01 = dict(head, experiment="MR.01",
                title="tile-resolution ladder (grid 3x3, timestamps on)",
                variant="mosaic embedding; tile px 224/160/112/96/64",
                results=mr01_rows, mean_N9=res["mean_N9"],
                best_tile_px=best["tile_px"], best_auroc=best["auroc"],
                **{"smallest_within_0.02_auroc": smallest["tile_px"]},
                verdict=("adopt small tiles" if smallest["tile_px"] < 224
                         else "keep 224 px"),
                headline=best)

    # ---- MR.02 density ladder (tile 112 fixed) ----
    mr02_rows = {}
    for (r, c) in DENSITY_GRIDS:
        frames = r * c
        tag = f"m_3x3_t112" if (r, c) == (3, 3) else f"m_{r}x{c}_t112"
        d = row_mosaic(tag, r, c, DENSITY_TILE, frames)
        mk = f"mean_N{frames}"
        d["mean_frames_auroc"] = res[mk]["auroc"]
        d["mean_frames_ci95"] = res[mk]["ci95"]
        d["mosaic_minus_mean"] = round(d["auroc"] - res[mk]["auroc"], 4)
        mr02_rows[f"{r}x{c}"] = d
    mr02 = dict(head, experiment="MR.02",
                title="density ladder (tile 112, timestamps on) vs mean-of-frames",
                variant="mosaic embedding; grids 2x2..6x6 at tile 112",
                results=mr02_rows,
                verdict=("mosaic never beats mean-of-frames at equal frames "
                         "(E2 gap not recovered by lower-res density)"),
                headline=max(mr02_rows.values(), key=lambda x: x["auroc"]))

    # ---- MR.03 equal-canvas duel + ablation ----
    duel = {}
    for tag, r, c, tp in CANVAS_DUEL:
        duel[tag] = row_mosaic(tag, r, c, tp, r * c)
    a224, a4168 = duel["m_3x3_t224"], duel["m_4x4_t168"]
    abl = {"m_4x4_t112": row_mosaic("m_4x4_t112", 4, 4, 112, 16),
           "m_4x4_t112_nots": row_mosaic("m_4x4_t112_nots", 4, 4, 112, 16),
           "m_4x4_t112_shuf": row_mosaic("m_4x4_t112_shuf", 4, 4, 112, 16),
           "mean_N16": res["mean_N16"]}
    mr03 = dict(head, experiment="MR.03",
                title="equal-canvas duel (9x224px vs 16x168px) + ablation",
                variant=("equal canvas ~462-465k px: 3x3 tile224 (9 frames) vs "
                         "4x4 tile168 (16 frames); ablation at 4x4 tile112"),
                duel=duel,
                duel_delta_auroc_more_smaller=round(a4168["auroc"] - a224["auroc"], 4),
                duel_delta_ms=round(a4168["mosaic_embed_ms_per_win"] -
                                    a224["mosaic_embed_ms_per_win"], 1),
                ablation=abl,
                ablation_delta_timestamps=round(
                    abl["m_4x4_t112"]["auroc"] - abl["m_4x4_t112_nots"]["auroc"], 4),
                ablation_delta_shuffle=round(
                    abl["m_4x4_t112"]["auroc"] - abl["m_4x4_t112_shuf"]["auroc"], 4),
                verdict=("duel: more smaller tiles "
                         + ("win" if a4168["auroc"] >= a224["auroc"] else "lose")
                         + " at equal canvas; mosaic still below mean-of-frames"))

    for eid, d in (("MR.01", mr01), ("MR.02", mr02), ("MR.03", mr03)):
        json.dump(d, open(os.path.join(R, f"{eid}.json"), "w"), indent=1)

    # ---- console / markdown table ----
    print("| combo | frames | tile px | canvas px | AUROC [CI95] | ms/win | vs control | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in mr01_rows.items():
        print(f"| MR.01 3x3 {k} | 9 | {v['tile_px']} | {v['canvas_px']} | "
              f"{v['auroc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] | "
              f"{v['mosaic_embed_ms_per_win']:.1f} | mean9 {v['vs_meanframes_N9']:+.3f} |")
    for k, v in mr02_rows.items():
        print(f"| MR.02 {k} tile112 | {v['frames']} | 112 | {v['canvas_px']} | "
              f"{v['auroc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] | "
              f"{v['mosaic_embed_ms_per_win']:.1f} | mean {v['mosaic_minus_mean']:+.3f} |")
    for k, v in duel.items():
        print(f"| MR.03 duel {k} | {v['frames']} | {v['tile_px']} | {v['canvas_px']} | "
              f"{v['auroc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] | "
              f"{v['mosaic_embed_ms_per_win']:.1f} | |")
    for k, v in abl.items():
        if "auroc" in v and "tile_px" in v:
            print(f"| MR.03 abl {k} | 16 | 112 | {canvas_px(4,4,112)} | "
                  f"{v['auroc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] | "
                  f"{v['mosaic_embed_ms_per_win']:.1f} | |")
        elif "auroc" in v:
            print(f"| MR.03 control {k} | 16 | - | - | "
                  f"{v['auroc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] | - | |")
    print("\nMR.01 smallest tile within 0.02 of best:",
          mr01["smallest_within_0.02_auroc"], "px (best", best["tile_px"],
          f"{best['auroc']:.3f})")
    print("MR.03 equal-canvas delta (16x168 - 9x224):",
          mr03["duel_delta_auroc_more_smaller"], "AUROC,",
          mr03["duel_delta_ms"], "ms/win")
    print("MR.03 ablation d(ts)=", mr03["ablation_delta_timestamps"],
          "d(shuffle)=", mr03["ablation_delta_shuffle"])
    print("mean-of-frames:", {k.split('_')[-1]: res[k]["auroc"]
                              for k in res if k.startswith("mean_")})
    print("wrote MR.01..03 to", R)


if __name__ == "__main__":
    main()
