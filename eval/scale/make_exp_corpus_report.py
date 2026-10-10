"""Assemble the D1 corpus-expansion report from the download artifacts.

Reads provenance.jsonl + download_verify_more.json + selection_more.json +
manifest_scale_ext.json + manifest_scale_all.json and the pre-existing
selection/manifest, then writes:
  eval/results/exp_corpus/D1.01.json
  docs/reports/exp_corpus.md
Disk free (before/after) is passed via env DISK_BEFORE_GB / DISK_AFTER_GB.
"""
import json
import os
from datetime import datetime, timezone

BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
EC = "eval/results/exp_corpus"
PROV = f"{EC}/provenance.jsonl"
VERIFY = f"{EC}/download_verify_more.json"
SEL = f"{EC}/selection_more.json"
MAN_EXT = "queues/manifest_scale_ext.json"
MAN_ALL = "queues/manifest_scale_all.json"
EX_SEL = f"{BUNDLE}/queues/selection_small.json"
EX_MAN = f"{BUNDLE}/queues/manifest_scale.json"
OUT_JSON = f"{EC}/D1.01.json"
OUT_MD = "docs/reports/exp_corpus.md"


def load_prov():
    rows = []
    if os.path.exists(PROV):
        for line in open(PROV):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    prov = load_prov()
    verify = json.load(open(VERIFY)) if os.path.exists(VERIFY) else {}
    sel = json.load(open(SEL)) if os.path.exists(SEL) else {}
    man_ext = json.load(open(MAN_EXT)) if os.path.exists(MAN_EXT) else {}
    man_all = json.load(open(MAN_ALL)) if os.path.exists(MAN_ALL) else {}
    ex_sel = json.load(open(EX_SEL))
    ex_man = json.load(open(EX_MAN))

    ex_facs = sorted({f["path"].split("/")[0] for f in ex_sel["files"]})
    new_facs = sorted({r["tar"].split("/")[0] for r in prov})
    gained = sorted(set(new_facs) - set(ex_facs))
    new_clips = sum(r.get("n_clips", 0) for r in prov)
    new_bytes = sum(r.get("bytes", 0) for r in prov)
    new_hours = round(man_ext.get("total_hours", 0.0), 3)
    n_bad = sum(r.get("n_bad", 0) for r in prov)

    total_facs = sorted(set(ex_facs) | set(new_facs))
    total_clips = ex_man.get("n_clips", 0) + new_clips
    total_hours = round(ex_man.get("total_hours", 0.0) + new_hours, 3)
    total_windows = ex_man.get("n_windows", 0) + man_ext.get("n_windows", 0)

    disk_before = float(os.environ.get("DISK_BEFORE_GB", "0"))
    disk_after = float(os.environ.get("DISK_AFTER_GB", "0"))

    result = {
        "task": "D1 corpus expansion (Egocentric-10K, gated HF dataset)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "builddotai/Egocentric-10K",
        "selection_rule": sel.get("rule"),
        "selection_seed": sel.get("seed"),
        "new": {
            "n_tars": len(prov),
            "n_tars_planned": sel.get("n_files"),
            "n_clips": new_clips,
            "n_windows": man_ext.get("n_windows", 0),
            "hours": new_hours,
            "bytes": new_bytes,
            "GB": round(new_bytes / 1e9, 2),
            "n_factories": len(new_facs),
            "n_new_factories": len(gained),
            "factories_gained": gained,
            "n_workers": len({"/".join(r["tar"].split("/")[i] for i in (0, 2))
                              for r in prov}),
            "ffprobe_failures": n_bad,
        },
        "total": {
            "n_clips": total_clips,
            "n_windows": total_windows,
            "hours": total_hours,
            "n_factories": len(total_facs),
            "factories": total_facs,
        },
        "download": {
            "MB_per_s": verify.get("MB_per_s"),
            "elapsed_s": verify.get("elapsed_s"),
            "elapsed_min": round(verify.get("elapsed_s", 0) / 60, 1),
            "retries": verify.get("n_retries", 0),
            "failures": verify.get("n_fail", 0),
            "stopped_early": verify.get("stopped_early", False),
        },
        "disk": {
            "before_free_GB": disk_before,
            "after_free_GB": disk_after,
            "min_free_floor_GB": 15.0,
        },
        "artifacts": {
            "provenance": PROV,
            "manifest_ext": MAN_EXT,
            "manifest_all": MAN_ALL,
            "selection": SEL,
            "verify": VERIFY,
        },
    }
    json.dump(result, open(OUT_JSON, "w"), indent=1)

    md = f"""# D1 — Egocentric-10K corpus expansion

_Generated {result['created_utc']} by `eval/scale/make_exp_corpus_report.py`._

Dataset: `{result['dataset']}` (gated). Selection rule: {result['selection_rule']}
(seed {result['selection_seed']}). Media lives under
`$JADAL_POD_ROOT/jadal_pod_bundle/data/ego10k_scale/` and is **not** committed.

## New footage

| metric | value |
|---|---|
| tars downloaded (planned) | {result['new']['n_tars']} / {result['new']['n_tars_planned']} |
| new clips | {result['new']['n_clips']} |
| new 10 s windows | {result['new']['n_windows']} |
| new footage hours | {result['new']['hours']} |
| new tars bytes | {result['new']['GB']} GB |
| factories in new set | {result['new']['n_factories']} |
| **NEW factories gained** | **{result['new']['n_new_factories']}** |
| new workers | {result['new']['n_workers']} |
| ffprobe failures | {result['new']['ffprobe_failures']} |

Factories gained: {', '.join(gained) if gained else '(none)'}

## Total corpus (existing + new)

| metric | value |
|---|---|
| clips | {result['total']['n_clips']} |
| 10 s windows | {result['total']['n_windows']} |
| footage hours | {result['total']['hours']} |
| factories | {result['total']['n_factories']} |

## Download health

| metric | value |
|---|---|
| throughput | {result['download']['MB_per_s']} MB/s |
| elapsed | {result['download']['elapsed_min']} min |
| retries | {result['download']['retries']} |
| failures | {result['download']['failures']} |
| stopped early (disk guard) | {result['download']['stopped_early']} |
| disk free before | {result['disk']['before_free_GB']} GB |
| disk free after | {result['disk']['after_free_GB']} GB |
| free floor enforced | {result['disk']['min_free_floor_GB']} GB |

## Artifacts

- provenance: `{PROV}`
- extended manifest (new clips): `{MAN_EXT}`
- full manifest (existing + new): `{MAN_ALL}`
- selection: `{SEL}`
- verify summary: `{VERIFY}`
- metrics: `{OUT_JSON}`

## Resume

The download is resumable: tars already present in `provenance.jsonl` are
skipped, and partial HF downloads resume in place. To continue, re-run
`HF_HUB_DISABLE_XET=1 python eval/scale/dl_more.py` from the worktree, then
`python eval/scale/make_manifest_ext.py` and this report generator.
"""
    open(OUT_MD, "w").write(md)
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    print(json.dumps(result["new"], indent=1))
    print(json.dumps(result["total"], indent=1))


if __name__ == "__main__":
    main()
