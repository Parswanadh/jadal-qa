"""E3.05 decision synthesis. Regen: same run_e3.py command; reads the four JSONs."""
import json, os
D = "/home/parshu/projects/jadal-qa-wt/exp-temporal/eval/results/exp_temporal"
r01 = json.load(open(os.path.join(D, "E3.01_policies.json")))
r02 = json.load(open(os.path.join(D, "E3.02_ood_position.json")))
r03 = json.load(open(os.path.join(D, "E3.03_early_exit.json")))
r04 = json.load(open(os.path.join(D, "E3.04_compute.json")))
P = r01["per_policy"]

def row(p):
    s = P[p]
    return dict(mean_err_L=round(s["mean_err_L"], 3), ci_L=[round(x, 3) for x in s["ci_L"]],
                agree_L=s["agree_L"], mean_delay_windows=round(s["mean_delay_L"], 2),
                miss_L=s["miss_L"], frac=round(s["mean_frac"], 2))

# tolerance: errL<=0.06, 7/7 verdicts, delay<=1 window
cands = {p: row(p) for p in ("every-2", "every-3", "every-4", "cusum-default",
                             "skip-first-120s", "random-1/2-s7")}
strict = [p for p in ("every-2", "every-3", "every-4") if P[p]["mean_err_L"] <= 0.06
          and P[p]["agree_L"] == 1.0 and P[p]["mean_delay_L"] <= 1.0]
res = dict(
    n_clips=7, n_windows=150, n_ood=60, n_ood_bad=9,
    label_source="human gold (IndEgo) + human OOD labels; synthesis of E3.01-E3.04",
    protocol="tolerance selection on label-oracle clip error (isolates sampling from model error); "
             "model error stated separately because full-coverage OOF estimate already errs 0.167/4-7 verdicts",
    candidates=cands,
    strict_tolerance=dict(rule="errL<=0.06 & 7/7 verdicts & delay<=1 window", pass_list=strict),
    decisions=[
        "ADOPT IndEgo-like (480p/10fps assembly-logistics): probe every 2nd window (51% windows, "
        "clip-fraction err 0.049 [0.019,0.087], 7/7 verdicts, delay 0.3 windows) + early exit after "
        "4 s observed (AUROC 0.860 vs 0.848 full). Combined ~0.20 footage processed (~0.9 h per 4.4 h, x5).",
        "BUDGET OPTION: every 3rd window (35%, err 0.117, 6/7 verdicts, delay 1.3 windows) only if "
        "half-cost is refused; every-4/6/8 miss bad stretches (miss_L up to 3/7).",
        "DROP CUSUM-as-sampler: default (d=0.1,h=0.5) never fires on flat OOF streams (identical to "
        "every-4 grid); LOCO-tuned (mostly h=0.3) matches grid error (LOCO errM 0.214 vs every-4 0.209) "
        "at equal cost. Keep CUSUM only for physical-collapse early-reject (lab scope).",
        "DROP skip-first-M for IndEgo: err 0.088->0.418, verdicts fall to 5/7; start is the cleaner part.",
        "OOD Egocentric-10K factory: NO skip rule (MW p=0.42 raw/0.47 normalised, n_bad=9) and NO "
        "certified sparse policy — transfer fails (frozen models 0.4-0.53) so sampling policy is moot "
        "until the scorer works. Re-run position test at n_bad~24 (labels2).",
        "Sampling is NOT the bottleneck: model clip-fraction err 0.167 at FULL coverage, 4/7 verdicts. "
        "Fix the scorer (E5) before squeezing sampling further.",
    ],
    thin_evidence=["7 clips only: CIs span ~0.07-0.18 on errL; pallet2 n=6",
                   "OOD position test n_bad=9: null result is absence of evidence, not evidence of absence",
                   "gold windows are a sparse sample of each clip: 'full coverage' = all gold windows, "
                   "not exhaustive video; true continuous-clip errors need dense labels",
                   "early-exit curve is IndEgo-only; OOD curve needs a working OOD scorer"],
    command="PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_temporal/run_e3.py --out eval/results/exp_temporal")
res["adopted_indego"] = ("probe every 2nd window + 4 s early exit; "
                        "budget option every 3rd; CUSUM-as-sampler dropped; no skip rule")
json.dump(res, open(os.path.join(D, "E3.05_decision.json"), "w"), indent=1)
print("wrote E3.05_decision.json; strict-pass:", strict)
