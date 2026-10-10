# Frozen pre-registered artifacts

`ensemble_v1.prereg.json` is the **byte-exact** frozen pre-registered model config for
the out-of-domain (OOD) evaluation. Do not edit it: editing it invalidates the
pre-registration.

- Recorded (authority) sha256: `4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31`
- Recorded in: `jadal_pod_bundle/SHA256SUMS.txt` line 1
  (`./caches/ensemble/ensemble_v1.json`), `jadal_pod_bundle/caches/ensemble/ood_scores.json`
  (`config_sha256`), `configs/ensemble_v1.json.sha256`, and truncated in
  `jadal_pod_bundle/docs/FACTS.md`.
- Recovered byte-exactly from git: blob `0ede3b83d93ce23dc5b68cd20fb7de0af703ad27`
  = `configs/ensemble_v1.json` at freeze commit `b21e3af`.
- Verified: `sha256sum configs/frozen/ensemble_v1.prereg.json` == the recorded hash.

Why this copy exists: the copies on disk elsewhere were **path-remapped after the
freeze** (`embed_command`, `pixel_command`, `views.eg2.embed_model_dir`) and therefore
no longer match the recorded hash:

| copy | observed sha256 | matches pre-registration |
|---|---|---|
| `configs/frozen/ensemble_v1.prereg.json` (this file) | `4da1068a78d5d150…` | yes |
| `configs/ensemble_v1.json` (repo, pod-remapped) | `d62a95a473a31085…` | no |
| `jadal_pod_bundle/caches/ensemble/ensemble_v1.json` (pod-remapped) | `cf7fd7d34df2c3ca…` | no |

The runner records observed vs recorded hashes separately and refuses to score on a
mismatch; it never substitutes the observed hash for the recorded one.

Run: `bash tools/pod/run_ood_frozen.sh` (see `docs/reports/ood_preregistration.md`).
