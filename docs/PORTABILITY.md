# Portability: absolute paths in eval/gpu/

The GPU benchmark scripts in `eval/gpu/` still hard-code absolute paths from the
author's machine (`/home/parshu/...`). They are listed here and were **not**
edited. Each path needs to become a repo-relative path, or an env override with
the same default, before these scripts run elsewhere.

Total: 33 lines in 12 files.

| file:line | text |
|---|---|
| `eval/gpu/ablate.py:14` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
| `eval/gpu/ablate.py:70` | `"regenerate": "/home/parshu/envs/jadal/bin/python eval/gpu/ablate.py",` |
| `eval/gpu/cascade.py:4` | `Run: /home/parshu/envs/jadal/bin/python eval/gpu/cascade.py` |
| `eval/gpu/cascade.py:30` | `SPEC = "/home/parshu/projects/jadal-qa-wt/spectral_data"` |
| `eval/gpu/determinism.py:14` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
| `eval/gpu/e2e.py:5` | `flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \` |
| `eval/gpu/e2e.py:6` | `/home/parshu/envs/jadal/bin/python eval/gpu/e2e.py [nwin] [batch]` |
| `eval/gpu/e2e.py:18` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
| `eval/gpu/e2e.py:129` | `"regenerate": ("flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock "` |
| `eval/gpu/e2e.py:130` | `"/home/parshu/envs/jadal/bin/python eval/gpu/e2e.py"),` |
| `eval/gpu/factory.py:3` | `Corpus: /home/parshu/projects/awing/data/ego10k_scale/clips/factory_*/worker_*/*.mp4` |
| `eval/gpu/factory.py:25` | `CLIP_GLOB = "/home/parshu/projects/awing/data/ego10k_scale/clips/factory_*/worker_*/*.mp4"` |
| `eval/gpu/factory.py:59` | `"regenerate": "flock ... /home/parshu/envs/jadal/bin/python eval/gpu/factory.py --sweep"}` |
| `eval/gpu/factory.py:155` | `"regenerate": "flock ... /home/parshu/envs/jadal/bin/python eval/gpu/factory.py --sustain",` |
| `eval/gpu/factory.py:167` | `json.load(open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))["items"]}` |
| `eval/gpu/holdc.py:6` | `flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \` |
| `eval/gpu/holdc.py:7` | `/home/parshu/envs/jadal/bin/python eval/gpu/holdc.py` |
| `eval/gpu/holdc.py:22` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
| `eval/gpu/microbench.py:4` | `flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \` |
| `eval/gpu/microbench.py:5` | `/home/parshu/envs/jadal/bin/python eval/gpu/microbench.py` |
| `eval/gpu/microbench.py:22` | `"/home/parshu/projects/awing/data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4",` |
| `eval/gpu/microbench.py:388` | `"regenerate": "flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock "` |
| `eval/gpu/microbench.py:389` | `"/home/parshu/envs/jadal/bin/python eval/gpu/microbench.py [--only memcpy,kernels\|decode]",` |
| `eval/gpu/nvdec_probe.py:11` | `CLIP = "/home/parshu/projects/awing/data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"` |
| `eval/gpu/nvdec_probe.py:55` | `"regenerate": "flock ... /home/parshu/envs/jadal/bin/python eval/gpu/nvdec_probe.py"},` |
| `eval/gpu/run_features.py:4` | `/home/parshu/envs/jadal/bin/python eval/gpu/run_features.py` |
| `eval/gpu/run_features.py:19` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
| `eval/gpu/run_features.py:20` | `SYNTH = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_synth.json"` |
| `eval/gpu/stream.py:22` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
| `eval/gpu/stream.py:157` | `"regenerate": "/home/parshu/envs/jadal/bin/python eval/gpu/stream.py",` |
| `eval/gpu/telemetry.py:1` | `"""Join benchmark windows to /home/parshu/projects/jadal-qa-wt/logs/gpu_telemetry.csv.` |
| `eval/gpu/telemetry.py:12` | `CSV = "/home/parshu/projects/jadal-qa-wt/logs/gpu_telemetry.csv"` |
| `eval/gpu/throughput.py:25` | `GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"` |
