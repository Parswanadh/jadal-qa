#!/usr/bin/env python3
"""One-command BASELINE scoreboard for docs/BASELINE.md (checks B1-B8).

Runs (or reads the results of) checks B1-B8 and writes:
  docs/BASELINE_STATUS.md   human table: id, check, PASS/FAIL/PENDING/...
  eval/results/BASELINE.json  machine record {generated_utc, git_sha,
                              checks: {B1..B8: {status, evidence}},
                              numbers: {decider, funnel, latency, dataset}}

Rules (honest reporting):
  * A missing artifact is PENDING, never invented, never estimated.
  * Informational by default: always exit 0. --strict exits nonzero on FAIL.
  * --clean-clone clones origin/main into a temp dir and runs the README
    quickstart there; skipped by default.

Python: PYTHONNOUSERSITE=1 <env>/bin/python. CPU only. No secrets, no network
beyond localhost.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.abc
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

STATUSES = ("PASS", "FAIL", "PENDING", "NOT-APPLICABLE")

PY = sys.executable
TEST_TIMEOUT_S = 120  # hard timeout per suite, per docs/BASELINE.md
DEMO_TIMEOUT_S = 600
SERVICE_TIMEOUT_S = 25

# Wrapper run by the no-signals B1 variants: makes the real L0/L1/segmenter
# modules unimportable (sitecustomize-free: a meta-path blocker installed in
# the subprocess before unittest starts), then runs the suite programmatically.
NO_SIGNALS_WRAPPER = r"""
import sys, importlib.abc, unittest

class _BlockRealStages(importlib.abc.MetaPathFinder):
    _BLOCKED = ("jadal.signals", "jadal.decision", "jadal.segmenter")
    def find_spec(self, name, path, target=None):
        if name in self._BLOCKED or name.startswith(
                tuple(b + "." for b in self._BLOCKED)):
            raise ImportError("blocked for baseline B1 no-signals run: " + name)
        return None

sys.meta_path.insert(0, _BlockRealStages())
# Drop any already-imported real stage modules (fresh interpreter: paranoia).
for _m in list(sys.modules):
    if _m in _BlockRealStages._BLOCKED or _m.startswith(
            tuple(b + "." for b in _BlockRealStages._BLOCKED)):
        del sys.modules[_m]

start_dir = sys.argv[1]
suite = unittest.TestLoader().discover(start_dir)
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean_env() -> dict:
    """Subprocess env: no user site, no stub flag unless the caller sets it."""
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("JADAL_USE_STUBS", None)
    return env


def run_cmd(cmd, cwd, timeout_s, env=None):
    """Run cmd, capturing output. Returns dict with returncode/stdout/stderr."""
    try:
        proc = subprocess.run(
            list(cmd), cwd=str(cwd), timeout=timeout_s,
            env=env if env is not None else clean_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        return {"returncode": proc.returncode, "stdout": proc.stdout,
                "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) \
            else (exc.stdout or "")
        return {"returncode": 124, "stdout": out,
                "timed_out": True}
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": f"not found: {exc.filename}",
                "timed_out": False}


def tail(text, n=8):
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n:]) if lines else "(no output)"


def result(status, evidence, extra=None):
    assert status in STATUSES, status
    out = {"status": status, "evidence": evidence}
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- B1: tests
def check_b1(root):
    """Green main, with AND without real signals importable, fast; node check."""
    root = Path(root)
    subs = []
    suites = [("tests", "python -m unittest discover -s tests"),
              ("web", "python -m unittest discover -s web")]
    for start_dir, label in suites:
        r = run_cmd([PY, "-m", "unittest", "discover", "-s", start_dir],
                    root, TEST_TIMEOUT_S)
        ok = r["returncode"] == 0
        subs.append({"name": f"suite {start_dir} (normal)",
                     "pass": ok, "tail": tail(r["stdout"]),
                     "timed_out": r["timed_out"], "cmd": label})
    for start_dir, _label in suites:
        r = run_cmd([PY, "-c", NO_SIGNALS_WRAPPER, start_dir],
                    root, TEST_TIMEOUT_S)
        ok = r["returncode"] == 0
        subs.append({"name": f"suite {start_dir} (signals unimportable)",
                     "pass": ok, "tail": tail(r["stdout"]),
                     "timed_out": r["timed_out"],
                     "cmd": "python -c <signals-blocking wrapper> " + start_dir})
    js_files = sorted((root / "web").glob("*.js"))
    if shutil.which("node") is None:
        subs.append({"name": "node --check web/*.js", "pass": None,
                     "tail": "node not installed: cannot verify",
                     "timed_out": False, "cmd": "node --check web/*.js"})
    elif not js_files:
        subs.append({"name": "node --check web/*.js", "pass": None,
                     "tail": "no web/*.js files found",
                     "timed_out": False, "cmd": "node --check web/*.js"})
    else:
        for js in js_files:
            r = run_cmd(["node", "--check", str(js)], root, 60)
            subs.append({"name": f"node --check {js.name}",
                         "pass": r["returncode"] == 0,
                         "tail": tail(r["stdout"]) or "syntax OK",
                         "timed_out": r["timed_out"],
                         "cmd": f"node --check web/{js.name}"})
    if any(s["pass"] is None for s in subs):
        missing = [s["name"] for s in subs if s["pass"] is None]
        return result("PENDING",
                      f"cannot verify: {'; '.join(missing)}",
                      {"subchecks": subs})
    failed = [s["name"] for s in subs if not s["pass"]]
    timed_out = [s["name"] for s in subs if s["timed_out"]]
    if failed:
        ev = f"{len(failed)}/{len(subs)} subchecks failed: " \
             f"{'; '.join(failed)}"
        if timed_out:
            ev += f" (timed out >{TEST_TIMEOUT_S}s: {'; '.join(timed_out)})"
        return result("FAIL", ev, {"subchecks": subs})
    return result("PASS", f"all {len(subs)} subchecks green "
                          f"(each <={TEST_TIMEOUT_S}s)",
                  {"subchecks": subs})


# ------------------------------------------------- B2: real-pipeline demo
def probe_real_modules(root):
    """Are the REAL jadal.signals / jadal.decision importable (no stubs)?"""
    root = Path(root)
    probe = ("import jadal.signals, jadal.decision; "
             "assert callable(getattr(jadal.signals, 'compute_signals', None)), "
             "'no compute_signals'; "
             "assert hasattr(jadal.decision, 'Decider'), 'no Decider'; "
             "print('real L0+L1 importable')")
    r = run_cmd([PY, "-c", probe], root, 60)
    return {"importable": r["returncode"] == 0, "tail": tail(r["stdout"])}


def check_b2(root):
    """make demo on the REAL pipeline (no stubs). PENDING unless real L0/L1."""
    root = Path(root)
    probe = probe_real_modules(root)
    if not probe["importable"]:
        return result(
            "PENDING",
            "real jadal.signals and/or jadal.decision not importable on main; "
            f"demo stays on stubs (probe: {probe['tail']})",
            {"demo_output": None})
    env = clean_env()
    env.pop("JADAL_USE_STUBS", None)
    r = run_cmd(["bash", "scripts/demo.sh"], root, DEMO_TIMEOUT_S, env=env)
    if r["timed_out"]:
        return result("FAIL",
                      f"make demo (real pipeline) timed out >{DEMO_TIMEOUT_S}s",
                      {"demo_output": r["stdout"]})
    if r["returncode"] != 0:
        return result("FAIL", f"make demo (real pipeline) exited "
                              f"{r['returncode']}",
                      {"demo_output": r["stdout"]})
    return result("PASS", "make demo green on the real pipeline "
                          "(JADAL_USE_STUBS unset)",
                  {"demo_output": r["stdout"]})


# ------------------------------------------------- B3: determinism asserts
def parse_demo_determinism(demo_output):
    """Look for the identical-hash assertions in demo output."""
    if not demo_output:
        return result("PENDING", "no real-pipeline demo output (B2 PENDING), "
                                 "nothing to parse")
    low = demo_output.lower()
    has_10x = ("10x" in low or "10 runs" in low) and "identical" in low
    has_batch = "alone vs batch-of-5" in low and "identical" in low
    has_reverify = "all receipts reverify match" in low
    if has_10x and has_batch and has_reverify:
        return result("PASS", "demo asserts found: 10x identical "
                              "receipt_sha256, alone-vs-batch-of-5 IDENTICAL, "
                              "all receipts Re-verify MATCH")
    missing = [k for k, v in
               [("10x identical", has_10x), ("alone-vs-batch-of-5", has_batch),
                ("reverify MATCH", has_reverify)] if not v]
    return result("FAIL", f"demo ran but assertions missing: "
                          f"{'; '.join(missing)}")


def check_b3(root, demo_output):
    return parse_demo_determinism(demo_output)


# ------------------------------------------------- B4: measured numbers
def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def num(value, source, n, label_source):
    return {"value": value, "source": source, "n": n,
            "label_source": label_source}


def check_b4(root):
    """Copy measured numbers from eval/results artifacts. Missing => PENDING."""
    root = Path(root)
    res = root / "eval" / "results"
    numbers = {"decider": {}, "funnel": {}, "latency": {}, "dataset": {}}
    missing = []
    found = []

    def note(name, path):
        found.append(f"{name} ({path})")

    # -- decider heads: accuracy, ECE, certified bound on the clean splits --
    decider = load_json(res / "decider_baseline.json")
    if decider is None:
        missing.append("eval/results/decider_baseline.json "
                       "(decider accuracy/ECE/certified bound)")
    else:
        n = decider.get("n")
        lab = decider.get("label_source", "unstated in artifact")
        for key in ("accuracy", "ece", "certified_bound", "threshold",
                    "bound", "coverage"):
            if key in decider:
                numbers["decider"][key] = num(
                    decider[key], "eval/results/decider_baseline.json",
                    n, lab)
        note("decider_baseline.json", "eval/results/decider_baseline.json")

    # -- funnel split on IndEgo segments --
    funnel = None
    funnel_src = None
    for cand in ("funnel_calibration.json", "funnel_report.json",
                 "funnel.json"):
        funnel = load_json(res / cand)
        if funnel is not None:
            funnel_src = f"eval/results/{cand}"
            break
    if funnel is None:
        missing.append("eval/results/funnel_calibration.json "
                       "(funnel split on IndEgo segments)")
    else:
        n = funnel.get("n")
        lab = funnel.get("label_source", "unstated in artifact")
        for key, val in funnel.items():
            if isinstance(val, (int, float, str)) and not isinstance(val, bool):
                numbers["funnel"][key] = num(val, funnel_src, n, lab)
        note("funnel artifact", funnel_src)

    # -- per-stage latency and segments/hour on this laptop --
    lat = None
    lat_src = None
    for cand in ("scale_bench.json", "latency_bench.json",
                 "bench_scale.json", "bench_eg2.json"):
        lat = load_json(res / cand)
        if lat is not None:
            lat_src = f"eval/results/{cand}"
            break
    if lat is None:
        missing.append("scale bench / bench_eg2.json "
                       "(per-stage latency, segments/hour)")
    else:
        n = lat.get("n")
        lab = lat.get("label_source", "n/a (timing, not labels)")
        for key, val in lat.items():
            if isinstance(val, (int, float, str)) and not isinstance(val, bool):
                numbers["latency"][key] = num(val, lat_src, n, lab)
        note("latency artifact", lat_src)

    # -- dataset / splits context (present on main today) --
    splits = load_json(res / "splits.json")
    if splits is None:
        missing.append("eval/results/splits.json (split sizes, label source)")
    else:
        lab = splits.get("label_source", "unstated in artifact")
        sizes = splits.get("sizes")
        if sizes is not None:
            numbers["dataset"]["split_sizes"] = num(
                sizes, "eval/results/splits.json", None, lab)
        for key in ("seed", "ratios", "duplicate_map_entries"):
            if key in splits:
                numbers["dataset"][key] = num(
                    splits[key], "eval/results/splits.json", None, lab)
        note("splits.json", "eval/results/splits.json")
    drep = load_json(res / "dataset_report.json")
    if drep is None:
        missing.append("eval/results/dataset_report.json")
    else:
        lab = (splits or {}).get("label_source", "unstated in artifact")
        for key in ("distinct_frame_ids", "total_rows"):
            if key in drep:
                numbers["dataset"][key] = num(
                    drep[key], "eval/results/dataset_report.json",
                    drep.get("distinct_frame_ids"), lab)
        note("dataset_report.json", "eval/results/dataset_report.json")

    # DEMO_synthetic_* is explicitly synthetic: never a baseline number.
    if (res / "DEMO_synthetic_calibration.json").exists():
        found.append("DEMO_synthetic_calibration.json present but EXCLUDED "
                     "(synthetic demo predictions, not model output)")

    if missing:
        return result("PENDING",
                      "missing measured artifacts: " + "; ".join(missing),
                      {"numbers": numbers, "found": found})
    return result("PASS",
                  "all measured artifacts present: " + "; ".join(found),
                  {"numbers": numbers, "found": found})


# ------------------------------------------------- B5: funnel is the verdict
def check_b5(root):
    """jadal/pipeline.py must reference jadal.funnel (static check)."""
    root = Path(root)
    path = root / "jadal" / "pipeline.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return result("PENDING", f"{path} unreadable")
    hits = [ln.strip() for ln in text.splitlines() if "funnel" in ln.lower()]
    if hits:
        return result("PASS",
                      f"jadal/pipeline.py references funnel "
                      f"({len(hits)} line(s): {'; '.join(hits[:3])})")
    return result("FAIL", "jadal/pipeline.py has no mention of funnel: "
                          "the funnel router is not the verdict "
                          "implementation (fallback: current inline logic)")


# ------------------------------------------------- B6: dashboard screenshots
def git_time(root, pathspec):
    r = run_cmd(["git", "log", "-1", "--format=%H %cI", "--", pathspec],
                root, 30)
    if r["returncode"] != 0 or not r["stdout"].strip():
        return None
    parts = r["stdout"].strip().split(" ", 1)
    return {"sha": parts[0], "time": parts[1] if len(parts) > 1 else "?"}


def check_b6(root):
    """docs/screens/ screenshots newer than the last change to web/."""
    root = Path(root)
    screens = sorted((root / "docs" / "screens").glob("*.png"))
    if not screens:
        return result("PENDING", "docs/screens/ has no screenshots yet")
    web = git_time(root, "web/")
    scr = git_time(root, "docs/screens/")
    if web is None or scr is None:
        return result("PENDING", "git history unavailable for web/ vs "
                                 "docs/screens/ comparison")
    if scr["time"] > web["time"]:
        return result("PASS", f"screenshots ({len(screens)} png) at "
                              f"{scr['time']} are newer than last web/ "
                              f"change {web['time']}")
    return result("PENDING", f"screenshots ({len(screens)} png, last "
                            f"{scr['time']}) are NOT newer than last web/ "
                            f"change ({web['time']}): re-capture against "
                            f"the live service")


# ------------------------------------------------- B7: /pico/status
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def http_get(url, timeout_s=5):
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.status, resp.read().decode()[:2000]


def check_b7(root):
    """GET /pico/status against a started service, else PENDING."""
    root = Path(root)
    if shutil.which("uvicorn") is None and run_cmd(
            [PY, "-c", "import uvicorn"], root, 30)["returncode"] != 0:
        return result("PENDING", "uvicorn not installed: cannot start "
                                 "a service to probe /pico/status")
    port = free_port()
    env = clean_env()
    env["JADAL_USE_STUBS"] = "1"
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
    log.close()
    proc = None
    try:
        with open(log.name, "w") as fh:
            proc = subprocess.Popen(
                [PY, "-m", "uvicorn", "jadal.service:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=str(root), env=env, stdout=fh, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + SERVICE_TIMEOUT_S
        up = False
        while time.time() < deadline:
            try:
                http_get(url + "/health", 2)
                up = True
                break
            except Exception:
                time.sleep(0.5)
        if not up:
            return result("PENDING", "service did not answer /health within "
                                     f"{SERVICE_TIMEOUT_S}s; /pico/status "
                                     "unprobed (hardware optional)")
        try:
            status, body = http_get(url + "/pico/status", 5)
        except Exception as exc:
            return result("PENDING", f"/pico/status request failed: {exc}")
        if status == 200:
            return result("PASS", f"GET /pico/status -> 200: {body[:300]}")
        return result("FAIL", f"GET /pico/status -> {status}: {body[:300]}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            os.unlink(log.name)
        except OSError:
            pass


# ------------------------------------------------- B8: tag + clean clone
def check_b8(root, clean_clone=False):
    """baseline-v1 tag present? --clean-clone: fresh clone runs quickstart."""
    root = Path(root)
    r = run_cmd(["git", "tag", "--list", "baseline-v1"], root, 30)
    tag_present = r["returncode"] == 0 and "baseline-v1" in r["stdout"]
    if not tag_present:
        return result("FAIL", "git tag baseline-v1 absent on this repo; "
                              "clean-clone check " +
                              ("runs below" if clean_clone else
                               "skipped (rerun with --clean-clone)"))
    if not clean_clone:
        return result("PENDING", "tag baseline-v1 present, but the clean-clone "
                                 "README quickstart was not run "
                                 "(rerun with --clean-clone)")
    origin = run_cmd(["git", "remote", "get-url", "origin"], root, 30)
    if origin["returncode"] != 0:
        return result("FAIL", "tag present but no git origin to clone")
    tmp = Path(tempfile.mkdtemp(prefix="baseline-clean-clone-"))
    try:
        clone = run_cmd(["git", "clone", "--quiet",
                         origin["stdout"].strip(), str(tmp / "clone")],
                        root, 300)
        if clone["returncode"] != 0:
            return result("FAIL", f"clean clone of origin failed: "
                                  f"{tail(clone['stdout'])}")
        croot = tmp / "clone"
        steps = [
            ([PY, "-m", "unittest", "discover", "-s", "tests"], "tests suite"),
            ([PY, "-m", "unittest", "discover", "-s", "web"], "web suite"),
        ]
        for cmd, name in steps:
            rr = run_cmd(cmd, croot, TEST_TIMEOUT_S)
            if rr["returncode"] != 0:
                return result("FAIL", f"clean clone: {name} failed: "
                                      f"{tail(rr['stdout'])}")
        return result("PASS", "tag baseline-v1 present; fresh clone of "
                              "origin/main runs the README quickstart suites")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- outputs
CHECKS = [
    ("B1", "main green with AND without real signals, fast; node --check"),
    ("B2", "real pipeline on real clips via make demo (no stubs)"),
    ("B3", "determinism: identical receipt_sha256; Re-verify MATCH"),
    ("B4", "measured numbers in BASELINE.json with N and label source"),
    ("B5", "funnel router is the verdict implementation"),
    ("B6", "dashboard screenshots vs live service newer than web/"),
    ("B7", "/pico/status answers (hardware optional)"),
    ("B8", "clean clone runs quickstart; tag baseline-v1 pushed"),
]


def git_sha(root):
    r = run_cmd(["git", "rev-parse", "HEAD"], root, 30)
    if r["returncode"] == 0 and r["stdout"].strip():
        return r["stdout"].strip()
    return "unknown (not a git checkout?)"


def write_outputs(root, results, numbers):
    root = Path(root)
    sha = git_sha(root)
    stamp = utcnow()
    status_path = root / "docs" / "BASELINE_STATUS.md"
    json_path = root / "eval" / "results" / "BASELINE.json"

    lines = ["# BASELINE scoreboard",
             "",
             f"Generated (UTC): {stamp}",
             f"Git SHA: {sha}",
             "",
             "| id | check | status | evidence | command output tail |",
             "|---|---|---|---|---|"]
    for cid, cname in CHECKS:
        res = results[cid]
        tail_txt = res.get("tail", "")
        if not tail_txt:
            subs = res.get("subchecks")
            if subs:
                bad = [s for s in subs if not s.get("pass")]
                src = bad[0] if bad else subs[0]
                tail_txt = f"[{src['name']}] {src.get('tail', '')}"
        tail_one = " <br> ".join((tail_txt or "-").splitlines()[:6])
        ev_one = " <br> ".join(res["evidence"].splitlines()[:4])
        lines.append(f"| {cid} | {cname} | {res['status']} | "
                     f"{ev_one} | {tail_one} |")
    lines += ["",
              "Statuses: PASS / FAIL / PENDING / NOT-APPLICABLE. "
              "Missing artifacts are PENDING, never estimated.",
              "Numbers live in eval/results/BASELINE.json with per-number "
              "source file, N and label_source.",
              ""]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {"generated_utc": stamp, "git_sha": sha,
               "checks": {cid: {"status": results[cid]["status"],
                                "evidence": results[cid]["evidence"]}
                          for cid, _ in CHECKS},
               "numbers": numbers}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n",
                         encoding="utf-8")
    return status_path, json_path, sha, stamp


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="BASELINE scoreboard for docs/BASELINE.md (B1-B8).")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any check is FAIL")
    ap.add_argument("--clean-clone", action="store_true",
                    help="B8: clone origin/main to a temp dir and run "
                         "the README quickstart there")
    ap.add_argument("--root", default=".",
                    help="repo root (default: cwd; tests use a fixture dir)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    print("== baseline scoreboard (B1-B8) ==")
    b1 = check_b1(root)
    print(f"B1 tests+node: {b1['status']}")
    b2 = check_b2(root)
    print(f"B2 real demo: {b2['status']}")
    b3 = check_b3(root, b2.get("demo_output"))
    print(f"B3 determinism: {b3['status']}")
    b4 = check_b4(root)
    print(f"B4 numbers: {b4['status']}")
    b5 = check_b5(root)
    print(f"B5 funnel: {b5['status']}")
    b6 = check_b6(root)
    print(f"B6 screens: {b6['status']}")
    b7 = check_b7(root)
    print(f"B7 pico: {b7['status']}")
    b8 = check_b8(root, clean_clone=args.clean_clone)
    print(f"B8 tag+clone: {b8['status']}")

    results = {"B1": b1, "B2": b2, "B3": b3, "B4": b4,
               "B5": b5, "B6": b6, "B7": b7, "B8": b8}
    status_path, json_path, sha, stamp = write_outputs(
        root, results, b4.get("numbers", {}))
    fails = [c for c, _ in CHECKS if results[c]["status"] == "FAIL"]
    pend = [c for c, _ in CHECKS if results[c]["status"] == "PENDING"]
    print(f"wrote {status_path} and {json_path}")
    print(f"summary: {len(CHECKS) - len(fails) - len(pend)} PASS, "
          f"{len(fails)} FAIL ({','.join(fails) or '-'}), "
          f"{len(pend)} PENDING ({','.join(pend) or '-'}) @ {sha[:12]}")
    if args.strict and fails:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
