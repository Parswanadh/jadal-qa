"""Join benchmark windows to /home/parshu/projects/jadal-qa-wt/logs/gpu_telemetry.csv.

CSV has no header; column order (from tools/gpu_telemetry.sh):
timestamp, power.draw, power.limit, clocks.sm, clocks.mem, temperature.gpu,
utilization.gpu, memory.used, thr_sw_power, thr_sw_thermal, thr_hw_thermal, thr_hw_brake.
Timestamps are local wall-clock 'YYYY/MM/DD HH:MM:SS.mmm'. No lock needed (read-only).
"""
import csv
import datetime as dt
import os

CSV = "/home/parshu/projects/jadal-qa-wt/logs/gpu_telemetry.csv"


def _parse_ts(s):
    return dt.datetime.strptime(s.strip().split(".")[0], "%Y/%m/%d %H:%M:%S")


def _num(s, unit=""):
    s = s.strip().replace(" " + unit, "").replace("%", "").replace("MiB", "").strip()
    if s in ("[N/A]", "N/A", "Not Active", ""):
        return None
    try:
        return float(s.replace("MHz", "").strip())
    except ValueError:
        return s


def window_stats(start, end):
    """start/end: datetime (local) or 'YYYY/MM/DD HH:MM:SS' str. Returns summary dict."""
    if isinstance(start, str):
        start = dt.datetime.strptime(start, "%Y/%m/%d %H:%M:%S")
    if isinstance(end, str):
        end = dt.datetime.strptime(end, "%Y/%m/%d %H:%M:%S")
    rows = []
    with open(CSV) as f:
        for r in csv.reader(f):
            if len(r) < 12:
                continue
            try:
                ts = _parse_ts(r[0])
            except ValueError:
                continue
            if start <= ts <= end:
                rows.append(r)
    if not rows:
        return {"n": 0, "note": "no telemetry rows in window"}
    power = [x for x in (_num(r[1], "W") for r in rows) if isinstance(x, float)]
    sm = [x for x in (_num(r[3]) for r in rows) if isinstance(x, float)]
    memc = [x for x in (_num(r[4]) for r in rows) if isinstance(x, float)]
    temp = [x for x in (_num(r[5]) for r in rows) if isinstance(x, float)]
    util = [x for x in (_num(r[6]) for r in rows) if isinstance(x, float)]
    memused = [x for x in (_num(r[7]) for r in rows) if isinstance(x, float)]

    def s(v):
        v = sorted(v)
        return {"n": len(v), "mean": round(sum(v) / len(v), 1),
                "min": v[0], "p50": v[len(v) // 2], "max": v[-1]} if v else None

    thr = {}
    for i, k in enumerate(["sw_power_cap", "sw_thermal", "hw_thermal", "hw_brake"]):
        c = sum(1 for r in rows if r[8 + i].strip() == "Active")
        thr[k] = c
    return {"n": len(rows), "power_W": s(power), "sm_clock_MHz": s(sm),
            "mem_clock_MHz": s(memc), "temp_C": s(temp), "util_pct": s(util),
            "mem_used_MiB": s(memused), "throttle_active_rows": thr,
            "thermal_note": ("laptop GPU is thermally limited; clocks below "
                             "boost are expected under sustained load")}


def now_str():
    return dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
