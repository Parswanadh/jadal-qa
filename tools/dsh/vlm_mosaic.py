"""Frozen-prompt caller for the V-track (Muse Spark 1.3, image contact sheets).

Wraps `portal/muse_video.py::_post` (the OpenCode Go Responses endpoint).  The API
key is read in-process from OpenCode's auth store by that module; it is NEVER
printed, logged or written here.

Every call is appended to a JSONL ledger recording model id, prompt name, the full
prompt text, the raw reply, seconds, and the PNG sha256.  The model is
probabilistic: nothing here is deterministic, and a "receipt" for a call records
model + prompt + reply, not a guarantee.

Frozen prompt texts live in PROMPTS and their sha256 is written into each
experiment's PREREG, so a prompt cannot change after a run without the hash moving.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time

PORTAL = os.environ.get("JADAL_PORTAL", "/home/parshu/projects/awing/portal")
if PORTAL not in sys.path:
    sys.path.insert(0, PORTAL)

import muse_video as MV  # noqa: E402

MODEL_ID = MV.MODEL_ID
MODEL_LABEL = MV.MODEL_LABEL
DEFAULT_LEDGER = os.environ.get(
    "JADAL_VLM_LEDGER", "/tmp/opencode/dsh/vlm_calls.jsonl")

REASON_CODES = sorted(MV.REASON_CODES)

# ---------------------------------------------------------------- prompts ----
PROMPTS = {
    # V01: can the model actually read the sheet, tile by tile?
    "read_tiles": (
        "This image is a contact sheet of frames from ONE head-camera video, in time "
        "order (left to right, then top to bottom). Every tile carries a large label "
        "of the form '#<tile number>  <MM:SS.s>s' giving that tile's number and time.\n"
        "For EVERY tile, output exactly one line in this form and nothing else:\n"
        "tile <number> @ <timestamp> :: <what is visible, <=12 words>\n"
        "Do not skip a tile. Do not add a JSON block. Do not summarise."
    ),
    # V02/V03: the quality verdict (adapted from the shipped video prompt).
    "quality": (
        "This single image is a contact sheet of frames from ONE 10-second head-camera "
        "factory video, in time order (left to right, top to bottom). Each tile carries "
        "a large label '#<tile number>  <MM:SS.s>s' giving that tile's number and time.\n"
        "First, in one short line per tile, say what the worker's hands are doing at "
        "that timestamp. Then answer JSON only on the last line: "
        '{"usable": true|false, "reason_code": "OK|TOO_DARK|BLOCKED_VIEW|NO_ACTIVE_WORK|'
        'HANDS_NOT_VISIBLE|DUPLICATE_OR_FROZEN", "task_progress": '
        '"advancing|no_relevant_work|cannot_tell", "explanation": "<=25 words"}'
    ),
    # V03(e): same question, reasoning step forbidden.
    "quality_json_only": (
        "This single image is a contact sheet of frames from ONE 10-second head-camera "
        "factory video, in time order (left to right, top to bottom). Each tile carries "
        "a large label '#<tile number>  <MM:SS.s>s' giving that tile's number and time.\n"
        "Answer with JSON only on one line and no other text: "
        '{"usable": true|false, "reason_code": "OK|TOO_DARK|BLOCKED_VIEW|NO_ACTIVE_WORK|'
        'HANDS_NOT_VISIBLE|DUPLICATE_OR_FROZEN", "task_progress": '
        '"advancing|no_relevant_work|cannot_tell", "explanation": "<=25 words"}'
    ),
    # V05 L1: is relevant work progressing across this 30-second slice?
    "slice_l1": (
        "This single image is a contact sheet of frames from ONE 30-second slice of a "
        "head-camera factory video, in time order (left to right, top to bottom). Each "
        "tile carries a large label '#<tile number>  <MM:SS.s>s' giving that tile's "
        "number and the time within the ORIGINAL clip.\n"
        "Judge whether the worker is doing work that is relevant to their task: not "
        "merely being busy, and not merely being present. Walking, chatting, waiting, "
        "or standing at the wrong place count as no relevant work.\n"
        "Answer with JSON only on one line and no other text: "
        '{"progressing": "advancing|no_relevant_work|cannot_tell", '
        '"reason": "<=20 words", "visibility": "clear|partial|blocked"}'
    ),
    # V05 L2: text-only summary over the slice answers -> clip-level completion.
    "l2_text": (
        "You are given the JSON answers for consecutive 30-second slices of ONE "
        "head-camera task clip, in order. Decide whether the TASK was COMPLETED.\n"
        "Working definition of completed (the builder's, not the customer's): the "
        "worker picks an object from one place, puts it in another place, and does so "
        "accurately -- right object, right destination, no drop and no redo. Judge from "
        "the visible start and end state, NOT from how busy the hands look.\n"
        "Answer with JSON only on one line and no other text: "
        '{"completed": "completed|not_completed|cannot_tell", '
        '"farthest_step": "<=15 words", "explanation": "<=25 words"}'
    ),
}

PROMPT_SHA256 = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in PROMPTS.items()}


# ------------------------------------------------------------------ calls ----
def png_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _extract_json(text: str):
    """Last JSON object in the reply, or None.  Does not validate the fields."""
    matches = re.findall(r"\{[^{}]*\}", text, re.S)
    for cand in reversed(matches):
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj:
            return obj
    return None


def parse_quality(text: str):
    """(usable, reason_code, task_progress, explanation) or None."""
    obj = _extract_json(text)
    if not obj or not isinstance(obj.get("usable"), bool):
        return None
    code = str(obj.get("reason_code", "")).strip()
    if code not in MV.REASON_CODES:
        code = "UNPARSED_CODE"
    prog = str(obj.get("task_progress", "")).strip()
    if prog not in ("advancing", "no_relevant_work", "cannot_tell"):
        prog = "cannot_tell"
    return obj["usable"], code, prog, str(obj.get("explanation", "")).strip()


def parse_tile_lines(text: str):
    """{tile_number: (timestamp_string, description)} from a sheet reply.

    Two shapes occur in practice and BOTH are accepted here:
      * ``tile 7 @ 00:05.4s :: description``   (what `read_tiles` asks for)
      * ``#7  00:05.4s: description``          (what the model usually echoes back,
        copying the sheet's own label -- diagnosed by the V-A stream, which found the
        original stricter pattern recovered 0 lines from 160 real replies)
    """
    out = {}
    pats = [re.compile(r"\s*tile\s*#?(\d+)\s*@\s*([0-9]{1,2}:[0-9]{2}(?:\.[0-9])?)\s*s?\s*::\s*(.*)", re.I),
            re.compile(r"\s*#(\d+)\s*@?\s*([0-9]{1,2}:[0-9]{2}(?:\.[0-9])?)\s*s?\s*[:\-]\s*(.*)"),
            re.compile(r"\s*tile\s*#?(\d+)\s*@?\s*([0-9]{1,2}:[0-9]{2}(?:\.[0-9])?)\s*s?\s*[:\-]\s*(.*)", re.I)]
    for line in text.splitlines():
        for pat in pats:
            m = pat.match(line)
            if m:
                out[int(m.group(1))] = (m.group(2).strip(), m.group(3).strip())
                break
    return out


def ask(png: bytes | str, prompt_name: str, *, timeout: int = 200,
        ledger: str | None = DEFAULT_LEDGER, extra: dict | None = None) -> dict:
    """One image call with a frozen prompt.  Never raises; always returns a dict.

    Returns keys: ok, model, prompt_name, prompt_sha256, seconds, raw, png_sha256,
    parsed (dict or None), error (when ok is False), plus whatever ``extra`` holds.
    """
    if isinstance(png, str):
        with open(png, "rb") as f:
            data = f.read()
    else:
        data = png
    prompt = PROMPTS[prompt_name]
    sha = png_sha256(data)
    started = time.time()
    rec = {"model": MODEL_LABEL, "model_id": MODEL_ID, "prompt_name": prompt_name,
           "prompt_sha256": PROMPT_SHA256[prompt_name], "png_sha256": sha,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    try:
        body = {"model": MODEL_ID, "reasoning": {"effort": "low"},
                "max_output_tokens": 1800,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _data_url(data)}]}]}
        payload = MV._post(body, timeout)
        raw = MV._output_text(payload)
        rec.update({"ok": True, "raw": raw, "seconds": round(time.time() - started, 2)})
    except Exception as exc:                                   # noqa: BLE001
        detail = str(exc)[:300]
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:                                  # pragma: no cover
                pass
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {detail}",
                    "raw": "", "seconds": round(time.time() - started, 2)})
    rec["parsed"] = (_extract_json(rec["raw"]) if rec.get("raw") else None)
    rec["quality"] = (parse_quality(rec["raw"]) if rec.get("raw") else None)
    rec["tiles"] = (parse_tile_lines(rec["raw"]) if rec.get("raw") else {})
    if ledger:
        os.makedirs(os.path.dirname(os.path.abspath(ledger)), exist_ok=True)
        with open(ledger, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def ask_text(user_text: str, prompt_name: str, *, timeout: int = 200,
             ledger: str | None = DEFAULT_LEDGER, extra: dict | None = None) -> dict:
    """Text-only call (the V05 L2 summariser).  Same ledger, same honesty rules."""
    prompt = PROMPTS[prompt_name]
    started = time.time()
    rec = {"model": MODEL_LABEL, "model_id": MODEL_ID, "prompt_name": prompt_name,
           "prompt_sha256": PROMPT_SHA256[prompt_name], "png_sha256": None,
           "text_only": True, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        rec.update(extra)
    try:
        body = {"model": MODEL_ID, "reasoning": {"effort": "low"},
                "max_output_tokens": 1200,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt + "\n\nSLICES:\n" + user_text}]}]}
        payload = MV._post(body, timeout)
        raw = MV._output_text(payload)
        rec.update({"ok": True, "raw": raw, "seconds": round(time.time() - started, 2)})
    except Exception as exc:                                   # noqa: BLE001
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "raw": "", "seconds": round(time.time() - started, 2)})
    rec["parsed"] = (_extract_json(rec["raw"]) if rec.get("raw") else None)
    if ledger:
        os.makedirs(os.path.dirname(os.path.abspath(ledger)), exist_ok=True)
        with open(ledger, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def count_calls(ledger: str = DEFAULT_LEDGER) -> dict:
    """{'image': n, 'text': n, 'total': n} from the ledger (call-count accounting)."""
    img = txt = 0
    if os.path.exists(ledger):
        with open(ledger) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("text_only"):
                    txt += 1
                else:
                    img += 1
    return {"image": img, "text": txt, "total": img + txt}
