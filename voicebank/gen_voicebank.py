"""Generate the voicebank wav files with Sarvam Bulbul v3.

Reads `voicebank/messages.json` (reason code x language, plus one header
sentence per batch decision) and writes `{lang}/{CODE}.wav`, e.g.
`hi-IN/BLUR.wav` and `hi-IN/BATCH_REJECT.wav`.

    export SARVAM_API_KEY=...               # never put the key in the repo
    python voicebank/gen_voicebank.py --dry-run
    python voicebank/gen_voicebank.py

Behaviour the build contract asks for:
- the key is read from the environment (`SARVAM_API_KEY`) and from nowhere else;
- `model="bulbul:v3"` is always passed explicitly, never left to the API default;
- the speaker comes from the Bulbul v3 list verified in
  `docs/sarvam_reference.md` (default `simran`, override with `--speaker`);
- existing files are skipped, so a re-run only tops up what is missing
  (`--force` overwrites);
- `--dry-run` costs nothing and prints characters and the estimated spend at
  Rs 30 per 10,000 characters;
- no request is ever longer than `--max-chars` (default 2500).

Only `text`, `language_code`, `model` and `speaker` are sent: `pace` and
`speech_sample_rate` are in the REST reference, but their SDK parameter names
are unconfirmed, so they are left out rather than guessed. If the installed
SDK does not accept `language_code`, the call is retried once with
`target_language_code` and the choice is reported.

NOT DONE
--------
- no wav file has been generated here: this environment has no `SARVAM_API_KEY`
  and `sarvamai` is not installed, so only `--dry-run` and the pure functions
  have been exercised;
- Sarvam documents no TTS determinism, so re-running may not reproduce
  byte-identical audio (fine for playback, but no hash of a wav is allowed in
  a receipt);
- the speaker has not been audited per language; every text stays
  `status: machine_draft` until a native speaker reviews it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

VOICEBANK_DIR = Path(__file__).resolve().parent
DEFAULT_MESSAGES = VOICEBANK_DIR / "messages.json"
DEFAULT_OUT_DIR = VOICEBANK_DIR
DEFAULT_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "simran"
DEFAULT_MAX_CHARS = 2500
PRICE_INR_PER_10K_CHARS = 30.0
BATCH_FILE_PREFIX = "BATCH_"

V3_SPEAKERS = (
    "Shubh", "Aditya", "Ritu", "Priya", "Neha", "Rahul", "Pooja", "Rohan",
    "Simran", "Kavya", "Amit", "Dev", "Ishita", "Shreya", "Ratan", "Varun",
    "Manan", "Sumit", "Roopa", "Kabir", "Aayan", "Ashutosh", "Advait",
    "Anand", "Tanya", "Tarun", "Sunny", "Mani", "Gokul", "Vijay", "Shruti",
    "Suhani", "Mohit", "Kavitha", "Rehan", "Soham", "Rupali",
)
V3_SPEAKERS_LOWER = {name.lower() for name in V3_SPEAKERS}


@dataclass(frozen=True)
class Job:
    """One wav file to produce."""

    lang: str
    code: str
    kind: str
    text: str
    out_path: Path

    @property
    def chars(self) -> int:
        return len(self.text)


class CatalogError(ValueError):
    """The catalog is missing text, too long, or an unreviewed draft."""


def load_catalog(path: str | Path = DEFAULT_MESSAGES) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate_cost_inr(chars: int, rate: float = PRICE_INR_PER_10K_CHARS) -> float:
    """Sarvam bills Bulbul v3 at Rs 30 per 10,000 characters."""
    if chars < 0:
        raise ValueError("chars cannot be negative")
    return chars * rate / 10_000.0


def validate_catalog(catalog: dict, max_chars: int = DEFAULT_MAX_CHARS) -> None:
    """Fail loudly on a catalog that must not be synthesised."""
    languages = catalog.get("languages") or []
    reason_codes = catalog.get("reason_codes") or []
    if not languages:
        raise CatalogError("catalog has no languages")
    if not reason_codes:
        raise CatalogError("catalog has no reason_codes")
    for code in reason_codes:
        entries = catalog.get("messages", {}).get(code)
        if not entries:
            raise CatalogError(f"reason code {code} has no messages")
        for lang in languages:
            entry = entries.get(lang)
            if not entry or not str(entry.get("text", "")).strip():
                raise CatalogError(f"{code}/{lang} has no text")
            if len(entry["text"]) > max_chars:
                raise CatalogError(
                    f"{code}/{lang} is {len(entry['text'])} chars, over the {max_chars} limit"
                )
    for decision in catalog.get("batch_decisions") or []:
        entries = catalog.get("batch_headers", {}).get(decision)
        if not entries:
            raise CatalogError(f"batch decision {decision} has no header sentence")
        for lang in languages:
            entry = entries.get(lang)
            if not entry or not str(entry.get("text", "")).strip():
                raise CatalogError(f"batch header {decision}/{lang} has no text")
            if len(entry["text"]) > max_chars:
                raise CatalogError(
                    f"batch header {decision}/{lang} is {len(entry['text'])} chars,"
                    f" over the {max_chars} limit"
                )


def build_jobs(
    catalog: dict,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    langs: Iterable[str] | None = None,
    codes: Iterable[str] | None = None,
) -> list[Job]:
    """Deterministic job list: languages in catalog order, then reason codes in
    catalog order, then batch decisions in catalog order."""
    out_root = Path(out_dir)
    want_langs = [str(x) for x in langs] if langs is not None else list(catalog["languages"])
    want_codes = [str(x) for x in codes] if codes is not None else None
    if want_codes is not None:
        allowed = set(catalog["reason_codes"]) | {BATCH_FILE_PREFIX + d.upper() for d in catalog["batch_decisions"]}
        unknown = sorted(set(want_codes) - allowed)
        if unknown:
            raise CatalogError(f"unknown codes: {', '.join(unknown)}")

    jobs: list[Job] = []
    for lang in want_langs:
        reason_codes = want_codes if want_codes is not None else catalog["reason_codes"]
        for code in reason_codes:
            if code not in catalog["messages"]:
                continue
            entry = catalog["messages"][code].get(lang)
            if not entry:
                raise CatalogError(f"no message for {code}/{lang}")
            jobs.append(
                Job(lang=lang, code=code, kind="reason", text=entry["text"],
                    out_path=out_root / lang / f"{code}.wav")
            )
        headers = {BATCH_FILE_PREFIX + d.upper(): d for d in catalog["batch_decisions"]}
        for file_code, decision in headers.items():
            if want_codes is not None and file_code not in want_codes:
                continue
            entry = catalog["batch_headers"][decision].get(lang)
            if not entry:
                raise CatalogError(f"no batch header for {decision}/{lang}")
            jobs.append(
                Job(lang=lang, code=file_code, kind="batch", text=entry["text"],
                    out_path=out_root / lang / f"{file_code}.wav")
            )
    return jobs


def split_jobs(jobs: Sequence[Job], force: bool = False) -> tuple[list[Job], list[Job]]:
    """(to generate, skipped because the file already exists)"""
    todo: list[Job] = []
    skipped: list[Job] = []
    for job in jobs:
        if force or not job.out_path.exists():
            todo.append(job)
        else:
            skipped.append(job)
    return todo, skipped


def check_max_chars(text: str, max_chars: int) -> None:
    if len(text) > max_chars:
        raise ValueError(
            f"request text is {len(text)} chars, over the {max_chars} char limit;"
            " split it in messages.json instead"
        )


def api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Export it in your shell (never commit it)."
        )
    return key


def make_client(key: str):
    """Lazy import so that importing this module never needs the SDK."""
    try:
        from sarvamai import SarvamAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the sarvamai package is not installed in this interpreter"
            " (pip install -U sarvamai is the one agent owning the env's job)"
        ) from exc
    return SarvamAI(api_subscription_key=key)


def audio_bytes(response) -> bytes:
    """`audios` is a list of base64-encoded WAV strings, one per input text."""
    audios = getattr(response, "audios", None)
    if audios is None and isinstance(response, dict):
        audios = response.get("audios")
    if not audios:
        raise RuntimeError(f"Sarvam returned no audio: {response!r}")
    return base64.b64decode(audios[0])


def synthesise(
    text: str,
    lang: str,
    *,
    speaker: str = DEFAULT_SPEAKER,
    model: str = DEFAULT_MODEL,
    max_chars: int = DEFAULT_MAX_CHARS,
    client=None,
) -> tuple[bytes, str]:
    """Return (wav bytes, the language parameter name that worked)."""
    check_max_chars(text, max_chars)
    if client is None:
        client = make_client(api_key())
    last_error: TypeError | None = None
    for param_name in ("language_code", "target_language_code"):
        try:
            response = client.text_to_speech.convert(
                text=text,
                **{param_name: lang},
                model=model,
                speaker=speaker,
            )
        except TypeError as exc:
            last_error = exc
            continue
        return audio_bytes(response), param_name
    raise RuntimeError(
        "client.text_to_speech.convert accepted neither `language_code` nor"
        f" `target_language_code`: {last_error}"
    )


def format_plan(jobs: Sequence[Job], skipped: Sequence[Job]) -> str:
    lines = [
        f"{'lang':6} {'code':20} {'chars':>5}  output",
        "-" * 60,
    ]
    for job in jobs:
        lines.append(f"{job.lang:6} {job.code:20} {job.chars:5}  {job.out_path}")
    for job in skipped:
        lines.append(f"{job.lang:6} {job.code:20} {job.chars:5}  SKIP (exists) {job.out_path}")
    lines.append("-" * 60)
    chars = sum(job.chars for job in jobs)
    lines.append(
        f"to generate: {len(jobs)} files, {chars} chars, estimated Rs"
        f" {estimate_cost_inr(chars):.2f} at Rs {PRICE_INR_PER_10K_CHARS:.0f} per 10K chars"
    )
    lines.append(f"skipped (already on disk): {len(skipped)} files")
    return "\n".join(lines)


def resolve_speaker(speaker: str) -> str:
    if speaker.lower() not in V3_SPEAKERS_LOWER:
        raise CatalogError(
            f"speaker {speaker!r} is not in the verified Bulbul v3 list:"
            f" {', '.join(V3_SPEAKERS)}"
        )
    return speaker


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate voicebank wav files from messages.json with Sarvam Bulbul v3."
    )
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--lang", action="append", default=None,
                        help="limit to a language, repeatable (default: all)")
    parser.add_argument("--code", action="append", default=None,
                        help="limit to a reason code or BATCH_ACCEPT/BATCH_REJECT, repeatable")
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--dry-run", action="store_true",
                        help="cost and plan only, no API call")
    parser.add_argument("--force", action="store_true", help="overwrite existing wav files")
    args = parser.parse_args(argv)

    speaker = resolve_speaker(args.speaker)
    catalog = load_catalog(args.messages)
    validate_catalog(catalog, max_chars=args.max_chars)
    jobs = build_jobs(catalog, out_dir=args.out_dir, langs=args.lang, codes=args.code)
    todo, skipped = split_jobs(jobs, force=args.force)
    print(format_plan(todo, skipped))
    print(f"model: {args.model}   speaker: {speaker}   max chars per request: {args.max_chars}")

    if args.dry_run:
        print("dry run: no API call made, no file written")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    client = make_client(api_key())
    used_param = ""
    for job in todo:
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        data, used_param = synthesise(
            job.text,
            job.lang,
            speaker=speaker,
            model=args.model,
            max_chars=args.max_chars,
            client=client,
        )
        job.out_path.write_bytes(data)
        print(f"wrote {job.out_path} ({len(data)} bytes)")
    print(f"done: {len(todo)} files, language parameter was {used_param!r}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except (RuntimeError, CatalogError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
