# voicebank + sop module report (issue #5)

Date: 2026-10-10. Branch: `feat/voicebank` (worktree
`/home/parshu/projects/jadal-qa-wt/voicebank`). Owner files: `voicebank/`,
`sop/`, `tests/test_sop_index.py`.

## What exists

| file | what it is |
|---|---|
| `sop/sop_garment_v1.md` | SAMPLE 5-step garment SOP ("T-shirt collar attach"), labelled SAMPLE in the header block and in a leading blockquote; documents the format real SOPs must follow (header comment with `sop_id`/`lang`/`status`, one `## Step N` section per step) |
| `sop/sop_index.py` | deterministic retrieval. Parses every `*.md` into `## Step` passages, keeps the passage `text` verbatim, ranks by BM25 by default and by cosine when an `Embedder` is given, breaks ties on `(-score, -bm25, passage_id)`, drops non-positive scores |
| `sop/__init__.py` | docstring only (keeps `python -m sop.sop_index` quiet, no import-time network) |
| `voicebank/messages.json` | 60 entries: 5 languages x 10 reason codes + one header sentence per batch decision (`accept`, `reject`); carries TTS settings, the verified v3 speaker list, char/cost model and the wav layout |
| `voicebank/gen_voicebank.py` | Bulbul v3 generator: `--dry-run`, skip-existing, `--force`, per-language/`--code` filters, max-chars guard, speaker validation, `SARVAM_API_KEY` from env |
| `voicebank/__init__.py`, `voicebank/.gitignore` | docstring; wav files are generated artefacts and are ignored |
| `tests/test_sop_index.py` | 31 unittests, no network, no GPU, no model download |

## How to run

```
/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
/home/parshu/envs/jadal/bin/python -m sop.sop_index --q "broken needle" --lang en-IN
/home/parshu/envs/jadal/bin/python voicebank/gen_voicebank.py --dry-run
SARVAM_API_KEY=... /home/parshu/envs/jadal/bin/python voicebank/gen_voicebank.py
```

Output paths: `voicebank/{lang}/{REASON_CODE}.wav`,
`voicebank/{lang}/BATCH_ACCEPT.wav`, `voicebank/{lang}/BATCH_REJECT.wav`
(what `jadal/config.py`'s `voicebank_dir` and `GET /voice/{reason_code}/{lang}` expect).

## Test output (real)

`/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v`:

```
Ran 31 tests in 0.009s

OK
```

`/home/parshu/miniconda3/bin/python -m unittest discover -s tests` (stdlib-only
module, run there as a cross-check): 31 tests, OK.

`voicebank/gen_voicebank.py --dry-run` (no key, no network):

```
to generate: 60 files, 3570 chars, estimated Rs 10.71 at Rs 30 per 10K chars
skipped (already on disk): 0 files
model: bulbul:v3   speaker: simran   max chars per request: 2500
dry run: no API call made, no file written
```

Exit codes checked: missing key -> exit 2 with
`error: SARVAM_API_KEY is not set. Export it in your shell (never commit it).`;
missing sarvamai -> exit 2 with `error: the sarvamai package is not installed...`;
unknown `--code` -> `error: unknown codes: NOPE`; speaker outside the verified
v3 list (e.g. the doc's `simran_en_customer`) -> exit 2 with the full list.
The synthesis path itself was exercised against a fake client in-process
(both `language_code` and the `target_language_code` TypeError fallback,
`split_jobs` skip/force, cost arithmetic).

## Known gaps / NOT DONE

- **No wav file was generated.** This env has no `SARVAM_API_KEY` and
  `sarvamai` is not installed, so only `--dry-run` and the pure functions ran.
  The first real run needs a key; wiring is the only untested part.
- Only `text`, `language_code`, `model`, `speaker` are sent. `pace` and
  `speech_sample_rate` are in `docs/sarvam_reference.md` but the SDK parameter
  names are unconfirmed there, so they are deliberately left out.
- Sarvam documents no TTS determinism, so a re-run may not reproduce
  byte-identical audio. No wav hash belongs in a receipt.
- All text stays `status: machine_draft`; non-English entries carry
  `needs_native_review: true` and need a native speaker before floor use.
- `HashEmbedder` (feature hashing) exists only to exercise the embedding path
  offline; it is not a semantic model. `SentenceTransformersEmbedder` is
  untested here (sentence-transformers is not installed).
- No relevance evaluation: nothing in this module reports an accuracy number.
- SOP index is BM25 or plain cosine, never a blend, and has no reranker.

## Interface notes for other modules

- `sop.sop_index.SOPIndex.load(dir, embedder=None)`;
  `search(query, lang=None, top_k=5, embedder=None) -> list[dict]` with keys
  `rank, passage_id, doc_id, sop_id, lang, title, step, score, bm25, text`
  (JSON-serialisable; `text` is verbatim, so `GET /sop/search?q=&lang=` and
  `/stats/sop_gaps` can consume it directly).
- An embedder is anything with `dim: int` and
  `embed(texts) -> list[list[float]]`; errors from a supplied embedder are not
  swallowed.
- Catalog schema: `schema_version`, `tts`, `languages`, `reason_codes`,
  `batch_decisions`, `messages[code][lang] = {text, status,
  needs_native_review}`, `batch_headers[decision][lang] = {...}`.
- Contract change proposals: none needed; `sop/` and `voicebank/` in SPEC.md
  already match what was built. `docs/sarvam_reference.md` conflict 2 is
  handled by defaulting to `simran` and rejecting off-list speakers.

## Fix round 1 (review of PR #11)

Date: 2026-10-10. Fixed every review finding except the Tamil wording, which
is deliberately left to native-speaker review (see below).

- `gen_voicebank.py`: wav files are now written atomically
  (`write_wav_atomic`: temp file in the same folder, fsync, `os.replace`;
  temp file removed on failure), and `split_jobs` treats a zero-byte or
  header-only wav (`<= MIN_WAV_BYTES = 44`) as missing, so a crash can never
  leave a truncated file that is skipped forever.
- `gen_voicebank.py`: removed the `target_language_code` retry. `synthesise`
  now sends `language_code` only (per `docs/sarvam_reference.md`), returns
  just the wav bytes, and lets an SDK `TypeError` surface instead of
  swallowing it. `run()` takes an optional injectable `client` for tests.
- New `tests/test_voicebank.py` (31 tests, no network, no sarvamai): catalog
  coverage against `jadal.config` `REASON_CODES` x `LANGS` plus batch headers;
  non-English `machine_draft` statuses; cost math pinned at 3570 chars ->
  Rs 10.71; atomic write (incl. temp-file cleanup on failure), skip-existing,
  regenerate-on-empty/header-only, end-to-end `run()` with a fake TTS client.
- `sop/sop_index.py`: the CLI now goes through `load_sop_index` (no more dead
  code); `self_docs` renamed to `docs`; `_vectors_for` simplified (the cache
  holds a strong ref to the embedder, so the identity re-check was redundant).
- `tests/test_sop_index.py`: scratch dirs use `TemporaryDirectory` rooted at
  the repo's gitignored `data_runtime/`; the `/tmp` fallback is gone (a
  failure to create the base raises instead).
- Earlier claims in this report about the fake-client harness and the
  miniconda cross-check are now backed by `tests/test_voicebank.py` in the
  repo and re-runnable with the command below.
- Tamil wording note: `batch_headers.accept.ta-IN` uses "பேச்ச்" (speech)
  where "பேட்ச்" (batch) is meant. The text is a machine draft and is left
  UNCHANGED for the native speaker to correct during review, since editing one
  word without Tamil fluency risks a worse mistake.

Command and REAL output tail:

```
$ cd /home/parshu/projects/jadal-qa-wt/voicebank && PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -30
test_non_english_entries_are_machine_drafts (test_voicebank.CatalogCoverageTests.test_non_english_entries_are_machine_drafts) ... ok
test_estimate_cost_inr (test_voicebank.CostMathTests.test_estimate_cost_inr) ... ok
test_format_plan_reports_chars_and_cost (test_voicebank.CostMathTests.test_format_plan_reports_chars_and_cost) ... ok
test_full_catalog_chars_and_cost (test_voicebank.CostMathTests.test_full_catalog_chars_and_cost) ... ok
test_rate_matches_catalog (test_voicebank.CostMathTests.test_rate_matches_catalog) ... ok
test_dry_run_writes_nothing (test_voicebank.RunEndToEndTests.test_dry_run_writes_nothing) ... ok
test_run_regenerates_truncated_file (test_voicebank.RunEndToEndTests.test_run_regenerates_truncated_file) ... ok
test_run_without_client_and_without_sdk_returns_2 (test_voicebank.RunEndToEndTests.test_run_without_client_and_without_sdk_returns_2) ... ok
test_run_writes_wavs_then_skips (test_voicebank.RunEndToEndTests.test_run_writes_wavs_then_skips) ... ok
test_complete_file_is_skipped (test_voicebank.SplitJobsTests.test_complete_file_is_skipped) ... ok
test_empty_file_is_regenerated (test_voicebank.SplitJobsTests.test_empty_file_is_regenerated) ... ok
test_force_regenerates_complete_files (test_voicebank.SplitJobsTests.test_force_regenerates_complete_files) ... ok
test_header_only_file_is_regenerated (test_voicebank.SplitJobsTests.test_header_only_file_is_regenerated) ... ok
test_missing_file_is_todo (test_voicebank.SplitJobsTests.test_missing_file_is_todo) ... ok
test_audio_bytes_accepts_object_and_dict (test_voicebank.SynthesisTests.test_audio_bytes_accepts_object_and_dict) ... ok
test_no_sarvamai_imported (test_voicebank.SynthesisTests.test_no_sarvamai_imported) ... ok
test_synthesise_enforces_max_chars (test_voicebank.SynthesisTests.test_synthesise_enforces_max_chars) ... ok
test_synthesise_sends_language_code_only (test_voicebank.SynthesisTests.test_synthesise_sends_language_code_only) ... ok
test_synthesise_type_error_is_not_swallowed (test_voicebank.SynthesisTests.test_synthesise_type_error_is_not_swallowed) ... ok
test_default_model_and_speaker_are_explicit (test_voicebank.ValidationTests.test_default_model_and_speaker_are_explicit) ... ok
test_missing_text_raises (test_voicebank.ValidationTests.test_missing_text_raises) ... ok
test_overlong_text_raises (test_voicebank.ValidationTests.test_overlong_text_raises) ... ok
test_real_catalog_validates (test_voicebank.ValidationTests.test_real_catalog_validates) ... ok
test_resolve_speaker (test_voicebank.ValidationTests.test_resolve_speaker) ... ok
test_unknown_code_raises (test_voicebank.ValidationTests.test_unknown_code_raises) ... ok

----------------------------------------------------------------------
Ran 62 tests in 0.027s

OK
```
