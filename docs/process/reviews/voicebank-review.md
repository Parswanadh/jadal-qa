SPEC: PASS, QUALITY: CHANGES NEEDED

Scope: PR #11, branch feat/voicebank (diff vs main at reviews/voicebank.diff), for issue #5. Review only; no code changed, nothing merged, pushed or commented.

Tests run: `PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests -v` in the voicebank worktree: Ran 31 tests, OK. Same 31 pass under miniconda3. `gen_voicebank.py --dry-run` (no key): 60 files, 3570 chars, Rs 10.71, exit 0. The sarvamai package is not installed in the env, so no live TTS call was made.

## SPEC compliance

1. messages.json covers all 10 REASON_CODES (matches jadal/config.py) x hi-IN, kn-IN, ta-IN, te-IN, en-IN, plus accept/reject batch headers. 50 reason entries and 10 header entries. Every entry has status=machine_draft; all 40 non-English entries have needs_native_review=true. PASS. (voicebank/messages.json)
2. gen_voicebank.py: sarvamai SDK, `model=DEFAULT_MODEL` ("bulbul:v3") passed explicitly, speaker validated against the 36-name v3 list (matches docs/sarvam_reference.md), key from SARVAM_API_KEY only, max 2500 chars enforced, skip-existing via split_jobs, --dry-run prints chars and Rs 30/10K cost. PASS, with the caveat in Important #2. (voicebank/gen_voicebank.py:50-60, 74-85, 224-240, 290-340)
3. sop/sop_index.py: deterministic. Sorted by passage_id, ordering key `(-score, -bm25, passage_id)`, scores rounded to 6 dp, text is the verbatim slice, BM25 fallback needs no GPU. PASS. (sop/sop_index.py:260-280, 455-480)
4. sop/sop_garment_v1.md is labelled SAMPLE in the header comment and in a blockquote. PASS. (sop/sop_garment_v1.md:1-14)
5. tests/test_sop_index.py asserts tie-break determinism (BM25 and embedding paths) and verbatim text (test_text_is_verbatim_and_never_generated, test_passage_text_is_a_verbatim_slice_of_the_file). PASS.
6. No hardcoded keys (grep over the module, catalog and docs). PASS.

Protected files: none of jadal/config.py, jadal/__init__.py, docs/SPEC.md is touched. No network calls at import time. PASS.

## Findings

### Critical
None.

### Important
1. voicebank/gen_voicebank.py:335 `job.out_path.write_bytes(data)` is not atomic. A crash or Ctrl-C mid-write leaves a truncated .wav, and split_jobs then skips it forever (it only checks exists). Fix: write to `out_path.with_suffix(".wav.tmp")` and `os.replace()` it into place.
2. voicebank/gen_voicebank.py:203-216 `synthesise` retries on any TypeError with `target_language_code`. That contradicts docs/sarvam_reference.md, which states TTS uses `language_code` and treats `target_language_code` for TTS as unconfirmed (that name belongs to Translate). It also swallows unrelated TypeErrors from the SDK and retries them. Fix: call with `language_code` only, drop the fallback, and let a TypeError surface. If the fallback is kept, check the SDK signature with inspect.signature instead.
3. No unittest covers voicebank/ at all. SPEC "Done means" requires a unittest per module. The untested paths are the ones that cost money or touch disk: build_jobs filters, split_jobs skip/force, format_plan cost arithmetic, validate_catalog, resolve_speaker, synthesise with a fake client. The report (docs/reports/voicebank.md) says these were "exercised against a fake client in-process", but that harness is not in the repo and cannot be re-run. Fix: add tests/test_gen_voicebank.py with a fake client and tempdir output, and assert the cost figure (3570 chars -> Rs 10.71).

### Minor
4. voicebank/messages.json, batch_headers.accept.ta-IN: "உங்கள் பேச்ச் QA" uses "பேச்ச்" (speech), not "பேட்ச்" (batch). It is a machine draft, so flag it to the native reviewer. Fix: correct the word, or leave it to the native review and note it there.
5. tests/test_sop_index.py:46 falls back to `tempfile.mkdtemp()` (system /tmp) when data_runtime is not writable. SPEC says never write results to /tmp. Fix: raise instead of falling back, or use the scratchpad.
6. sop/sop_index.py:488 `load_sop_index` is unused (dead code). Fix: delete it or use it from the CLI.
7. sop/sop_index.py:364-369 the local `self_docs` is a misnamed variable. Fix: rename to `docs`.
8. docs/reports/voicebank.md claims the fake-client path was exercised and cites a miniconda cross-check. Neither is reproducible from the repo. Fix: remove the claim or back it with the test from Important #3.
9. sop/sop_index.py:446 `_vectors_for` checks `cached[0] is not embedder` after already keying on `id(embedder)`. The check is redundant but harmless. Fix: simplify.
