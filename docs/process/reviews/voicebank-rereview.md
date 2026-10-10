ALL ADDRESSED: yes, NEW BREAKAGE: none

Scope: fix round 1 diff only (3e6a7fb..origin/feat/voicebank, origin HEAD 83f75fe, matches worktree HEAD). Saved at reviews/voicebank-fix1.diff. No code changed, nothing pushed, merged or commented on GitHub.

## Original findings (reviews/voicebank-review.md)

Critical: none in the original review.

Important 1 (non-atomic wav write, truncated files skipped forever): ADDRESSED.
- gen_voicebank.py `write_wav_atomic`: temp file via mkstemp in the same dir, fsync, os.replace; temp removed on any exception.
- gen_voicebank.py `split_jobs` now uses `is_complete_wav` (size > MIN_WAV_BYTES = 44), so zero-byte and header-only files are regenerated.

Important 2 (target_language_code retry, swallowed TypeError): ADDRESSED.
- gen_voicebank.py `synthesise` calls `client.text_to_speech.convert(text=, language_code=, model=, speaker=)` once. No retry loop. TypeError propagates.
- voicebank/docs/sarvam_reference.md line 25 supports language_code for TTS and treats target_language_code for TTS as unconfirmed.

Important 3 (no unittest for voicebank/): ADDRESSED.
- New tests/test_voicebank.py (31 tests, no network, no sarvamai). Covers catalog x REASON_CODES x LANGS, machine_draft flags, cost math (3570 chars -> Rs 10.71), validate_catalog, resolve_speaker, synthesise with fake client, split_jobs skip/force/empty/header-only, atomic write and temp cleanup, end-to-end run().

Minor items (not required for this re-review, status noted for the deferred list):
- Minor 4 (Tamil "பேச்ச்" typo): deliberately NOT changed; left for native review and noted in the report. Still open by design.
- Minor 5, 6, 7, 8, 9: ADDRESSED (/tmp fallback removed from sop test base; load_sop_index now used by CLI main; self_docs renamed to docs; _vectors_for simplified; report claims now backed by tests/test_voicebank.py).

## NEW Critical/Important from fix diff: none

Checked the changed call sites: `synthesise` has one caller (run), updated for the bytes-only return. `load_sop_index(directory, embedder=)` matches its signature. `run(argv, client=None)` is backward compatible.

## Deferred minors (out of scope)
- is_complete_wav only checks size; a pre-existing non-empty truncated wav is still skipped. Atomic writes make new truncation unlikely.
- write_wav_atomic: if os.fdopen fails, the mkstemp fd is not closed (negligible).

## Tests (run now)
Command: PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests
Result: Ran 62 tests, OK (31 sop_index + 31 voicebank).

## Report check (docs/reports/voicebank.md "Fix round 1")
- The "Ran 62 tests ... OK" block matches the verbose tail -30 output I reproduced exactly (same 25 test lines and footer).
- "31 tests" for tests/test_voicebank.py is correct.
- The fake-client and miniconda claims are now backed by the repo test file. Miniconda cross-check was not re-run.
