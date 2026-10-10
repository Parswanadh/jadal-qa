"""Voicebank: message catalog and the Bulbul v3 generator that reads it.

`voicebank/messages.json` is the single source of the spoken words (reason code
x language, plus one header sentence per batch decision); `gen_voicebank.py`
turns those strings into `{lang}/{CODE}.wav` with Sarvam Bulbul v3 and skips
files that already exist. `python voicebank/gen_voicebank.py --dry-run` prints
the plan and the estimated spend without touching the network. Importing this
package does no network I/O and needs no API key.
"""
