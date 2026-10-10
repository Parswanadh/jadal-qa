#!/usr/bin/env bash
# Block commits that contain secrets. Usage: secrets_guard.sh            (scans staged diff)
#                                            secrets_guard.sh --tree     (scans tracked + untracked, non-ignored files)
# Prints MASKED matches only (first 4 chars + length). Exit 1 on any match, 0 if clean.
set -u
PAT='hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(API_KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*[ ]*[=:][ ]*["'"'"']?[A-Za-z0-9_./+-]{16,}'
if [ "${1:-}" = "--tree" ]; then
  src=$( (git ls-files; git ls-files --others --exclude-standard) | sort -u | xargs -r grep -InEH "$PAT" 2>/dev/null )
else
  src=$(git diff --cached -U0 | grep -E '^\+[^+]' | grep -nE "$PAT" 2>/dev/null )
fi
if [ -n "$src" ]; then
  echo "secrets_guard: POSSIBLE SECRET(S) FOUND (masked):" >&2
  echo "$src" | while IFS= read -r l; do m=$(echo "$l" | grep -oE "$PAT" | head -1); echo "  ${l%%:*}: ${m:0:4}…(len ${#m})" >&2; done
  exit 1
fi
echo "secrets_guard: clean"
