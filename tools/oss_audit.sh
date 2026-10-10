#!/usr/bin/env bash
# oss_audit.sh: READ-ONLY pre-publication audit of this repository.
#
# Scans the working tree (tracked files, untracked files that are not
# ignored, and git-ignored .env files) and every commit reachable from --all
# (diffs, commit messages and commit metadata). It writes nothing into the
# repo and changes no git state.
#
# Output shows MASKED matches only: the first 4 characters and the length.
# A full secret or a full e-mail address is never printed.
#
# Exit 0: no secret pattern matched. Exit 1: a secret pattern matched in the
# working tree or in history. Exit 2: not inside a git repository.
# E-mail, path, size and media findings are printed for the owner's review;
# they do not change the exit code.
set -u

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "oss_audit: not inside a git repository" >&2
  exit 2
}
cd "$ROOT" || exit 2

# Secret patterns, POSIX ERE, one alternation. No backslashes, so the
# string is safe to pass through awk -v. Bounded repeats are written as
# X{n}X* because mawk mis-reports the length of X{n,} matches.
SECRET_RE="hf_[A-Za-z0-9]{20}[A-Za-z0-9]*|github_pat_[A-Za-z0-9_]{20}[A-Za-z0-9_]*|ghp_[A-Za-z0-9]{20}[A-Za-z0-9]*|sk-[A-Za-z0-9]{20}[A-Za-z0-9]*|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|[A-Z_]*(KEY|TOKEN|SECRET)[=][\"']?[^[:space:]\"'<\$]{8}[^[:space:]\"'<\$]*"
EMAIL_RE="[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+([.][A-Za-z0-9-]+)+"

# Shared awk helpers. mask() never lets a full value through.
AWK_LIB='
function mask(s) { return substr(s, 1, 4) "...(len " length(s) ")" }
function allowed(e,   l, d) {
  l = tolower(substr(e, 1, index(e, "@") - 1))
  d = tolower(substr(e, index(e, "@") + 1))
  if (l ~ /^(no-?reply)$/) return 1
  if (d !~ /[a-z][a-z]+$/) return 1
  if (d ~ /^(example[.](com|org|net)|users[.]noreply[.]github[.]com)$/) return 1
  return 0
}
'

# Text files to scan: tracked, untracked-not-ignored, and git-ignored .env files.
TEXT_FILES=()
while IFS= read -r -d '' f; do
  if [ -f "$f" ] && grep -Iq . -- "$f" 2>/dev/null; then
    TEXT_FILES+=("$f")
  fi
done < <(
  git ls-files -z --cached --others --exclude-standard
  find . -maxdepth 2 -name '.env*' -not -name '.env.example' -not -path './.git/*' -print0 2>/dev/null
)

echo "== oss_audit: $ROOT"
echo "== text files scanned in working tree: ${#TEXT_FILES[@]}"
echo

echo "== 1. Secret patterns, working tree"
tree_sec=$(awk -v re="$SECRET_RE" "$AWK_LIB"'
  { s = $0
    while (match(s, re)) {
      printf "SECRET  %s:%d  %s\n", FILENAME, FNR, mask(substr(s, RSTART, RLENGTH))
      s = substr(s, RSTART + RLENGTH)
    } }' "${TEXT_FILES[@]}" | sort -u)
echo "${tree_sec:-  none}"
echo

echo "== 2. Secret patterns, git history (diffs and messages, all refs)"
hist_out=$(git log --all -p --no-color --format='@@C %h%n%B' 2>/dev/null | awk -v re="$SECRET_RE" -v em="$EMAIL_RE" "$AWK_LIB"'
  /^@@C / { c = $2; next }
  { s = $0
    while (match(s, re)) {
      printf "SECRET  history %s  %s\n", c, mask(substr(s, RSTART, RLENGTH))
      s = substr(s, RSTART + RLENGTH)
    }
    s = $0
    while (match(s, em)) {
      e = substr(s, RSTART, RLENGTH)
      if (!allowed(e)) printf "EMAIL   history %s  %s\n", c, mask(e)
      s = substr(s, RSTART + RLENGTH)
    } }' | sort -u)
hist_sec=$(printf '%s\n' "$hist_out" | grep '^SECRET')
echo "${hist_sec:-  none}"
echo

echo "== 3. Personal e-mail addresses (commit metadata, messages, diffs, working tree)"
meta_em=$(git log --all --format='%ae%n%ce' 2>/dev/null | awk -v em="$EMAIL_RE" "$AWK_LIB"'
  { if (match($0, "^" em "$") && !allowed($0)) printf "EMAIL   metadata  %s\n", mask($0) }')
tree_em=$(awk -v em="$EMAIL_RE" "$AWK_LIB"'
  { s = $0
    while (match(s, em)) {
      e = substr(s, RSTART, RLENGTH)
      if (!allowed(e)) printf "EMAIL   %s:%d  %s\n", FILENAME, FNR, mask(e)
      s = substr(s, RSTART + RLENGTH)
    } }' "${TEXT_FILES[@]}")
email_out=$(printf '%s\n%s\n%s\n' "$meta_em" "$(printf '%s\n' "$hist_out" | grep '^EMAIL')" "$tree_em" | grep '^EMAIL' | sort -u)
email_count=$(printf '%s\n' "$email_out" | grep -c '^EMAIL')
echo "${email_out:-  none}"
echo

echo "== 4. Lines containing a personal home path, per file (/home/<user>)"
path_found=0
for f in "${TEXT_FILES[@]}"; do
  n=$(grep -Ec '/home/[A-Za-z0-9_.-]+' -- "$f" 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then
    printf 'PATH    %5d  %s\n' "$n" "$f"
    path_found=1
  fi
done
[ "$path_found" -eq 0 ] && echo "  none"
echo

echo "== 5. Blobs larger than 2 MB (all history)"
big=$(git rev-list --objects --all 2>/dev/null \
  | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' 2>/dev/null \
  | awk '$2 == "blob" && $3 > 2097152 {
      p = $0; sub(/^[^ ]+ [^ ]+ [^ ]+ /, "", p)
      printf "BLOB    %s  %.2f MB  %s\n", substr($1, 1, 8), $3 / 1048576, p }' | sort -u)
echo "${big:-  none}"
echo

echo "== 6. Image and video files (working tree and history; names only)"
media=$( { git rev-list --objects --all 2>/dev/null \
    | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' 2>/dev/null \
    | awk '$2 == "blob" { p = $0; sub(/^[^ ]+ [^ ]+ [^ ]+ /, "", p); print p }'
  git ls-files -co --exclude-standard; } \
  | sort -u | grep -Ei '[.](png|jpe?g|gif|webp|bmp|tiff?|heic|mp4|mov|avi|mkv|webm|m4v)$' | sed 's/^/MEDIA   /')
echo "${media:-  none}"
echo

secrets_n=$( { printf '%s\n' "$tree_sec"; printf '%s\n' "$hist_sec"; } | grep -c '^SECRET' )
echo "== summary: secret matches=$secrets_n  e-mail findings=$email_count  big blobs=$(printf '%s\n' "$big" | grep -c '^BLOB')  media files=$(printf '%s\n' "$media" | grep -c '^MEDIA')"
if [ "$secrets_n" -gt 0 ]; then
  echo "== RESULT: FAIL (secret pattern matched; fix before publishing)"
  exit 1
fi
echo "== RESULT: no secret pattern matched (review e-mail, path, blob and media findings above)"
exit 0
