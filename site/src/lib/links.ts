export const REPO_BLOB = "https://github.com/Parswanadh/jadal-qa/blob/main/";

/** Link to a file in the repository, with an optional line anchor. */
export function repoHref(file: string, line?: number | null): string {
  return `${REPO_BLOB}${file}${line ? `#L${line}` : ""}`;
}
