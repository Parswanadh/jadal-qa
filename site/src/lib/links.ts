import { el } from "../dom";

/**
 * Base URL for "Source" links, set at build time:
 *   VITE_REPO_BLOB=https://github.com/OWNER/REPO/blob/main/ npm run build
 * Unset (the default) shows plain file paths without links, so a private
 * repository does not produce dead links.
 */
const raw = (import.meta.env.VITE_REPO_BLOB as string | undefined) ?? "";
export const REPO_BLOB: string = raw === "" || raw.endsWith("/") ? raw : `${raw}/`;

export function sourceHref(file: string, line?: number | null): string | null {
  if (REPO_BLOB === "") return null;
  return `${REPO_BLOB}${file}${line ? `#L${line}` : ""}`;
}

/** A source reference: a link when REPO_BLOB is configured, plain code otherwise. */
export function sourceNode(file: string, line?: number | null): HTMLElement {
  const label = line ? `${file}:${line}` : file;
  const href = sourceHref(file, line);
  return href
    ? el("a", { href, className: "src" }, label)
    : el("code", { className: "src" }, label);
}
