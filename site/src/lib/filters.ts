import type { RegistryRow } from "../data/types";

export const FILTERS = [
  "all",
  "adopted",
  "maybe",
  "negative",
  "inconclusive",
  "open",
  "withdrawn",
] as const;
export type Filter = (typeof FILTERS)[number];

const ADOPTED = new Set(["adopt", "adopt-with-caveats", "yes"]);
const NEGATIVE = new Set(["drop", "no"]);

export function isFilter(value: string | null): value is Filter {
  return value !== null && (FILTERS as readonly string[]).includes(value);
}

/**
 * Filters are derived from the registry's own fields:
 * negative = verdict drop or no; withdrawn = the word appears in any field.
 */
export function matchesFilter(row: RegistryRow, filter: Filter): boolean {
  switch (filter) {
    case "all":
      return true;
    case "adopted":
      return row.verdict !== null && ADOPTED.has(row.verdict);
    case "maybe":
      return row.verdict === "maybe";
    case "negative":
      return row.verdict !== null && NEGATIVE.has(row.verdict);
    case "inconclusive":
      return row.verdict === "inconclusive";
    case "open":
      return row.status !== "done";
    case "withdrawn":
      return `${row.title} ${row.summary}`.toLowerCase().includes("withdrawn");
  }
}

export function matchesQuery(row: RegistryRow, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (q === "") return true;
  return `${row.id} ${row.title} ${row.summary} ${row.track}`.toLowerCase().includes(q);
}

export function countBy(rows: readonly RegistryRow[], filter: Filter): number {
  return rows.filter((r) => matchesFilter(r, filter)).length;
}
