import type { RegistryRow } from "../data/types";

/** Latest line per id wins; first-seen order is kept. Input is ordered by file line. */
export function latestPerId(rows: readonly RegistryRow[]): RegistryRow[] {
  const order: string[] = [];
  const latest = new Map<string, RegistryRow>();
  for (const row of rows) {
    if (!latest.has(row.id)) order.push(row.id);
    latest.set(row.id, row);
  }
  return order.map((id) => latest.get(id)!);
}

/** Parse JSONL text into rows (blank lines skipped). Used by the test fixture check. */
export function parseJsonl(text: string): unknown[] {
  return text
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as unknown);
}
