import bakeJson from "./data/bake.json";
import registryJson from "./data/registry.json";
import type { Bake, Figure, RegistryFile, RegistryRow } from "./data/types";
import { latestPerId } from "./lib/registry";

export const bake = bakeJson as unknown as Bake;
const registryFile = registryJson as unknown as RegistryFile;

/** The registry is baked with latest-line-wins already; re-applying is a no-op. */
export const registry: RegistryRow[] = latestPerId(registryFile.rows);
export const registrySource: string = registryFile.source;

const byId = new Map<string, Figure>(bake.figures.map((f) => [f.id, f]));

export function figure(id: string): Figure {
  const f = byId.get(id);
  if (!f) throw new Error(`missing figure: ${id}`);
  return f;
}

export function num(id: string): number {
  const v = figure(id).value;
  if (typeof v !== "number") throw new Error(`figure ${id} is not numeric`);
  return v;
}

export function flag(id: string): boolean {
  const v = figure(id).value;
  if (typeof v !== "boolean") throw new Error(`figure ${id} is not boolean`);
  return v;
}
