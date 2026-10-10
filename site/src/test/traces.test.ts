import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import bakeJson from "../data/bake.json";
import registryJson from "../data/registry.json";
import type { Bake, Figure, Path, RegistryFile } from "../data/types";

// Repo root, from site/src/test/ up three levels.
const REPO = new URL("../../../", import.meta.url);
const bake = bakeJson as unknown as Bake;
const registry = registryJson as unknown as RegistryFile;

function text(file: string): string {
  return readFileSync(new URL(file, REPO), "utf-8");
}

function walk(root: unknown, keys: Path): unknown {
  let node: unknown = root;
  for (const k of keys) {
    node = (node as Record<string | number, unknown>)[k];
  }
  return node;
}

describe("every baked figure traces to its source file", () => {
  for (const fig of bake.figures as Figure[]) {
    it(`${fig.id} <- ${fig.source.file}`, () => {
      const src = fig.source;
      if (src.keys) {
        const parsed: unknown = JSON.parse(text(src.file));
        expect(walk(parsed, src.keys)).toEqual(fig.value);
      } else {
        expect(src.literal, "literal source").toBeTruthy();
        expect(text(src.file)).toContain(src.literal!);
        if (src.line !== null) {
          const line = text(src.file).split("\n")[src.line - 1] ?? "";
          expect(line).toContain(src.literal!);
        }
      }
    });
  }
});

describe("other baked blocks trace to the repo", () => {
  it("S0 rules match jadal/poc/cascade.py", () => {
    for (const rule of bake.s0_rules) {
      expect(text(rule.source.file)).toContain(rule.source.literal);
    }
    expect(bake.s0_rules.map((r) => r.code)).toEqual(["BLACK", "TOO_DARK", "LENS_BLOCKED", "OVEREXPOSED", "FROZEN"]);
  });

  it("the reject record is the one quoted in README.md", () => {
    const readme = text(bake.reject.source.file);
    expect(readme).toContain(bake.reject.source.literal);
    expect(bake.reject.record.window_id).toBe("f069_w004_c00077_t0000");
    expect(bake.reject.record.reasons[0]?.code).toBe("S1_UNUSABLE");
    expect(bake.reject.record.reasons[0]?.evidence.value).toBe(0.0013);
    expect(bake.reject.record.receipt_sha256).toBe("ca3ae717…");
  });

  it("fps sweep rows equal eval/results/exp_fps/e1_01.json", () => {
    const raw = JSON.parse(text(bake.e101.source.file)) as { rows: { fps: number; auroc: number; ci95: number[] }[] };
    expect(bake.e101.rows.map((r) => r.fps)).toEqual(raw.rows.map((r) => r.fps));
    bake.e101.rows.forEach((r, i) => {
      expect(r.auroc).toBe(raw.rows[i]!.auroc);
      expect(r.ci).toEqual(raw.rows[i]!.ci95);
    });
  });

  it("the registry is the latest line per id from experiments/registry.jsonl", () => {
    const lines = text("experiments/registry.jsonl").split("\n").filter((l) => l.trim() !== "");
    const latest = new Map<string, { status?: string; verdict?: string | null; summary?: string }>();
    for (const line of lines) {
      const rec = JSON.parse(line) as { id: string; status?: string; verdict?: string | null; summary?: string };
      latest.set(rec.id, rec);
    }
    expect(bake.registry_lines).toBe(lines.length);
    expect(bake.registry_unique).toBe(latest.size);
    expect(registry.rows.length).toBe(latest.size);
    for (const row of registry.rows) {
      const src = latest.get(row.id);
      expect(src, row.id).toBeDefined();
      expect(row.status).toBe(src?.status ?? "unknown");
      expect(row.verdict).toBe(src?.verdict ?? null);
    }
  });
});
