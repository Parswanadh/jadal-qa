import { describe, expect, it } from "vitest";
import { latestPerId, parseJsonl } from "./registry";
import { parseHash, parseStill } from "./route";
import { countBy, matchesFilter, matchesQuery } from "./filters";
import { fmtCI, fmtInt, fmtPct } from "./format";
import { stateToHash, readState } from "../views/experiments";
import type { RegistryRow } from "../data/types";
import { REPO_BLOB, sourceHref } from "./links";

function row(id: string, over: Partial<RegistryRow> = {}): RegistryRow {
  return {
    id,
    track: id.split(".")[0] ?? id,
    title: `title ${id}`,
    status: "done",
    verdict: null,
    summary: "",
    results: null,
    finished: null,
    line: 1,
    ...over,
  };
}

describe("registry: latest line per id wins", () => {
  const fixture = [
    '{"id":"E1.01","status":"running","verdict":null}',
    '{"id":"E2.01","status":"done","verdict":"drop"}',
    '{"id":"E1.01","status":"done","verdict":"adopt"}',
    "",
    '{"id":"E3.01","status":"done","verdict":"maybe"}',
  ].join("\n");

  it("keeps first-seen order and the last line for each id", () => {
    const rows = parseJsonl(fixture) as { id: string; status: string; verdict: string | null }[];
    const merged = latestPerId(
      rows.map((r, i) => row(r.id, { status: r.status, verdict: r.verdict, line: i + 1 })),
    );
    expect(merged.map((r) => r.id)).toEqual(["E1.01", "E2.01", "E3.01"]);
    expect(merged[0]?.verdict).toBe("adopt");
    expect(merged[0]?.status).toBe("done");
  });
});

describe("hash routing", () => {
  it("reads the query from the hash, not from location.search", () => {
    const { route, params } = parseHash("#/experiments?filter=negative&q=fps");
    expect(route).toBe("experiments");
    expect(params.get("filter")).toBe("negative");
    expect(params.get("q")).toBe("fps");
  });

  it("falls back to home for empty and unknown routes", () => {
    expect(parseHash("").route).toBe("home");
    expect(parseHash("#/").route).toBe("home");
    expect(parseHash("#/nope?x=1").route).toBe("home");
  });

  it("clamps the still progress and rejects non-numbers", () => {
    expect(parseStill(new URLSearchParams("still=0.4"))).toBe(0.4);
    expect(parseStill(new URLSearchParams("still=3"))).toBe(1);
    expect(parseStill(new URLSearchParams("still=-2"))).toBe(0);
    expect(parseStill(new URLSearchParams("still=abc"))).toBeNull();
    expect(parseStill(new URLSearchParams(""))).toBeNull();
  });

  it("round-trips experiment state through the hash", () => {
    const state = readState(new URLSearchParams("filter=negative&q=fps&id=E1.01"));
    expect(stateToHash(state)).toBe("#/experiments?filter=negative&q=fps&id=E1.01");
    expect(stateToHash(readState(new URLSearchParams("filter=bogus")))).toBe("#/experiments");
  });
});

describe("filters derive from registry fields", () => {
  const rows = [
    row("E1.01", { verdict: "adopt" }),
    row("E2.01", { verdict: "drop" }),
    row("Z3.02", { verdict: "no" }),
    row("E5.04", { verdict: "inconclusive" }),
    row("E4.01", { status: "running", verdict: null }),
  ];

  it("negative means verdict drop or no", () => {
    expect(countBy(rows, "negative")).toBe(2);
    expect(matchesFilter(rows[0]!, "negative")).toBe(false);
  });

  it("open means not done", () => {
    expect(countBy(rows, "open")).toBe(1);
  });

  it("search matches id, title, summary and track, case-insensitively", () => {
    expect(matchesQuery(row("E1.01", { summary: "FPS sweep" }), "fps")).toBe(true);
    expect(matchesQuery(row("E1.01"), "  ")).toBe(true);
    expect(matchesQuery(row("E1.01"), "zzz")).toBe(false);
  });
});

describe("formatting", () => {
  it("formats an AUROC with its interval to three places", () => {
    expect(fmtCI(0.8336, 0.7149, 0.9136)).toBe("0.834 [0.715, 0.914]");
  });

  it("formats integers with separators and shares as percents", () => {
    expect(fmtInt(1587)).toBe("1,587");
    expect(fmtPct(0.4423)).toBe("44.2%");
  });
});

describe("source links", () => {
  it("are off unless VITE_REPO_BLOB is set at build time", () => {
    expect(REPO_BLOB).toBe("");
    expect(sourceHref("README.md", 89)).toBeNull();
  });
});
