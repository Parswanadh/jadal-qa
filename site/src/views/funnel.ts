import { repoHref } from "../lib/links";
import { el } from "../dom";
import { bake, figure, num } from "../data";
import { fmtInt } from "../lib/format";
import { funnelStrip } from "./strip";

type StageId = "s0" | "s1" | "s2";

const STAGES: { id: StageId; title: string; short: string }[] = [
  { id: "s0", title: "S0 · exact rules", short: "S0 rules" },
  { id: "s1", title: "S1 · whole-window head", short: "S1 head" },
  { id: "s2", title: "S2 · review queue", short: "S2 review" },
];

function ruleTable(): HTMLElement {
  const rows = bake.s0_rules.map((r) =>
    el(
      "tr",
      {},
      el("td", {}, el("code", {}, r.code)),
      el("td", {}, el("code", {}, r.feature)),
      el("td", {}, `${r.op} ${r.threshold}`),
      el("td", {}, el("a", { href: repoHref(r.source.file, r.source.line), className: "src" }, `${r.source.file}:${r.source.line ?? ""}`)),
    ),
  );
  return el(
    "table",
    { className: "rules" },
    el("thead", {}, el("tr", {}, el("th", {}, "Code"), el("th", {}, "Probe"), el("th", {}, "Fires when"), el("th", {}, "Source"))),
    el("tbody", {}, ...rows),
  );
}

function stageBody(stage: StageId): HTMLElement[] {
  if (stage === "s0") {
    return [
      el("p", {}, "Five exact rules on simple probe statistics. A firing rejects the window and records the feature, value, operator and threshold it failed."),
      el("p", { className: "stat" }, `Fired on ${num("s0")} of ${fmtInt(num("windows"))} PoC windows, so every decision fell through to S1.`),
      ruleTable(),
      el("pre", { className: "evidence", "aria-label": "S0 evidence format" }, '{ "code": "BLACK", "evidence": { "feature": "luma_p95", "value": <measured>, "op": "<", "threshold": 0.03 } }'),
    ];
  }
  if (stage === "s1") {
    const record = JSON.stringify(bake.reject.record, null, 2);
    return [
      el("p", {}, `A whole-window logistic head on 26 per-pixel spectral and temporal features. Operating threshold ${num("s1_threshold")}. Of the ${fmtInt(num("s1"))} windows S1 decided, ${fmtInt(num("accept"))} were accepted and ${fmtInt(num("reject"))} rejected.`),
      el("p", {}, "An S1 reject carries one piece of evidence: the score against the threshold."),
      el("pre", { className: "evidence", "aria-label": "S1 evidence format" }, '"evidence": { "feature": "s1_score", "value": 0.0013, "op": "<", "threshold": 0.6 }'),
      el("h3", { className: "subhead" }, "A real reject from the PoC run"),
      el("pre", { className: "receipt" }, record),
      el(
        "p",
        { className: "fineprint" },
        "Receipts are content-hashed, not signed. Source: ",
        el("a", { href: repoHref(bake.reject.source.file, bake.reject.source.line) }, `${bake.reject.source.file}:${bake.reject.source.line}`),
        ". Full PoC file is local and not in this repo.",
      ),
    ];
  }
  return [
    el("p", {}, `Windows whose S1 score sits inside the margin around 0.5 (|p − 0.5| < ${num("s1_margin")}) go to review rather than a verdict.`),
    el("p", { className: "stat" }, `${num("s2")} windows handed on in the PoC run. The heavy review tier was not run in the PoC.`),
    el("p", {}, "Review is a hand-off state, not an accept. It is counted separately in every total."),
  ];
}

export function funnelView(still: number | null, stage: StageId): HTMLElement {
  const current = STAGES.find((s) => s.id === stage) ?? STAGES[0]!;
  const tabs = STAGES.map((s) =>
    el(
      "a",
      {
        className: "stage-tab",
        href: `#/funnel?stage=${s.id}`,
        "aria-current": s.id === current.id ? "true" : undefined,
        "aria-pressed": s.id === current.id ? "true" : "false",
      },
      el("span", { className: "stage-id" }, s.id.toUpperCase()),
      el("span", {}, s.short),
    ),
  );
  return el(
    "div",
    { className: "page funnel" },
    el(
      "section",
      { className: "intro" },
      el("p", { className: "kicker" }, "How a window is decided"),
      el("h1", {}, "Three stages, and every reject names its reason"),
      el("p", { className: "pitch" }, "Select a stage to see its rules, its evidence format and a real reject. Windows move left to right; a window leaves at the stage that decides it."),
    ),
    funnelStrip(still),
    el("nav", { className: "stage-tabs", "aria-label": "Funnel stages" }, ...tabs),
    el(
      "section",
      { className: "stage-panel", "aria-live": "polite" },
      el("h2", { className: "panel-title" }, current.title),
      ...stageBody(current.id),
    ),
    el(
      "p",
      { className: "fineprint" },
      `Counts: ${fmtInt(num("windows"))} windows = ${num("accept")} accept + ${num("reject")} reject + ${num("review")} review. Source: `,
      el("a", { href: repoHref(figure("windows").source.file) }, figure("windows").source.file),
      ".",
    ),
  );
}
