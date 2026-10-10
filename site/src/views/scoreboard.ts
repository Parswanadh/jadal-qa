import { sourceNode } from "../lib/links";
import { el } from "../dom";
import { figure, num, flag } from "../data";
import { fmt3 } from "../lib/format";

interface Row {
  label: string;
  scope: string;
  auroc: number;
  lo: number;
  hi: number;
  tag: "exploratory" | "near chance";
  note: string;
  source: string;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function rowsFromData(): Row[] {
  const ood = figure("ood_ensemble").source;
  return [
    {
      label: "In-domain · IndEgo · ensemble",
      scope: "n=150 windows, 7 clips, nested leave-one-clip-out",
      auroc: num("indomain_auroc"),
      lo: num("indomain_ci_lo"),
      hi: num("indomain_ci_hi"),
      tag: "exploratory",
      note: "One labeller. Pre-registered gates not met.",
      source: "eval/results/ensemble/nested_loco.json",
    },
    {
      label: "Factory · frozen ensemble",
      scope: `n=160 windows (138 good / ${num("ood_nbad")} bad), 47 factories`,
      auroc: num("ood_ensemble"),
      lo: num("ood_ens_lo"),
      hi: num("ood_ens_hi"),
      tag: "near chance",
      note: "Labelled by one person; results file pending merge.",
      source: ood.file,
    },
    {
      label: "Factory · PoC cascade (shipped)",
      scope: `same 160 windows; rejects ${num("ood_rejects")} of 138 good (38%)`,
      auroc: num("ood_cascade"),
      lo: num("ood_casc_lo"),
      hi: num("ood_casc_hi"),
      tag: "near chance",
      note: `Also rejects ${num("ood_bad_rejected")} of ${num("ood_nbad")} bad windows.`,
      source: figure("ood_cascade").source.file,
    },
  ];
}

function axisPos(v: number): string {
  return pct(Math.min(1, Math.max(0, v)));
}

function row(r: Row): HTMLElement {
  const whisker = el("span", {
    className: "whisker",
    style: `left:${axisPos(r.lo)};width:calc(${axisPos(r.hi)} - ${axisPos(r.lo)})`,
  });
  const point = el("span", { className: "point", style: `left:${axisPos(r.auroc)}` });
  const track = el(
    "div",
    {
      className: "bar-track",
      role: "img",
      "aria-label": `${r.label}: AUROC ${fmt3(r.auroc)}, 95% CI ${fmt3(r.lo)} to ${fmt3(r.hi)}`,
    },
    el("span", { className: "chance", "aria-hidden": "true" }),
    whisker,
    point,
  );
  return el(
    "article",
    { className: `score-row ${r.tag === "near chance" ? "bad" : "mid"}` },
    el(
      "header",
      {},
      el("h3", {}, r.label),
      el("span", { className: "tag" }, r.tag),
    ),
    el("p", { className: "scope" }, r.scope),
    track,
    el(
      "p",
      { className: "numbers" },
      el("strong", {}, fmt3(r.auroc)),
      ` [${fmt3(r.lo)}, ${fmt3(r.hi)}]`,
      el("span", { className: "note" }, ` · ${r.note}`),
    ),
    el("p", { className: "fineprint" }, "Source: ", sourceNode(r.source)),
  );
}

function axis(): HTMLElement {
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    el("span", { className: "tick", style: `left:${axisPos(v)}` }, v.toFixed(2)),
  );
  return el("div", { className: "axis", "aria-hidden": "true" }, ...ticks);
}

export function scoreboardView(): HTMLElement {
  const gatesMet = flag("gate_auroc") && flag("gate_ci");
  const rows = rowsFromData();
  return el(
    "div",
    { className: "page scoreboard" },
    el(
      "section",
      { className: "intro" },
      el("p", { className: "kicker" }, "Honest scoreboard"),
      el("h1", {}, "One axis from 0 to 1. Chance is 0.5."),
      el("p", { className: "pitch" }, "The in-domain result (IndEgo, exploratory) sits above chance. On factory footage, both scorers sit close to chance. Every row is drawn on the same axis, with its 95% interval."),
    ),
    el(
      "section",
      { className: "chart", "aria-label": "AUROC with 95% intervals on one axis" },
      el(
        "div",
        { className: "legend" },
        el("span", { className: "legend-item" }, el("span", { className: "legend-chance" }), "chance 0.5"),
        el("span", { className: "legend-item" }, el("span", { className: "legend-point" }), "AUROC"),
        el("span", { className: "legend-item" }, el("span", { className: "legend-whisker" }), "95% interval"),
      ),
      axis(),
      ...rows.map(row),
    ),
    el(
      "section",
      { className: "gates" },
      el("h2", { className: "eyebrow" }, "Pre-registered gates, in-domain"),
      el(
        "p",
        {},
        `AUROC >= 0.85: observed ${fmt3(num("indomain_auroc"))}, ${flag("gate_auroc") ? "met" : "not met"}. `,
        `CI-low >= 0.72: observed ${fmt3(num("indomain_ci_lo"))}, ${flag("gate_ci") ? "met" : "not met"}. `,
        gatesMet ? "Both gates met." : "The gates are not met.",
      ),
      el(
        "p",
        { className: "fineprint" },
        "Window-bootstrap intervals are optimistic: they treat windows as independent and ignore clip and factory correlation.",
      ),
      el(
        "p",
        { className: "fineprint" },
        "The 60-window precursor (n_bad = 9) is superseded and not charted. Rates from the 1,587-window PoC run are descriptive, not accuracy.",
      ),
    ),
  );
}
