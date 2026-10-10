import { el, svg } from "../dom";
import { bake } from "../data";
import { fmt3 } from "../lib/format";
import { sourceNode } from "../lib/links";

const W = 640;
const H = 300;
const PAD = { l: 52, r: 16, t: 16, b: 44 };

export function efficiencyView(): HTMLElement {
  const rows = bake.e101.rows;
  const inner = W - PAD.l - PAD.r - 60;
  const xs = rows.map((_, i) => PAD.l + 30 + (i * inner) / Math.max(1, rows.length - 1));
  const plotTop = PAD.t;
  const yPix = (v: number): number => plotTop + (1 - (v - 0.5) / 0.5) * (H - PAD.t - PAD.b);

  const grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1].map((v) =>
    svg("g", {}, svg("line", { x1: PAD.l, x2: W - PAD.r, y1: yPix(v), y2: yPix(v), class: "grid" }), svg("text", { x: PAD.l - 8, y: yPix(v) + 4, "text-anchor": "end", class: "tick" }, v.toFixed(1))),
  );
  const chance = svg("line", { x1: PAD.l, x2: W - PAD.r, y1: yPix(0.5), y2: yPix(0.5), class: "chance" });

  const series = (key: "pixel" | "spatial"): SVGElement[] => {
    const out: SVGElement[] = [];
    rows.forEach((r, i) => {
      const x = xs[i] ?? 0;
      const ci = key === "pixel" ? r.ci : r.ci_spatial;
      const au = key === "pixel" ? r.auroc : r.auroc_spatial;
      out.push(svg("line", { x1: x, x2: x, y1: yPix(ci[0]), y2: yPix(ci[1]), class: `whisk ${key}` }));
      out.push(svg("circle", { cx: x, cy: yPix(au), r: 4.5, class: `pt ${key}` }));
    });
    return out;
  };

  const labels = rows.map((r, i) =>
    svg("text", { x: xs[i] ?? 0, y: H - 20, "text-anchor": "middle", class: "tick" }, `${r.fps} fps`),
  );
  const axisTitle = svg("text", { x: (PAD.l + W - PAD.r) / 2, y: H - 2, "text-anchor": "middle", class: "axis-title" }, "frames per second used (IndEgo, n=150)");

  const chart = svg(
    "svg",
    { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "AUROC by frame rate with 95% intervals. Descriptive, IndEgo only." , class: "e101" },
    ...grid,
    chance,
    ...series("pixel"),
    ...series("spatial"),
    ...labels,
    axisTitle,
  );

  const tableRows = rows.map((r) =>
    el(
      "tr",
      {},
      el("td", {}, `${r.fps}`),
      el("td", {}, `${r.frames}`),
      el("td", {}, `${fmt3(r.auroc)} [${fmt3(r.ci[0])}, ${fmt3(r.ci[1])}]`),
      el("td", {}, `${fmt3(r.auroc_spatial)}`),
      el("td", {}, `${r.feat_ms.toFixed(0)} ms`),
    ),
  );

  return el(
    "div",
    { className: "page efficiency" },
    el(
      "section",
      { className: "intro" },
      el("p", { className: "kicker" }, "Efficiency"),
      el("h1", {}, "Fewer frames, similar AUROC, not resolvable at this n"),
      el("p", { className: "pitch" }, "Descriptive only. The fps sweep (E1.01) uses 150 IndEgo windows with 50 bad. The intervals are about ±0.1 wide, so differences between frame rates cannot be resolved, and no equivalence margin was set in advance."),
    ),
    el(
      "section",
      { className: "chart-card" },
      el("h2", { className: "eyebrow" }, "E1.01 · fps sweep, leave-one-clip-out"),
      chart,
      el("div", { className: "legend" }, el("span", { className: "legend-item" }, el("span", { className: "legend-point pixel" }), "pixel view, 26 features"), el("span", { className: "legend-item" }, el("span", { className: "legend-point spatial" }), "spatial only, 18 features")),
      el(
        "div",
        { className: "table-wrap" },
        el(
          "table",
          { className: "rules" },
          el("thead", {}, el("tr", {}, el("th", {}, "fps"), el("th", {}, "frames"), el("th", {}, "AUROC [95% CI]"), el("th", {}, "spatial AUROC"), el("th", {}, "feature time / window"))),
          el("tbody", {}, ...tableRows),
        ),
      ),
      el("p", { className: "fineprint" }, `${bake.e101.claim_status}. Source: `, sourceNode(bake.e101.source.file), "."),
    ),
    el(
      "section",
      { className: "chart-card" },
      el("h2", { className: "eyebrow" }, `RES.03 · ${bake.res03.title}`),
      el("p", {}, bake.res03.summary),
      el("p", { className: "fineprint" }, "Source: ", sourceNode("experiments/registry.jsonl", bake.res03.line), " (registry line; results JSON not in repo)."),
    ),
  );
}
