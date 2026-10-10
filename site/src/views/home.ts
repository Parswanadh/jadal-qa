import { el } from "../dom";
import { bake, num, flag } from "../data";
import { fmtCI, fmtInt, fmtPct } from "../lib/format";
import { funnelStrip } from "./strip";


function tile(value: string, label: string, note: string, tone: "good" | "warn" | "plain"): HTMLElement {
  return el(
    "article",
    { className: `tile ${tone}` },
    el("p", { className: "tile-value" }, value),
    el("h3", { className: "tile-label" }, label),
    el("p", { className: "tile-note" }, note),
  );
}

export function homeView(still: number | null): HTMLElement {
  const gatesMet = flag("gate_auroc") && flag("gate_ci");
  const gateNote = gatesMet
    ? "Pre-registered gates met."
    : "Pre-registered gates NOT met (AUROC >= 0.85, CI-low >= 0.72).";
  const indomainCI = fmtCI(
    num("indomain_auroc"),
    num("indomain_ci_lo"),
    num("indomain_ci_hi"),
  );

  const r = bake.reject.record;
  const receipt = el(
    "figure",
    { className: "receipt-card", "aria-label": "A real reject record from the PoC run" },
    el("figcaption", {}, el("span", { className: "mono" }, r.window_id), el("span", { className: "pill reject" }, "reject")),
    el(
      "pre",
      {},
      `stage     ${r.stage}\n`,
      `reason    ${r.reasons[0]?.code ?? ""}\n`,
      `evidence  s1_score ${r.reasons[0]?.evidence.value} ${r.reasons[0]?.evidence.op} ${r.reasons[0]?.evidence.threshold}\n`,
      `receipt   ${r.receipt_sha256}   content hash, not a signature`,
    ),
    el("p", { className: "fineprint" }, "Real record from the 1,587-window PoC run; the full file is local."),
  );

  return el(
    "div",
    { className: "page home" },
    el(
      "section",
      { className: "hero" },
      el("div", { className: "hero-copy" },
      el("p", { className: "kicker" }, "Jadal QA · results explorer"),
      el("h1", {}, "QA that shows its work, including where it fails"),
      el(
        "p",
        { className: "pitch" },
        "Each 10-second head-camera window gets accept, reject or review, with the rule or score that decided it and a content-hashed receipt. The numbers below come from files in this repository.",
      ),
      el(
        "div",
        { className: "cta" },
        el("a", { className: "btn primary", href: "#/funnel" }, "Walk the funnel"),
        el("a", { className: "btn", href: "#/scoreboard" }, "Read the scoreboard"),
      ),
      ),
      receipt,
    ),
    funnelStrip(still),
    el(
      "section",
      { className: "glance", "aria-label": "At a glance" },
      el("h2", { className: "eyebrow" }, "At a glance"),
      el(
        "div",
        { className: "tiles" },
        tile(
          fmtInt(num("windows")),
          "windows scored",
          `${num("hours").toFixed(3)} h of video from ${num("factories")} factories, PoC run`,
          "plain",
        ),
        tile(
          fmtPct(num("reject_share")),
          "rejected",
          `${fmtInt(num("reject"))} of ${fmtInt(num("windows"))} windows; descriptive, unlabelled`,
          "plain",
        ),
        tile(
          num("indomain_auroc").toFixed(3),
          "in-domain AUROC",
          `IndEgo, n=${num("indomain_n")} windows, ${indomainCI}. ${gateNote}`,
          "warn",
        ),
        tile(
          num("ood_cascade").toFixed(3),
          "factory AUROC",
          `Egocentric-10K, n=160 (138 good / ${num("ood_nbad")} bad). Near chance; CIs ignore clip and factory correlation.`,
          "warn",
        ),
      ),
      el(
        "p",
        { className: "fineprint" },
        `${bake.registry_unique} experiments on main (experiments/registry.jsonl, latest line per id). `,
        el("a", { href: "#/experiments" }, "Browse the log"),
        ".",
      ),
    ),
  );
}
