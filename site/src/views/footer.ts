import { el } from "../dom";

/** Shown on every page: what was measured and what was not. */
export function footer(): HTMLElement {
  const measured = [
    "In-domain AUROC on IndEgo windows (exploratory, one labeller), with the pre-registered gates not met.",
    "Factory footage (160 windows, 138 good / 22 bad): frozen ensemble and PoC cascade both sit close to chance.",
    "PoC run rates on 1,587 unlabelled windows: descriptive, not accuracy.",
    "Frame-rate sweep (E1.01) on IndEgo only: descriptive, intervals too wide to separate settings.",
  ];
  const notMeasured = [
    "No compute run on Raspberry Pi hardware has been measured.",
    "Hyperspectral data was denied: the pipeline is RGB only.",
    "Certified error bounds are not achievable at n=150 because clips are correlated.",
    "Receipts are content-hashed, not signed; a hash proves self-consistency, not authorship.",
    "Licence to be announced; the project is not yet open-sourced.",
  ];
  return el(
    "footer",
    { className: "site-footer" },
    el("h2", { className: "eyebrow" }, "What we measured, what we did not"),
    el(
      "div",
      { className: "cols" },
      el("div", {}, el("h3", {}, "Measured"), el("ul", {}, ...measured.map((t) => el("li", {}, t)))),
      el("div", {}, el("h3", {}, "Not measured, or ruled out"), el("ul", {}, ...notMeasured.map((t) => el("li", {}, t)))),
    ),
    el("p", { className: "fineprint" }, "Every figure is baked from a file in the repository by scripts/build_data.py, and the pages cite their sources. Build: npm run data, then npm run build."),
  );
}
