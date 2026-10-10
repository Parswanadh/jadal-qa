import { el } from "../dom";
import { fmtInt } from "../lib/format";
import { num } from "../data";

const STOPS = [
  { key: "in", name: "Window in", pos: 0 },
  { key: "s0", name: "S0 exact rules", pos: 0.25 },
  { key: "s1", name: "S1 head", pos: 0.5 },
  { key: "s2", name: "S2 review", pos: 0.75 },
  { key: "out", name: "Verdict", pos: 1 },
] as const;

const DURATION_MS = 3200;

export function reducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

function subline(key: (typeof STOPS)[number]["key"]): string {
  switch (key) {
    case "in":
      return "one 10-second window";
    case "s0":
      return `${num("s0")} fired`;
    case "s1":
      return `${fmtInt(num("s1"))} decided`;
    case "s2":
      return `${num("s2")} sent on`;
    case "out":
      return `${num("accept")} accept · ${num("reject")} reject · ${num("review")} review`;
  }
}

/**
 * The funnel strip: a window token travels S0 -> S1 -> S2 -> verdict and each
 * stage lights when the token reaches it. `still` (0..1) freezes the frame for
 * screenshots; reduced motion jumps straight to the settled state.
 */
export function funnelStrip(still: number | null): HTMLElement {
  const nodes = STOPS.map((stop) =>
    el(
      "li",
      { className: "node", style: `--pos:${stop.pos}` },
      el("span", { className: "dot", "aria-hidden": "true" }),
      el("span", { className: "name" }, stop.name),
      el("span", { className: "sub" }, subline(stop.key)),
    ),
  );
  const token = el("span", { className: "token", "aria-hidden": "true" });
  const track = el("ol", { className: "stops" }, ...nodes);
  const root = el(
    "figure",
    { className: "strip", role: "img", "aria-label": stripLabel() },
    el("div", { className: "lane" }, el("div", { className: "rail" }), token, track),
  );

  const setT = (t: number): void => {
    const clamped = Math.min(1, Math.max(0, t));
    root.style.setProperty("--t", clamped.toFixed(4));
    nodes.forEach((node, i) => {
      const stop = STOPS[i];
      if (stop) node.classList.toggle("lit", clamped >= stop.pos - 0.001);
    });
  };

  if (still !== null || reducedMotion()) {
    setT(still ?? 1);
    return root;
  }
  setT(0);
  const start = performance.now();
  const frame = (now: number): void => {
    const raw = Math.min(1, (now - start) / DURATION_MS);
    const eased = raw < 0.5 ? 2 * raw * raw : 1 - Math.pow(-2 * raw + 2, 2) / 2;
    setT(eased);
    if (raw < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return root;
}

function stripLabel(): string {
  return (
    `Funnel: a window passes S0 (${num("s0")} fired), S1 (${fmtInt(num("s1"))} decided) ` +
    `and S2 (${num("s2")} sent on). Verdicts: ${num("accept")} accept, ${num("reject")} reject, ` +
    `${num("review")} review.`
  );
}
