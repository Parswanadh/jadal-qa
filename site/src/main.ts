import "./style.css";
import { el } from "./dom";
import { parseHash, parseStill } from "./lib/route";
import { homeView } from "./views/home";
import { funnelView } from "./views/funnel";
import { scoreboardView } from "./views/scoreboard";
import { experimentsView, readState, stateToHash, type ExpState } from "./views/experiments";
import { efficiencyView } from "./views/efficiency";
import { footer } from "./views/footer";

const NAV = [
  { href: "#/", label: "Overview", route: "home" },
  { href: "#/funnel", label: "Funnel", route: "funnel" },
  { href: "#/scoreboard", label: "Scoreboard", route: "scoreboard" },
  { href: "#/experiments", label: "Experiments", route: "experiments" },
  { href: "#/efficiency", label: "Efficiency", route: "efficiency" },
] as const;

function header(active: string): HTMLElement {
  return el(
    "header",
    { className: "topbar" },
    el("div", { className: "wrap bar" },
      el("a", { className: "brand", href: "#/" }, "Jadal QA"),
      el("nav", { "aria-label": "Main" }, ...NAV.map((n) =>
        el("a", { href: n.href, "aria-current": n.route === active ? "page" : undefined }, n.label),
      )),
    ),
  );
}

function render(): void {
  const app = document.getElementById("app");
  if (!app) throw new Error("missing #app");
  const { route, params } = parseHash(location.hash);
  const still = parseStill(params);

  let page: HTMLElement;
  switch (route) {
    case "home":
      page = homeView(still);
      break;
    case "funnel": {
      const stage = params.get("stage") ?? "s0";
      const s = stage === "s1" || stage === "s2" ? stage : "s0";
      page = funnelView(still, s);
      break;
    }
    case "scoreboard":
      page = scoreboardView();
      break;
    case "experiments": {
      const state = readState(params);
      page = experimentsView(state, (next: ExpState) => {
        history.replaceState(null, "", stateToHash(next));
      });
      break;
    }
    case "efficiency":
      page = efficiencyView();
      break;
  }

  const main = el("main", { id: "main", className: "wrap", tabIndex: -1 }, page);
  app.replaceChildren(header(route), main, el("div", { className: "wrap" }, footer()));
  const target = params.get("id");
  if (route === "experiments" && target) {
    document.getElementById(`exp-${target}`)?.scrollIntoView({ block: "center" });
  } else {
    window.scrollTo(0, 0);
  }
}

window.addEventListener("hashchange", render);
render();
