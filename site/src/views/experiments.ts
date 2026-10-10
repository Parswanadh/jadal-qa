import { el } from "../dom";
import { registry, registrySource } from "../data";
import { FILTERS, countBy, isFilter, matchesFilter, matchesQuery, type Filter } from "../lib/filters";
import type { RegistryRow } from "../data/types";

export interface ExpState {
  filter: Filter;
  query: string;
  id: string | null;
}

export function readState(params: URLSearchParams): ExpState {
  const f = params.get("filter");
  return {
    filter: isFilter(f) ? f : "all",
    query: params.get("q") ?? "",
    id: params.get("id"),
  };
}

export function stateToHash(state: ExpState): string {
  const p = new URLSearchParams();
  if (state.filter !== "all") p.set("filter", state.filter);
  if (state.query.trim() !== "") p.set("q", state.query.trim());
  if (state.id) p.set("id", state.id);
  const qs = p.toString();
  return qs === "" ? "#/experiments" : `#/experiments?${qs}`;
}

function verdictTone(verdict: string | null): string {
  if (verdict === null) return "open";
  if (["adopt", "adopt-with-caveats", "yes"].includes(verdict)) return "adopt";
  if (["drop", "no"].includes(verdict)) return "drop";
  return verdict;
}

function detail(row: RegistryRow): HTMLElement {
  return el(
    "div",
    { className: "exp-detail" },
    el("p", {}, row.summary || "No summary recorded."),
    el(
      "dl",
      {},
      el("dt", {}, "status"), el("dd", {}, row.status),
      el("dt", {}, "finished"), el("dd", {}, row.finished ?? "not recorded"),
      el("dt", {}, "results"), el("dd", {}, row.results ? el("code", {}, row.results) : "none recorded"),
      el("dt", {}, "source"), el("dd", {}, el("code", {}, `${registrySource}:${row.line}`)),
    ),
  );
}

function rowView(row: RegistryRow, open: boolean, state: ExpState): HTMLElement {
  const link = stateToHash({ ...state, id: row.id });
  return el(
    "li",
    { className: `exp-row${open ? " open" : ""}`, id: `exp-${row.id}` },
    el(
      "div",
      { className: "exp-head" },
      el("a", { className: "exp-id", href: link, "aria-current": open ? "true" : undefined }, row.id),
      el(
        "span",
        { className: "exp-title" },
        row.title === row.id ? "(no title in registry)" : row.title,
        row.summary && !open ? el("span", { className: "exp-sum" }, row.summary) : null,
      ),
      el("span", { className: `tag tone-${verdictTone(row.verdict)}` }, row.verdict ?? row.status),
    ),
    open ? detail(row) : null,
  );
}

/** Renders the filter chips and search box; `onChange` receives the new state. */
export function controls(state: ExpState, onChange: (next: ExpState) => void): HTMLElement {
  const chips = FILTERS.map((f) =>
    el(
      "a",
      {
        className: "fchip",
        href: stateToHash({ ...state, filter: f, id: null }),
        "aria-pressed": f === state.filter ? "true" : "false",
      },
      `${f} `,
      el("span", { className: "count" }, String(countBy(registry, f))),
    ),
  );
  const input = el("input", {
    type: "search",
    value: state.query,
    placeholder: "Search id, title, summary",
    "aria-label": "Search experiments",
    oninput: (ev: Event) => {
      const value = (ev.target as HTMLInputElement).value;
      onChange({ ...state, query: value });
    },
  });
  return el(
    "div",
    { className: "exp-controls" },
    el("div", { className: "chips", role: "group", "aria-label": "Filter" }, ...chips),
    input,
  );
}

export function list(state: ExpState): HTMLElement {
  const rows = registry.filter((r) => matchesFilter(r, state.filter) && matchesQuery(r, state.query));
  const items = rows.map((r) => rowView(r, r.id === state.id, state));
  return el(
    "div",
    { className: "exp-list" },
    el("p", { className: "fineprint", "aria-live": "polite" }, `${rows.length} of ${registry.length} shown.`),
    rows.length === 0 ? el("p", { className: "empty" }, "No experiment matches. Clear the search or pick another filter.") : el("ol", {}, ...items),
  );
}

export function experimentsView(state: ExpState, onChange: (next: ExpState) => void): HTMLElement {
  const listHost = el("div", { className: "list-host" }, list(state));
  const controlHost = el("div", {}, controls(state, (next) => {
    listHost.replaceChildren(list(next));
    onChange(next);
  }));
  return el(
    "div",
    { className: "page experiments" },
    el(
      "section",
      { className: "intro" },
      el("p", { className: "kicker" }, "Experiment log"),
      el("h1", {}, "Every experiment, including the ones that failed"),
      el("p", { className: "pitch" }, `${registry.length} experiments from ${registrySource}, latest line per id. Negative results are kept on purpose.`),
    ),
    controlHost,
    listHost,
  );
}
