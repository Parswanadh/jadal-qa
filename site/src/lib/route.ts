export const ROUTES = ["home", "funnel", "scoreboard", "experiments", "efficiency"] as const;
export type Route = (typeof ROUTES)[number];

export interface ParsedHash {
  route: Route;
  params: URLSearchParams;
}

/**
 * Parses location.hash such as "#/experiments?filter=negative&q=fps".
 * Query parameters live inside the hash, so location.search is never read.
 */
export function parseHash(hash: string): ParsedHash {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const qAt = raw.indexOf("?");
  const pathPart = qAt >= 0 ? raw.slice(0, qAt) : raw;
  const query = qAt >= 0 ? raw.slice(qAt + 1) : "";
  const name = pathPart.replace(/^\/+/, "").replace(/\/+$/, "");
  const route: Route = (ROUTES as readonly string[]).includes(name) ? (name as Route) : "home";
  return { route: name === "" ? "home" : route, params: new URLSearchParams(query) };
}

/** Clamp a `still` progress value to [0, 1]; undefined or invalid returns null. */
export function parseStill(params: URLSearchParams): number | null {
  const raw = params.get("still");
  if (raw === null) return null;
  const v = Number(raw);
  if (!Number.isFinite(v)) return null;
  return Math.min(1, Math.max(0, v));
}
