export function fmt3(x: number): string {
  return x.toFixed(3);
}

export function fmtCI(auroc: number, lo: number, hi: number): string {
  return `${fmt3(auroc)} [${fmt3(lo)}, ${fmt3(hi)}]`;
}

export function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

export function fmtPct(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}
