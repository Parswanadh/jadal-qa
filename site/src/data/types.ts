export type Path = (string | number)[];

export interface Source {
  file: string;
  keys?: Path;
  literal?: string;
  line: number | null;
}

export interface Figure {
  id: string;
  label: string;
  value: number | boolean;
  display: string;
  source: Source;
}

export interface S0Rule {
  code: string;
  feature: string;
  op: "<" | ">";
  threshold: number;
  source: { file: string; literal: string; line: number | null };
}

export interface RegistryRow {
  id: string;
  track: string;
  title: string;
  status: string;
  verdict: string | null;
  summary: string;
  results: string | null;
  finished: string | null;
  line: number;
}

export interface RejectRecord {
  window_id: string;
  verdict: string;
  stage: string;
  reasons: { code: string; evidence: { feature: string; value: number; op: string; threshold: number } }[];
  score: number;
  features_digest: string;
  receipt_sha256: string;
}

export interface E101Row {
  fps: number;
  frames: number;
  auroc: number;
  ci: [number, number];
  auroc_spatial: number;
  ci_spatial: [number, number];
  feat_ms: number;
}

export interface Bake {
  figures: Figure[];
  s0_rules: S0Rule[];
  reject: { record: RejectRecord; source: { file: string; literal: string; line: number | null } };
  e101: {
    n: number;
    n_good: number;
    n_bad: number;
    label_source: string;
    claim_status: string;
    rows: E101Row[];
    source: Source;
  };
  res03: RegistryRow;
  registry_lines: number;
  registry_unique: number;
}

export interface RegistryFile {
  source: string;
  rows: RegistryRow[];
}
