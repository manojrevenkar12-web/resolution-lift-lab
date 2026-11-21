import { useCallback, useEffect, useState } from "react";

export type SetName = "heldout" | "dev";

export interface TopicRow { ari: number; nmi: number; assigned: number; clusters: number; gaps_found: number; workspaces: number; false_gap_flags: number; ari_en: number; ari_de: number }
export interface ResolutionRow { mae: number; bias: number; p90_abs_error: number; coverage: number | null; width: number | null; units: number }
export interface LiftRow { fixes: number; estimated: number; bias: number; rmse: number; coverage: number; claimed_wins: number; false_wins: number; pooled: number; pooled_low: number; pooled_high: number; true_mean: number }
export interface Report {
  set: SetName;
  seeds: number[];
  config_sha: string;
  generated_at: string;
  facts: {
    conversations: number; fixes: number; audits: number; billed: number; billed_not_resolved: number;
    true_resolution_rate: number; billed_rate: number;
    confirmed: number; assumed_resolved: number; assumed_unresolved: number; escalated: number;
  };
  topics: Record<string, TopicRow>;
  resolution: Record<string, ResolutionRow>;
  lift: Record<string, LiftRow>;
  lift_points: { design: string; estimate: number; truth: number }[];
  mde_median: number;
}

export type LoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "missing" }
  | { status: "ready"; report: Report };

/** Loads reports/<set>.json as written by `python -m rll.cli run`. */
export function useReport(set: SetName): { state: LoadState; retry: () => void } {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    setState({ status: "loading" });
    fetch(`./reports/${set}.json`)
      .then(async (res) => {
        if (!live) return;
        if (res.status === 404) return setState({ status: "missing" });
        if (!res.ok) throw new Error(`The report did not load (HTTP ${res.status}).`);
        const report = (await res.json()) as Report;
        if (live) setState({ status: "ready", report });
      })
      .catch((err: unknown) => live && setState({ status: "error", message: err instanceof Error ? err.message : String(err) }));
    return () => {
      live = false;
    };
  }, [set, attempt]);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return { state, retry };
}

export const pct = (x: number | null | undefined, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
export const pts = (x: number | null | undefined, d = 1) => {
  if (x == null) return "–";
  const v = x * 100;
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)} pts`;
};
export const num = (x: number) => new Intl.NumberFormat("en-GB").format(x);
