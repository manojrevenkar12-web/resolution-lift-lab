import { type Report, pts } from "../../shared/data/report";
import styles from "./lift.module.css";

export const DESIGN_NAMES: Record<string, string> = {
  before_after_billed: "Billed rate, week before vs two weeks after",
  before_after_model: "Classifier estimate, before vs after",
  did: "Difference-in-differences",
  did_rtm: "Difference-in-differences, skipping the trigger week",
  ab_holdout: "A/B holdout, pooled over fixes",
};
const ORDER = ["before_after_billed", "before_after_model", "did", "did_rtm", "ab_holdout"];

/** Each design's claim about the average fix, with its 95% interval, against the truth. */
export function EffectForest({ lift }: { lift: Report["lift"] }) {
  const rows = ORDER.flatMap((k) => (lift[k] ? [{ key: k, ...lift[k] }] : []));
  const truth = lift.ab_holdout?.true_mean ?? 0;
  const lo = Math.floor(Math.min(0, ...rows.map((r) => r.pooled_low)) * 100) / 100;
  const hi = Math.ceil(Math.max(...rows.map((r) => r.pooled_high)) * 100) / 100;
  const at = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const ticks = Array.from({ length: Math.round((hi - lo) * 100) + 1 }, (_, i) => lo + i / 100);
  return (
    <figure className={styles.forest}>
      <div className={styles.plot}>
        <div className={styles.row} aria-hidden="true">
          <span />
          <div className={styles.axis}>
            <span className={styles.truthLabel} style={{ left: at(truth) }}>
              true average {pts(truth)}
            </span>
          </div>
          <span />
        </div>
        {rows.map((r) => {
          const off = Math.abs(r.pooled - truth) > 0.01;
          const tone = r.key === "ab_holdout" ? "measured" : off ? "claim" : "muted";
          return (
            <div key={r.key} className={styles.row} data-tone={tone}>
              <span className={styles.label}>{DESIGN_NAMES[r.key]}</span>
              <div className={styles.track} title={`${pts(r.pooled)} (${pts(r.pooled_low)} to ${pts(r.pooled_high)})`}>
                <span className={styles.truthTick} style={{ left: at(truth) }} />
                <span className={styles.ci} style={{ left: at(r.pooled_low), right: `calc(100% - ${at(r.pooled_high)})` }} />
                <span className={styles.dot} style={{ left: at(r.pooled) }} />
                <span className="visually-hidden">
                  claims {pts(r.pooled)}, interval {pts(r.pooled_low)} to {pts(r.pooled_high)}
                </span>
              </div>
              <span className={styles.value}>{pts(r.pooled)}</span>
            </div>
          );
        })}
        <div className={styles.row} aria-hidden="true">
          <span />
          <div className={styles.axis}>
            {ticks.map((t) => (
              <span key={t} style={{ left: at(t) }}>
                {Math.round(t * 100)}
              </span>
            ))}
          </div>
          <span />
        </div>
      </div>
      <figcaption className={styles.caption}>Average effect of one content fix on the resolution rate, percentage points, with its 95% interval</figcaption>
    </figure>
  );
}
