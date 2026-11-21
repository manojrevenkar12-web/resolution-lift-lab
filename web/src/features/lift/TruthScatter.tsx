import { type Report } from "../../shared/data/report";
import { DESIGN_NAMES } from "./EffectForest";
import styles from "./lift.module.css";

const S = 220;
const LIM = 0.3;

/** One dot per fix: what the design said against what the fix really did. On the diagonal is right. */
export function TruthScatter({ points }: { points: Report["lift_points"] }) {
  const panels = ["before_after_billed", "ab_holdout"];
  const p = (v: number) => ((Math.max(-LIM, Math.min(LIM, v)) + LIM) / (2 * LIM)) * S;
  return (
    <div className={styles.scatters}>
      {panels.map((design) => {
        const pts = points.filter((q) => q.design === design);
        return (
          <figure key={design} className={styles.scatter}>
            <svg viewBox={`-4 -4 ${S + 8} ${S + 8}`} role="img" aria-label={`${DESIGN_NAMES[design]}: ${pts.length} fixes, estimate against true effect`}>
              <rect x={0} y={0} width={S} height={S} className={styles.frame} />
              <line x1={0} y1={S} x2={S} y2={0} className={styles.diagonal} />
              <line x1={p(0)} x2={p(0)} y1={0} y2={S} className={styles.grid} />
              <line y1={S - p(0)} y2={S - p(0)} x1={0} x2={S} className={styles.grid} />
              {pts.map((q, i) => (
                <circle key={i} cx={p(q.truth)} cy={S - p(q.estimate)} r={2.4} data-tone={design === "ab_holdout" ? "measured" : "claim"} className={styles.point} />
              ))}
            </svg>
            <figcaption>
              <b>{DESIGN_NAMES[design]}</b>
              <span>across: true effect · up: estimate · ±30 pts</span>
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}
