import { motion as m } from "framer-motion";

import { type Report, num, pct } from "../../shared/data/report";
import { motion } from "../../shared/theme/motion";
import styles from "./OutcomeBar.module.css";

/**
 * Every conversation the agent answered, by what really happened. Billed = the first three
 * segments; actually solved = the first two. The gap between the brackets is the finding.
 */
export function OutcomeBar({ facts }: { facts: Report["facts"] }) {
  const total = facts.confirmed + facts.assumed_resolved + facts.assumed_unresolved + facts.escalated;
  const segs = [
    { key: "confirmed", label: "Confirmed solved", n: facts.confirmed, tone: "solved" },
    { key: "assumed_resolved", label: "Went quiet, was solved", n: facts.assumed_resolved, tone: "solved-quiet" },
    { key: "assumed_unresolved", label: "Went quiet, had given up", n: facts.assumed_unresolved, tone: "gave-up" },
    { key: "escalated", label: "Asked for more help", n: facts.escalated, tone: "asked" },
  ];
  const share = (n: number) => n / total;
  const billed = share(facts.confirmed + facts.assumed_resolved + facts.assumed_unresolved);
  const solved = share(facts.confirmed + facts.assumed_resolved);
  return (
    <figure className={styles.figure} aria-label="What happened after the agent's last answer">
      <div className={styles.bracket} style={{ width: `${billed * 100}%` }} data-kind="billed">
        <span>Billed as resolved {pct(billed)}</span>
      </div>
      <m.div className={styles.bar} initial="hidden" animate="shown" variants={{ shown: { transition: { staggerChildren: motion.stagger } } }}>
        {segs.map((s) => (
          <m.div
            key={s.key}
            className={styles.seg}
            data-tone={s.tone}
            style={{ width: `${share(s.n) * 100}%` }}
            variants={{ hidden: { scaleX: 0 }, shown: { scaleX: 1 } }}
            transition={{ duration: motion.base, ease: motion.ease }}
            title={`${s.label}: ${num(s.n)} (${pct(share(s.n))})`}
          />
        ))}
      </m.div>
      <div className={styles.bracket} style={{ width: `${solved * 100}%` }} data-kind="solved">
        <span>Actually solved {pct(solved)}</span>
      </div>
      <figcaption className={styles.legend}>
        {segs.map((s) => (
          <span key={s.key} className={styles.key} data-tone={s.tone}>
            {s.label} <b>{pct(share(s.n))}</b>
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
