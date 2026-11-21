import { type Report, pct } from "../../shared/data/report";
import styles from "../../shared/ui/table.module.css";

const NAMES: Record<string, string> = {
  keyword_rules: "Keyword rules kept by hand",
  by_citation: "One topic per article the agent cited",
  word_kmeans: "Word TF-IDF, k-means",
  char_kmeans: "Character n-grams, k-means",
  char_kmeans_cited: "Character n-grams, merged where citations agree",
};
const ORDER = ["keyword_rules", "by_citation", "word_kmeans", "char_kmeans", "char_kmeans_cited"];

export function TopicsTable({ topics }: { topics: Report["topics"] }) {
  return (
    <div className={styles.wrap}>
      <table className={styles.table}>
        <caption className="visually-hidden">Topic discovery and content gaps by method</caption>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col" className={styles.n}>Agreement with true topics (ARI)</th>
            <th scope="col" className={styles.n}>English</th>
            <th scope="col" className={styles.n}>German</th>
            <th scope="col" className={styles.n}>Assigned</th>
            <th scope="col" className={styles.n}>Content gaps found</th>
            <th scope="col" className={styles.n}>False alarms</th>
          </tr>
        </thead>
        <tbody>
          {ORDER.map((k) => {
            const t = topics[k];
            if (!t) return null;
            return (
              <tr key={k} data-pick={k === "char_kmeans_cited" || undefined} data-status-quo={k === "keyword_rules" || undefined}>
                <th scope="row">{NAMES[k]}</th>
                <td className={styles.n}>{t.ari.toFixed(2)}</td>
                <td className={styles.n}>{t.ari_en.toFixed(2)}</td>
                <td className={styles.n}>{t.ari_de.toFixed(2)}</td>
                <td className={styles.n}>{pct(t.assigned, 0)}</td>
                <td className={styles.n}>
                  {t.gaps_found} of {t.workspaces}
                </td>
                <td className={styles.n}>{t.false_gap_flags}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
