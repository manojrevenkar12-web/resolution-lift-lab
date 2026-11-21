import { type Report, pct, pts } from "../../shared/data/report";
import styles from "../../shared/ui/table.module.css";

const NAMES: Record<string, string> = {
  billing_proxy: "Billed rate",
  return_adjusted: "Billed, minus customers who came back within 7 days",
  audit_only: "Weekly audit alone (≈70 labels)",
  model_only: "Classifier alone",
  ppi: "Prediction-powered inference",
};
const ORDER = ["billing_proxy", "return_adjusted", "audit_only", "model_only", "ppi"];

export function EstimatorTable({ rows }: { rows: Report["resolution"] }) {
  return (
    <table className={styles.table}>
      <caption className="visually-hidden">Error of each estimator of the true resolution rate</caption>
      <thead>
        <tr>
          <th scope="col">Estimator, per workspace and 4 weeks</th>
          <th scope="col" className={styles.n}>Mean error</th>
          <th scope="col" className={styles.n}>Bias</th>
          <th scope="col" className={styles.n}>95% interval held</th>
          <th scope="col" className={styles.n}>Interval width</th>
        </tr>
      </thead>
      <tbody>
        {ORDER.map((k) => {
          const r = rows[k];
          if (!r) return null;
          return (
            <tr key={k} data-pick={k === "ppi" || undefined} data-status-quo={k === "billing_proxy" || undefined}>
              <th scope="row">{NAMES[k]}</th>
              <td className={styles.n}>{pts(r.mae).replace("+", "")}</td>
              <td className={styles.n}>{pts(r.bias)}</td>
              <td className={styles.n}>{r.coverage == null ? "no interval" : pct(r.coverage, 0)}</td>
              <td className={styles.n}>{r.width == null ? "–" : pts(r.width).replace("+", "")}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
