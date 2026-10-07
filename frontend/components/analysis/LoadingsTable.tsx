import { TableWrap, tableClasses as t } from "@/components/common/ui";
import type { LoadingRow } from "@/lib/types";

/** Feature loadings: how much each engineered feature contributes to each component. Scrolls sideways on small screens. */
export function LoadingsTable({ rows }: { rows: LoadingRow[] }) {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0].loadings);
  return (
    <TableWrap>
      <table className={t.table}>
        <thead>
          <tr>
            <th className={t.th}>Feature</th>
            {cols.map((c) => (
              <th key={c} className={`${t.th} text-right`}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.feature}>
              <td className={`${t.td} font-medium`}>{r.feature}</td>
              {cols.map((c) => (
                <td key={c} className={`${t.td} num text-right`}>
                  {r.loadings[c].toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </TableWrap>
  );
}
