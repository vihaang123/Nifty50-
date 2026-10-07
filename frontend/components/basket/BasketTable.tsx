import { TableWrap, tableClasses as t, Swatch } from "@/components/common/ui";
import { behaviorColor } from "@/lib/colors";
import type { BasketStock } from "@/lib/types";
import { formatPercent, formatRupees } from "@/lib/utils";

export function BasketTable({ stocks, showReason = true }: { stocks: BasketStock[]; showReason?: boolean }) {
  return (
    <TableWrap>
      <table className={t.table}>
        <thead>
          <tr>
            <th className={t.th}>Stock</th>
            <th className={t.th}>Cap Category</th>
            <th className={t.th}>Behaviour</th>
            <th className={`${t.th} text-right`}>Weight</th>
            <th className={`${t.th} text-right`}>Allocation</th>
            {showReason && <th className={`${t.th} min-w-[18rem]`}>Reason</th>}
          </tr>
        </thead>
        <tbody>
          {stocks.map((s) => (
            <tr key={s.symbol}>
              <td className={`${t.td} font-medium`}>{s.symbol}</td>
              <td className={t.td}>{s.cap_category}</td>
              <td className={t.td}>
                <span className="inline-flex items-center gap-1.5">
                  <Swatch color={behaviorColor(s.behavior_class)} />
                  {s.behavior_class}
                </span>
              </td>
              <td className={`${t.td} num text-right`}>{formatPercent(s.weight)}</td>
              <td className={`${t.td} num text-right`}>{formatRupees(s.allocation, true)}</td>
              {showReason && <td className={`${t.td} text-ink-2`}>{s.reason}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </TableWrap>
  );
}
