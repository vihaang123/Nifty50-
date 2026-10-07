"use client";
import { Fragment, useState } from "react";
import { TableWrap, tableClasses as t } from "@/components/common/ui";
import type { RebalanceInfo, RebalanceRow } from "@/lib/types";
import { formatDate, formatPercent, formatRupees } from "@/lib/utils";

const STATUS_TEXT: Record<string, string> = {
  rebalanced: "Rebalanced",
  skipped_kept_previous_basket: "Skipped, kept previous basket",
  skipped_no_basket: "Skipped, no basket yet",
};

/** One row per rebalance, straight from the API. Open a row to see the stocks bought that day. */
export function RebalanceTable({ rebalances, history }: { rebalances: RebalanceInfo[]; history: RebalanceRow[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <TableWrap>
      <table className={t.table}>
        <thead>
          <tr>
            <th className={t.th}>Date</th>
            <th className={t.th}>Status</th>
            <th className={t.th}>Basket</th>
            <th className={`${t.th} text-right`}>Portfolio value</th>
            <th className={`${t.th} text-right`}>Stocks</th>
            <th className={t.th}>Details</th>
          </tr>
        </thead>
        <tbody>
          {rebalances.map((r) => {
            const isOpen = open === r.rebalance_date;
            const rows = history.filter((h) => h.rebalance_date === r.rebalance_date);
            return (
              <Fragment key={r.rebalance_date}>
                <tr>
                  <td className={`${t.td} num whitespace-nowrap`}>{formatDate(r.rebalance_date)}</td>
                  <td className={t.td}>{STATUS_TEXT[r.status] ?? r.status}</td>
                  <td className={`${t.td} min-w-[14rem]`}>{r.symbols.length ? r.symbols.join(", ") : "none"}</td>
                  <td className={`${t.td} num text-right`}>{r.capital == null ? "n/a" : formatRupees(r.capital)}</td>
                  <td className={`${t.td} num text-right`}>{r.n_holdings}</td>
                  <td className={t.td}>
                    {rows.length > 0 ? (
                      <button type="button" className="text-accent underline underline-offset-2" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : r.rebalance_date)}>
                        {isOpen ? "Hide" : "Show"}
                      </button>
                    ) : (
                      <span className="text-muted">{r.reason ?? "n/a"}</span>
                    )}
                  </td>
                </tr>
                {isOpen && (
                  <tr>
                    <td colSpan={6} className="border-b border-line bg-paper px-3 py-3">
                      <table className="w-full min-w-[28rem] text-left text-sm">
                        <thead>
                          <tr className="text-ink-2">
                            <th className="py-1 pr-3 font-medium">Stock</th>
                            <th className="py-1 pr-3 font-medium">Cap</th>
                            <th className="py-1 pr-3 font-medium">Behaviour</th>
                            <th className="py-1 pr-3 text-right font-medium">Weight</th>
                            <th className="py-1 text-right font-medium">Allocation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((h) => (
                            <tr key={h.symbol}>
                              <td className="py-1 pr-3 font-medium">{h.symbol}</td>
                              <td className="py-1 pr-3">{h.cap_category}</td>
                              <td className="py-1 pr-3">{h.behavior_class}</td>
                              <td className="num py-1 pr-3 text-right">{formatPercent(h.weight)}</td>
                              <td className="num py-1 text-right">{formatRupees(h.allocation, true)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </TableWrap>
  );
}
