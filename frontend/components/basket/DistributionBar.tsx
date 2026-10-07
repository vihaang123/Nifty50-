import { Legend } from "@/components/common/ui";
import type { Distribution } from "@/lib/types";
import { formatPercentPoints } from "@/lib/utils";

/** A single segmented bar. Segments are labelled in a legend underneath, so colour is never the only cue. */
export function DistributionBar({ title, distribution, colorOf, order }: { title: string; distribution: Record<string, Distribution>; colorOf: (name: string) => string; order: string[] }) {
  const names = [...order.filter((n) => n in distribution), ...Object.keys(distribution).filter((n) => !order.includes(n))];
  const shown = names.filter((n) => distribution[n].count > 0);
  return (
    <div>
      <h3 className="mb-2 text-sm font-medium text-ink-2">{title}</h3>
      <div className="flex h-3 w-full gap-0.5 overflow-hidden rounded" role="img" aria-label={`${title}: ${shown.map((n) => `${n} ${distribution[n].count}`).join(", ")}`}>
        {shown.map((n) => (
          <div key={n} style={{ width: `${distribution[n].percentage}%`, background: colorOf(n) }} />
        ))}
      </div>
      <div className="mt-2">
        <Legend items={names.map((n) => ({ label: `${n}: ${distribution[n].count} (${formatPercentPoints(distribution[n].percentage)})`, color: colorOf(n) }))} />
      </div>
    </div>
  );
}
