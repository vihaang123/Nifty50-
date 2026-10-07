import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function PageHeader({ title, subtitle }: { title: string; subtitle?: ReactNode }) {
  return (
    <div className="mb-6 max-w-3xl">
      <h1 className="font-serif text-3xl font-medium tracking-tight text-ink sm:text-4xl">{title}</h1>
      {subtitle && <p className="mt-2 text-ink-2">{subtitle}</p>}
    </div>
  );
}

export function Panel({ title, description, children, className, action }: { title?: string; description?: ReactNode; children: ReactNode; className?: string; action?: ReactNode }) {
  return (
    <section className={cn("rounded-md border border-line bg-surface p-4 sm:p-5", className)}>
      {(title || action) && (
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="font-serif text-xl font-medium text-ink">{title}</h2>}
            {description && <p className="mt-1 text-sm text-muted">{description}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatCard({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "good" | "bad" }) {
  return (
    <div className="rounded-md border border-line bg-surface p-4">
      <div className="text-sm text-muted">{label}</div>
      <div className={cn("num mt-1 font-serif text-2xl font-medium sm:text-3xl", tone === "bad" ? "text-danger-ink" : tone === "good" ? "text-good-ink" : "text-ink")}>{value}</div>
      {hint && <div className="mt-1 text-sm text-ink-2">{hint}</div>}
    </div>
  );
}

const NOTICE_STYLES = {
  info: "border-line bg-accent-soft text-ink",
  warn: "border-amber-ink/30 bg-amber-soft text-amber-ink",
  danger: "border-danger-ink/30 bg-danger-soft text-danger-ink",
} as const;

export function Notice({ tone = "info", title, children }: { tone?: keyof typeof NOTICE_STYLES; title?: string; children: ReactNode }) {
  return (
    <div role={tone === "danger" ? "alert" : "note"} className={cn("rounded-md border px-4 py-3 text-sm", NOTICE_STYLES[tone])}>
      {title && <div className="mb-0.5 font-medium">{title}</div>}
      <div>{children}</div>
    </div>
  );
}

/** Wraps a table so it scrolls sideways on narrow screens instead of breaking the page. */
export function TableWrap({ children }: { children: ReactNode }) {
  return <div className="-mx-1 overflow-x-auto px-1">{children}</div>;
}
export const tableClasses = {
  table: "w-full min-w-[32rem] border-collapse text-left text-sm",
  th: "whitespace-nowrap border-b border-line-strong px-3 py-2 font-medium text-ink-2",
  td: "border-b border-line px-3 py-2 align-top",
};

export function Button({ variant = "primary", className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" }) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
        variant === "primary" ? "bg-accent text-white hover:bg-[#173a57]" : "border border-line-strong bg-surface text-ink hover:bg-paper",
        className,
      )}
    />
  );
}

export function Swatch({ color }: { color: string }) {
  return <span aria-hidden className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: color }} />;
}

export function Legend({ items }: { items: Array<{ label: string; color: string }> }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink-2">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <Swatch color={item.color} />
          {item.label}
        </li>
      ))}
    </ul>
  );
}
