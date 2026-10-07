import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface FieldShellProps {
  label: string;
  id: string;
  error?: string;
  hint?: ReactNode;
}

const control = "w-full rounded-md border bg-surface px-3 py-2 text-sm text-ink placeholder:text-muted";

export function TextField({ label, id, error, hint, ...props }: FieldShellProps & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm font-medium text-ink-2">
        {label}
      </label>
      <input id={id} aria-invalid={!!error} aria-describedby={error ? `${id}-error` : hint ? `${id}-hint` : undefined} {...props} className={cn(control, error ? "border-danger-ink" : "border-line-strong")} />
      {error ? (
        <p id={`${id}-error`} className="mt-1 text-sm text-danger-ink">
          {error}
        </p>
      ) : (
        hint && (
          <p id={`${id}-hint`} className="mt-1 text-sm text-muted">
            {hint}
          </p>
        )
      )}
    </div>
  );
}

export function SelectField({ label, id, error, hint, children, ...props }: FieldShellProps & SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm font-medium text-ink-2">
        {label}
      </label>
      <select id={id} aria-invalid={!!error} {...props} className={cn(control, error ? "border-danger-ink" : "border-line-strong")}>
        {children}
      </select>
      {error ? <p className="mt-1 text-sm text-danger-ink">{error}</p> : hint && <p className="mt-1 text-sm text-muted">{hint}</p>}
    </div>
  );
}
