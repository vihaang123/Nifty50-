"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { SidebarStatus } from "./SidebarStatus";
import { StatusBadge } from "./StatusBadge";

export const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/pca", label: "PCA Analysis" },
  { href: "/lda", label: "LDA Analysis" },
  { href: "/similarity", label: "Stock Similarity" },
  { href: "/basket", label: "Basket Generator" },
  { href: "/backtest", label: "Backtest" },
];

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex flex-col gap-0.5 p-3">
      {NAV_ITEMS.map((item) => {
        const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn("rounded-md px-3 py-2 text-sm", active ? "bg-accent-soft font-medium text-accent" : "text-ink-2 hover:bg-paper hover:text-ink")}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

function Brand() {
  return (
    <Link href="/" className="block px-5 py-4">
      <span className="font-serif text-lg font-medium leading-tight text-ink">Multi-Cap Basket Research</span>
      <span className="block text-sm text-muted">PCA and LDA study</span>
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen lg:flex">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <Brand />
        <div className="flex-1">
          <Nav />
        </div>
        <SidebarStatus />
      </aside>

      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-line bg-surface px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <button
              type="button"
              className="rounded-md border border-line-strong px-3 py-1.5 text-sm lg:hidden"
              aria-expanded={open}
              aria-controls="mobile-nav"
              onClick={() => setOpen((v) => !v)}
            >
              {open ? "Close" : "Menu"}
            </button>
            <span className="font-serif text-lg font-medium">Financial Market Intelligence</span>
          </div>
          <StatusBadge />
        </header>

        {open && (
          <div id="mobile-nav" className="border-b border-line bg-surface lg:hidden">
            <Nav onNavigate={() => setOpen(false)} />
            <SidebarStatus />
          </div>
        )}

        <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 sm:py-8">{children}</main>
        <footer className="mx-auto max-w-6xl px-4 pb-8 text-sm text-muted sm:px-6">
          Research and educational system. Nothing here is financial advice, and no result predicts future returns.
        </footer>
      </div>
    </div>
  );
}
