import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { cloneElement, type ReactElement } from "react";
import { afterEach, beforeEach, vi } from "vitest";
import { clearApiCache } from "@/lib/api";

// jsdom has no layout, so ResponsiveContainer would render a 0x0 chart. Give charts a fixed size instead.
vi.mock("recharts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("recharts")>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactElement<{ width?: number; height?: number }> }) => (
      <div style={{ width: 800, height: 400 }}>{cloneElement(children, { width: 800, height: 400 })}</div>
    ),
  };
});

vi.mock("next/navigation", () => ({ usePathname: () => "/dashboard", useRouter: () => ({ push: vi.fn() }) }));

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_API_URL", "http://api.test");
  clearApiCache();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});
