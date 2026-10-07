"use client";
import { useCallback, useEffect, useRef, useState } from "react";

/** Runs a GET-style request on mount (and whenever `key` changes). `reload` asks again after a failure. */
export function useRequest<T>(fetcher: () => Promise<T>, key: string) {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{ id: string; data?: T; error?: unknown } | null>(null);
  const id = `${key}#${attempt}`;

  useEffect(() => {
    let cancelled = false;
    fetcher().then(
      (data) => !cancelled && setResult({ id, data }),
      (error) => !cancelled && setResult({ id, error }),
    );
    return () => {
      cancelled = true; // a late answer for an old key is ignored
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const current = result && result.id === id ? result : null;
  return {
    data: current?.error === undefined ? current?.data : undefined,
    error: current?.error,
    loading: current === null,
    reload: () => setAttempt((n) => n + 1),
  };
}

export type ActionState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: unknown };

/** For button-triggered requests (basket, backtest): only runs when `run` is called. Supports cancel and an elapsed-seconds counter. */
export function useAction<P, T>(action: (payload: P, signal: AbortSignal) => Promise<T>) {
  const [state, setState] = useState<ActionState<T>>({ status: "idle" });
  const [elapsed, setElapsed] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const runId = useRef(0);

  const loading = state.status === "loading";
  useEffect(() => {
    if (!loading) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [loading]);

  const run = useCallback(
    async (payload: P) => {
      controller.current?.abort();
      const ctl = new AbortController();
      controller.current = ctl;
      const myRun = ++runId.current;
      setElapsed(0);
      setState({ status: "loading" });
      try {
        const data = await action(payload, ctl.signal);
        if (myRun === runId.current) setState({ status: "success", data });
      } catch (error) {
        if (myRun !== runId.current) return;
        if (error instanceof DOMException && error.name === "AbortError") setState({ status: "idle" });
        else setState({ status: "error", error });
      }
    },
    [action],
  );

  const cancel = useCallback(() => controller.current?.abort(), []);
  useEffect(() => () => controller.current?.abort(), []);
  return { state, run, cancel, elapsed };
}
