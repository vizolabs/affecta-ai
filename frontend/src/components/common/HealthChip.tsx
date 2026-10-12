"use client";

import { useEffect, useState } from "react";
import { apiFetch, API_URL } from "../../lib/api";

interface Health {
  status?: string;
  version?: string;
  model?: string;
  model_loaded?: boolean;
}

const POLL_MS = 30_000;

/** Live local-backend health indicator for the header. */
export default function HealthChip() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const h = await apiFetch<Health>("/api/health");
        if (!cancelled) {
          setHealth(h);
          setDown(false);
        }
      } catch {
        if (!cancelled) setDown(true);
      }
    };
    void ping();
    const id = setInterval(ping, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const healthy = !down && health?.status === "ok";
  const label = health === null && !down
    ? "Checking…"
    : down
      ? "API offline"
      : health?.model_loaded
        ? `Model ${health.model ?? "?"} ready`
        : "API ok";

  return (
    <span
      title={
        healthy
          ? `Local backend ${health?.version ? `v${health.version} ` : ""}· model ${
              health?.model ?? "unknown"
            }${health?.model_loaded ? " (loaded)" : " (idle)"} · polls every 30s`
          : `Cannot reach the local API at ${API_URL}`
      }
      className={`hidden items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] sm:inline-flex ${
        down
          ? "border-red-500/40 bg-red-500/10 text-red-400"
          : healthy && health?.model_loaded
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
            : "border-border bg-background text-muted"
      }`}
    >
      <span
        aria-hidden
        className={`h-1.5 w-1.5 rounded-full ${
          down ? "bg-red-400" : healthy ? "bg-emerald-400" : "bg-amber-400"
        }`}
      />
      {label}
    </span>
  );
}
