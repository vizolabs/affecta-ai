import type { SessionOut } from "./types";

/** Human session name: explicit title → dominant emotion → date/time. */
export function sessionLabel(s: SessionOut): string {
  const meta = (s.metadata ?? {}) as Record<string, unknown>;
  const title = typeof meta.title === "string" && meta.title ? meta.title : null;
  if (title) return title;
  const dist = meta.expression_distribution as
    | Record<string, number>
    | undefined;
  if (dist) {
    const top = Object.entries(dist).sort((a, b) => b[1] - a[1])[0];
    if (top) return `${top[0]} session`;
  }
  const d = new Date(s.start_time);
  return `Session ${d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  })} ${d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

export function formatDuration(seconds?: number | null): string {
  if (seconds == null) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

export function slug(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}
