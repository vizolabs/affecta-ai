"use client";

/**
 * Analysis cadence scheduler.
 *
 * Two modes share one timer chain so the UI can always show an honest
 * "next check in …" countdown:
 *
 *  - live     — fire every `1000 / fps` ms (continuous capture)
 *  - periodic — fire every `intervalMs` ms (verdict flush; frames still
 *               stream at the live rate, the preview keeps rendering
 *               locally, independent of this timer)
 *
 * The chain self-reschedules against the *previous due time* rather than the
 * completion time, so long intervals do not accumulate drift.
 */

export type AnalysisMode = "live" | "periodic";

export interface CadenceConfig {
  mode: AnalysisMode;
  /** Frames per second — live mode only. */
  fps: number;
  /** Milliseconds between analyses — periodic mode only. */
  intervalMs: number;
}

export const DEFAULT_FPS = 8;
export const MIN_INTERVAL_MS = 30_000; // 30s
export const MAX_INTERVAL_MS = 600_000; // 10 min

export function delayFor(cfg: CadenceConfig): number {
  if (cfg.mode === "periodic") {
    return Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, cfg.intervalMs));
  }
  const fps = Math.min(15, Math.max(1, Math.round(cfg.fps) || DEFAULT_FPS));
  return Math.round(1000 / fps);
}

export interface CadenceHandle {
  /** Stop the chain and clear any pending timer. */
  stop(): void;
  /** Epoch ms of the next scheduled analysis (null when stopped). */
  nextDueAt(): number | null;
  /** Seconds until the next analysis (null when stopped). */
  secondsUntil(nowMs: number): number | null;
}

interface TimerApi {
  setTimeout: (fn: () => void, ms: number) => number;
  clearTimeout: (id: number) => void;
  now: () => number;
}

const defaultTimer: TimerApi = {
  setTimeout: (fn, ms) => window.setTimeout(fn, ms),
  clearTimeout: (id) => window.clearTimeout(id),
  now: () => Date.now(),
};

export function startCadence(
  cfg: CadenceConfig,
  onTick: () => void,
  timer: TimerApi = defaultTimer,
): CadenceHandle {
  let id: number | null = null;
  let due: number | null = null;
  let stopped = false;

  const schedule = () => {
    if (stopped) return;
    const delay = delayFor(cfg);
    const now = timer.now();
    // Keep the phase anchored to the previous due time (no cumulative drift),
    // but never schedule into the past after a long stall.
    due = Math.max(now, due ?? now) + delay;
    id = timer.setTimeout(fire, Math.max(0, due - now));
  };

  const fire = () => {
    if (stopped) return;
    id = null;
    due = timer.now();
    onTick();
    if (!stopped) schedule();
  };

  schedule();

  return {
    stop() {
      stopped = true;
      if (id !== null) timer.clearTimeout(id);
      id = null;
      due = null;
    },
    nextDueAt() {
      return due;
    },
    secondsUntil(nowMs: number) {
      if (due === null) return null;
      return Math.max(0, (due - nowMs) / 1000);
    },
  };
}

/* ---- formatting helpers (shared by live HUD + settings) ---- */

/** `95` -> `1:35`, `120` -> `2:00`. */
export function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const m = Math.floor(s / 60);
  const rest = s % 60;
  return `${m}:${rest.toString().padStart(2, "0")}`;
}

/** Human phrasing for the hold control: seconds below a minute, else minutes. */
export function formatHold(seconds: number): string {
  if (seconds === 0) return "off";
  if (seconds < 60) return `${Number(seconds.toFixed(1))} s`;
  const m = seconds / 60;
  return `${Number(m.toFixed(m % 1 === 0 ? 0 : 1))} min`;
}