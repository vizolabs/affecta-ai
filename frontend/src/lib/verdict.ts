"use client";

/**
 * Verdict window — Periodic ("check-in") mode aggregation.
 *
 * Frames stream continuously at the configured analysis rate. Every interval
 * (or an early flush: motion, re-acquire) the samples collected since the last
 * flush are mean-pooled into one pinned verdict: the dominant emotion, its
 * pooled confidence, the mean face box, and the mean region evidence. A pooled
 * top-1 below the uncertain floor pins "uncertain" instead of a bare label.
 */

export const VERDICT_UNCERTAIN_BELOW = 0.4;
/** Minimum window size before an instant (re-acquire) flush is allowed. */
export const VERDICT_INSTANT_MIN = 3;
/** No face sample for this long ⇒ the pinned verdict is marked stale. */
export const VERDICT_STALE_MS = 2500;

export type Box = [number, number, number, number];

export interface VerdictSample {
  faceId: number;
  /** Epoch ms when the sample was analysed — drives recency weighting. */
  ts: number;
  box: Box;
  probabilities: Record<string, number>;
  regions: Record<string, number> | null;
}

export interface Verdict {
  /** Dominant emotion key, or "uncertain". */
  label: string;
  /** Weighted probability of `label` (0..1). */
  confidence: number;
  uncertain: boolean;
  /** Second-highest pooled emotion (null when the window has one class). */
  runnerUp: { label: string; confidence: number } | null;
  /** Recency-weighted mean-pooled probabilities over the window. */
  probabilities: Record<string, number>;
  /** Recency-weighted region percentages (null when the window had none). */
  regions: Record<string, number> | null;
  /** Recency-weighted mean face box over the window. */
  box: Box;
  sampleCount: number;
  flushedAt: number;
}

export class VerdictBuffer {
  private samples: VerdictSample[] = [];
  private _faceId: number | null = null;

  get size(): number {
    return this.samples.length;
  }

  get faceId(): number | null {
    return this._faceId;
  }

  push(sample: VerdictSample): void {
    this.samples.push(sample);
    this._faceId = sample.faceId;
  }

  /**
   * Drain the window into a pooled verdict (null when empty).
   *
   * Samples are pooled with exponential recency weighting: a sample's weight
   * is `exp(-age / τ)` with `τ = max(2 s, span / 3)`, so recent frames steer
   * the verdict while older ones still damp flicker. Short (early-flush)
   * windows degrade gracefully to near-uniform weights.
   */
  flush(nowMs: number): Verdict | null {
    const n = this.samples.length;
    if (n === 0) return null;

    let minTs = Infinity;
    let maxTs = -Infinity;
    for (const s of this.samples) {
      if (s.ts < minTs) minTs = s.ts;
      if (s.ts > maxTs) maxTs = s.ts;
    }
    const tau = Math.max(2000, (maxTs - minTs) / 3);

    const probs: Record<string, number> = {};
    const regAcc: Record<string, number> = {};
    let regW = 0;
    const box: Box = [0, 0, 0, 0];
    let totalW = 0;

    for (const s of this.samples) {
      const w = Math.exp(-(nowMs - s.ts) / tau);
      totalW += w;
      for (const [k, v] of Object.entries(s.probabilities)) {
        probs[k] = (probs[k] ?? 0) + v * w;
      }
      if (s.regions) {
        regW += w;
        for (const [k, v] of Object.entries(s.regions)) {
          regAcc[k] = (regAcc[k] ?? 0) + v * w;
        }
      }
      box[0] += s.box[0] * w;
      box[1] += s.box[1] * w;
      box[2] += s.box[2] * w;
      box[3] += s.box[3] * w;
    }

    if (totalW <= 0) {
      this.samples = [];
      return null;
    }
    for (const k of Object.keys(probs)) probs[k] /= totalW;

    const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    let top = ranked[0]?.[0] ?? "neutral";
    let topP = ranked[0]?.[1] ?? 0;
    if (!Number.isFinite(topP) || topP < 0) {
      top = "neutral";
      topP = 0;
    }
    const runnerUp =
      ranked.length > 1 && ranked[1][1] > 0
        ? { label: ranked[1][0], confidence: ranked[1][1] }
        : null;

    let regions: Record<string, number> | null = null;
    if (regW > 0) {
      regions = {};
      for (const [k, v] of Object.entries(regAcc)) regions[k] = v / regW;
    }

    const uncertain = topP < VERDICT_UNCERTAIN_BELOW;
    const verdict: Verdict = {
      label: uncertain ? "uncertain" : top,
      confidence: topP,
      uncertain,
      runnerUp,
      probabilities: probs,
      regions,
      box: [box[0] / totalW, box[1] / totalW, box[2] / totalW, box[3] / totalW],
      sampleCount: n,
      flushedAt: nowMs,
    };

    this.samples = [];
    return verdict;
  }

  reset(): void {
    this.samples = [];
    this._faceId = null;
  }
}

/**
 * Mirror of backend REGION_BOXES (src/api/services/inference.py): fractional
 * (y0, y1, x0, x1) sub-boxes inside the face crop, used to place the verdict
 * region glow on the overlay canvas.
 */
export const REGION_FRAC: Record<
  string,
  [number, number, number, number][]
> = {
  eyebrows: [[0.1, 0.28, 0.15, 0.85]],
  eyes: [[0.25, 0.45, 0.15, 0.85]],
  nose: [[0.4, 0.65, 0.35, 0.65]],
  mouth: [[0.65, 0.95, 0.2, 0.8]],
  cheeks: [
    [0.45, 0.75, 0.05, 0.3],
    [0.45, 0.75, 0.7, 0.95],
  ],
};

/** Key of the highest-percentage region (null when absent or all zero). */
export function dominantRegion(
  regions: Record<string, number> | null,
): string | null {
  if (!regions) return null;
  let top: string | null = null;
  let topV = -Infinity;
  for (const [k, v] of Object.entries(regions)) {
    if (typeof v === "number" && v > topV) {
      topV = v;
      top = k;
    }
  }
  return top !== null && topV > 0 ? top : null;
}
