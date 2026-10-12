"use client";

/**
 * Cheap client-side motion estimation for periodic (check-in) mode.
 *
 * We deliberately avoid running the classifier to "notice" a change: a 64x48
 * grayscale downscale + mean absolute difference costs microseconds, while a
 * model inference costs tens of milliseconds. Motion only ever *accelerates*
 * the next scheduled analysis — it never becomes the analysis itself.
 */

export const MOTION_W = 64;
export const MOTION_H = 48;

/** Mean absolute difference of two equal-sized Uint8ClampedArray samples. */
export function computeMotionScore(
  prev: Uint8ClampedArray | null,
  curr: Uint8ClampedArray,
): number {
  if (!prev || prev.length !== curr.length || curr.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < curr.length; i++) sum += Math.abs(curr[i] - prev[i]);
  return sum / curr.length / 255; // normalised 0..1
}

export interface MotionGate {
  /** Newest sample, or null before the first frame. */
  last: Uint8ClampedArray | null;
  /** Epoch ms of the last motion-triggered send (cooldown). */
  lastTriggeredAt: number;
}

export function createMotionGate(): MotionGate {
  return { last: null, lastTriggeredAt: 0 };
}

/**
 * Should this sample force an analysis now?
 *
 * @param score      normalised motion score (see computeMotionScore)
 * @param threshold  minimum score to count as movement (0..1)
 * @param nowMs      current epoch ms
 * @param cooldownMs minimum gap between motion-triggered sends
 */
export function shouldTriggerMotion(
  score: number,
  threshold: number,
  nowMs: number,
  gate: MotionGate,
  cooldownMs: number,
): boolean {
  if (score < threshold) return false;
  if (nowMs - gate.lastTriggeredAt < cooldownMs) return false;
  gate.lastTriggeredAt = nowMs;
  return true;
}

/** Offsets sampled per second while watching for motion (cheap, no inference). */
export const MOTION_POLL_HZ = 5;

/** Default sensitivity: mean pixel delta ~7.5% of full range. */
export const DEFAULT_MOTION_THRESHOLD = 0.075;

/** Cooldown between motion-triggered analyses (ms). */
export const DEFAULT_MOTION_COOLDOWN_MS = 5000;