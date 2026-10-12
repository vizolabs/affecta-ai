export const EMOTIONS = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export const EMOTION_COLORS: Record<string, string> = {
  angry: "#ef4444",
  disgust: "#ec4899", // pink — CVD-safe vs old lime (collided with happy)
  fear: "#a855f7",
  happy: "#22c55e",
  neutral: "#94a3b8",
  sad: "#3b82f6",
  surprise: "#f59e0b",
};

export const EMOTION_EMOJI: Record<string, string> = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprise: "😮",
};

export const EMOTION_LABEL: Record<string, string> = {
  angry: "Angry",
  disgust: "Disgust",
  fear: "Fear",
  happy: "Happy",
  neutral: "Neutral",
  sad: "Sad",
  surprise: "Surprise",
};

export function emotionColor(emotion: string): string {
  return EMOTION_COLORS[emotion] ?? "#94a3b8";
}

export function personLabel(faceId: number): string {
  return `Person ${String(faceId + 1).padStart(2, "0")}`;
}

export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * Dual-emotion display rule (all modes).
 *
 * The top emotion is always shown. When a runner-up is nearly tied with it
 * (within DUAL_WITHIN of the top probability, and at least DUAL_MIN on its
 * own), both are shown as "JOY / SURPRISE" — a genuine 50/50 or 40/40 mix
 * reads better as two majors than as one arbitrarily chosen label.
 */
export const DUAL_WITHIN = 0.1;
export const DUAL_MIN = 0.15;

export interface EmotionShare {
  label: string;
  confidence: number;
}

/** Ranked emotion shares (highest first) from a probability map. */
export function rankedShares(
  probabilities: Record<string, number> | null | undefined,
): EmotionShare[] {
  if (!probabilities) return [];
  return Object.entries(probabilities)
    .filter(([, v]) => typeof v === "number" && v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([label, confidence]) => ({ label, confidence }));
}

/**
 * Primary emotion + optional near-tied secondary for display.
 * Returns `dual` when the runner-up is within DUAL_WITHIN of the top and
 * clears DUAL_MIN on its own (e.g. 50/50, 45/42, 40/38 — but not 60/45).
 */
export function primaryWithDual(
  probabilities: Record<string, number> | null | undefined,
  topOverride?: EmotionShare | null,
): { primary: EmotionShare | null; dual: EmotionShare | null } {
  const ranked = rankedShares(probabilities);
  const primary =
    topOverride ??
    (ranked.length > 0
      ? ranked[0]
      : null);
  if (!primary) return { primary: null, dual: null };
  // Prefer the explicit runner-up (verdict windows already computed it),
  // else derive from the probability map.
  const second: EmotionShare | null =
    ranked.length > 1 && ranked[1].label !== primary.label
      ? ranked[1]
      : null;
  const dual =
    second &&
    second.confidence >= DUAL_MIN &&
    primary.confidence - second.confidence <= DUAL_WITHIN
      ? second
      : null;
  return { primary, dual };
}

/** Display string: "JOY / SURPRISE" or "JOY". */
export function dualLabel(primary: EmotionShare, dual: EmotionShare | null): string {
  const p = (EMOTION_LABEL[primary.label] ?? primary.label).toUpperCase();
  if (!dual) return p;
  const d = (EMOTION_LABEL[dual.label] ?? dual.label).toUpperCase();
  return `${p} / ${d}`;
}
