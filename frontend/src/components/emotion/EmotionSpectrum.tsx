"use client";

import { EMOTION_LABEL, emotionColor, pct } from "../../lib/emotions";

interface Props {
  probabilities: Record<string, number>;
  ranked?: string[];
  evidenceScores?: Record<string, number>;
}

/** Ranked emotion bar list (probability + optional ranking-evidence column). */
export default function EmotionSpectrum({
  probabilities,
  ranked,
  evidenceScores,
}: Props) {
  const order =
    ranked ??
    Object.entries(probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([e]) => e);

  return (
    <ul className="space-y-1.5">
      {order.map((emotion) => {
        const p = probabilities[emotion] ?? 0;
        const ev = evidenceScores?.[emotion];
        return (
          <li key={emotion} className="flex items-center gap-2 text-xs">
            <span className="w-16 shrink-0 text-muted">
              {EMOTION_LABEL[emotion] ?? emotion}
            </span>
            <div className="relative h-3 flex-1 overflow-hidden rounded bg-border">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.min(100, p * 100)}%`,
                  backgroundColor: emotionColor(emotion),
                }}
              />
              {ev !== undefined && (
                <div
                  className="absolute inset-y-0 border-r-2 border-foreground/70"
                  style={{ left: 0, width: `${Math.min(100, ev * 100)}%` }}
                  title={`ranking evidence ${pct(ev)}`}
                />
              )}
            </div>
            <span className="w-10 shrink-0 text-right tabular-nums">
              {pct(p)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
