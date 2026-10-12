"use client";

import { emotionColor, EMOTION_LABEL } from "../../lib/emotions";
import type { TimelineSample } from "../../lib/types";

interface Props {
  samples: TimelineSample[];
  max?: number;
}

/**
 * Compact horizontal emotion strip: one column per 1s sample,
 * colored by the dominant face emotion in that sample.
 */
export default function EmotionTimeline({ samples, max = 120 }: Props) {
  const slice = samples.slice(-max);

  if (slice.length === 0) {
    return (
      <p className="py-3 text-center text-xs text-muted">
        Timeline will build as the session runs (1 sample / second)…
      </p>
    );
  }

  return (
    <div className="flex items-center gap-3 overflow-x-auto py-1">
      <div className="flex h-8 shrink-0 gap-px" aria-label="Emotion timeline">
        {slice.map((sample, i) => {
          const face = sample.faces[0];
          const emotion = face?.emotion ?? "neutral";
          return (
            <div
              key={`${sample.t}-${i}`}
              title={`${new Date(sample.t * 1000).toLocaleTimeString()} — ${
                EMOTION_LABEL[emotion] ?? emotion
              }${face ? ` ${Math.round(face.intensity)}` : ""}`}
              className="w-1.5 rounded-sm"
              style={{
                backgroundColor: emotionColor(emotion),
                opacity: face ? 0.45 + Math.min(1, face.intensity / 100) * 0.55 : 0.3,
              }}
            />
          );
        })}
      </div>
      <div className="flex shrink-0 flex-wrap gap-x-2 gap-y-0.5 text-[10px] text-muted">
        {["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"].map(
          (e) => (
            <span key={e} className="inline-flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ backgroundColor: emotionColor(e) }}
              />
              {EMOTION_LABEL[e]}
            </span>
          ),
        )}
      </div>
    </div>
  );
}
