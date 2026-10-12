"use client";

import type { FaceAnalysis } from "../../lib/types";
import {
  EMOTION_EMOJI,
  EMOTION_LABEL,
  dualLabel,
  emotionColor,
  pct,
  personLabel,
  primaryWithDual,
} from "../../lib/emotions";

interface Props {
  face: FaceAnalysis;
  selected: boolean;
  onSelect: () => void;
  onWhy: () => void;
  onHeatmap: () => void;
  /** Small contextual chip (e.g. "live frame" in periodic/verdict mode). */
  badge?: string;
}

const CERTAINTY_STYLE: Record<string, string> = {
  certain: "bg-emerald-500/15 text-emerald-400 border-emerald-500/40",
  uncertain: "bg-amber-500/15 text-amber-400 border-amber-500/40",
  ambiguous: "bg-red-500/15 text-red-400 border-red-500/40",
  insufficient: "bg-amber-500/15 text-amber-400 border-amber-500/40",
};

export default function PersonCard({
  face,
  selected,
  onSelect,
  onWhy,
  onHeatmap,
  badge,
}: Props) {
  const color = emotionColor(face.emotion);
  const certaintyClass =
    CERTAINTY_STYLE[face.certainty] ?? CERTAINTY_STYLE.uncertain;
  // Always the maximum emotion; when a runner-up is nearly tied, show both
  // majors (e.g. "Happy / Surprise 45% / 42%").
  const { primary, dual } = primaryWithDual(face.probabilities, {
    label: face.emotion,
    confidence: face.confidence,
  });
  const mainLabel = primary
    ? (EMOTION_LABEL[primary.label] ?? primary.label)
    : (EMOTION_LABEL[face.emotion] ?? face.emotion);
  const mainConf = primary ? primary.confidence : face.confidence;

  return (
    <div
      role="group"
      aria-label={`${personLabel(face.face_id)} ${face.emotion}`}
      onClick={onSelect}
      className={`cursor-pointer rounded-lg border p-3 transition ${
        selected
          ? "border-accent bg-accent/10"
          : "border-border bg-surface hover:border-border/80"
      }`}
    >
      <div className="mb-1 flex items-center justify-between">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          aria-pressed={selected}
          className="flex items-center gap-1.5 text-xs font-semibold text-muted hover:text-foreground"
        >
          {personLabel(face.face_id)}
          <span className="sr-only">{selected ? " (selected)" : " (select)"}</span>
          {badge && (
            <span className="rounded border border-border bg-background px-1 py-px text-[9px] font-medium uppercase tracking-wide text-muted">
              {badge}
            </span>
          )}
        </button>
        <span
          className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase ${certaintyClass}`}
        >
          {face.certainty}
        </span>
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-2xl" aria-hidden>
          {EMOTION_EMOJI[face.emotion] ?? "🙂"}
        </span>
        <span
          className="text-lg font-bold uppercase"
          style={{ color }}
        >
          {dual
            ? dualLabel({ label: primary!.label, confidence: mainConf }, dual)
            : mainLabel}
        </span>
      </div>
      {dual && (
        <p className="mt-0.5 font-mono text-[11px] text-muted">
          {pct(mainConf)} / {pct(dual.confidence)}
        </p>
      )}

      <dl className="mt-2 space-y-1 text-xs">
        <div className="flex justify-between">
          <dt className="text-muted">Confidence</dt>
          <dd className="font-medium">{pct(mainConf)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted">Intensity</dt>
          <dd className="font-medium">
            {Math.round(face.intensity)}
            <span className="text-muted"> ({face.intensity_category})</span>
          </dd>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded bg-border">
          <div
            className="h-full rounded"
            style={{
              width: `${Math.min(100, face.intensity)}%`,
              backgroundColor: color,
            }}
            aria-hidden
          />
        </div>
      </dl>

      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onWhy();
          }}
          className="flex-1 rounded border border-border px-2 py-1 text-xs hover:bg-border/50"
        >
          Why?
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onHeatmap();
          }}
          className="flex-1 rounded border border-border px-2 py-1 text-xs hover:bg-border/50"
        >
          Heatmap
        </button>
      </div>
    </div>
  );
}
