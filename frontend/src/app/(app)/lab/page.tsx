"use client";

import { useEffect, useRef, useState, type ChangeEvent } from "react";
import Button from "../../../components/common/Button";
import Card from "../../../components/common/Card";
import EmotionSpectrum from "../../../components/emotion/EmotionSpectrum";
import RegionChart from "../../../components/xai/RegionChart";
import { apiFetch, ApiError } from "../../../lib/api";
import {
  EMOTION_EMOJI,
  EMOTION_LABEL,
  emotionColor,
  personLabel,
  pct,
} from "../../../lib/emotions";
import type { AnalysisOut, ModelInfo, VideoAnalysisOut } from "../../../lib/types";
import { useAuth } from "../../../stores/auth";
import { usePageTitle } from "../../../hooks/usePageTitle";

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_VIDEO_BYTES = 60 * 1024 * 1024;

export default function LabPage() {
  usePageTitle("Research Lab");
  const token = useAuth((s) => s.token);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisOut | null>(null);
  const [video, setVideo] = useState<VideoAnalysisOut | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const previewRef = useRef<string | null>(null);

  const isVideo = file?.type.startsWith("video/") ?? false;

  const loadModels = (tok: string | null, cancelled?: { value: boolean }) => {
    setModelsLoading(true);
    setModelsError(false);
    apiFetch<ModelInfo[]>("/api/models", {}, tok)
      .then((data) => {
        if (cancelled?.value) return;
        setModels(data);
      })
      .catch(() => {
        if (cancelled?.value) return;
        setModelsError(true);
      })
      .finally(() => {
        if (!cancelled?.value) setModelsLoading(false);
      });
  };

  useEffect(() => {
    const cancelled = { value: false };
    loadModels(token, cancelled);
    return () => {
      cancelled.value = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    },
    [],
  );

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const chosen = e.target.files?.[0];
    if (!chosen) return;
    const isVid = chosen.type.startsWith("video/");
    if (!isVid && !chosen.type.startsWith("image/")) {
      setError("Only photos (JPG/PNG) and videos (MP4/MOV/AVI/MKV/WEBM) are supported.");
      return;
    }
    if (isVid && chosen.size > MAX_VIDEO_BYTES) {
      setError("Video too large (max 60MB).");
      return;
    }
    if (!isVid && chosen.size > MAX_IMAGE_BYTES) {
      setError("Image too large (max 8MB).");
      return;
    }
    setError(null);
    setResult(null);
    setVideo(null);
    setFile(chosen);
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const url = URL.createObjectURL(chosen);
    previewRef.current = url;
    setPreviewUrl(url);
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      if (isVideo) {
        const data = await apiFetch<VideoAnalysisOut>(
          "/api/analyze/video",
          { method: "POST", body: form },
          token,
        );
        setVideo(data);
        setResult(null);
        if (data.frames_analyzed === 0) setError("No frames could be analysed.");
      } else {
        form.append("xai", "true");
        form.append("face_id", "0");
        const data = await apiFetch<AnalysisOut>(
          "/api/analyze",
          { method: "POST", body: form },
          token,
        );
        setResult(data);
        setVideo(null);
        if (data.faces.length === 0) {
          setError("No face detected in this image.");
        }
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  const primary = result?.faces[0] ?? null;
  const xai = video ? video.xai : result?.xai ?? null;

  return (
    <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-2">
      <h1 className="sr-only">Research Lab</h1>
      {/* upload */}
      <Card title="Upload">
        <label
          htmlFor="lab-file"
          className="flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed border-border bg-background p-6 text-center text-sm text-muted transition hover:border-accent"
        >
          {previewUrl ? (
            isVideo ? (
              <video
                src={previewUrl}
                controls
                playsInline
                className="max-h-64 rounded bg-black object-contain"
              />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={previewUrl}
                alt="Upload preview"
                className="max-h-64 rounded object-contain"
              />
            )
          ) : (
            <>
              <span className="font-medium text-foreground">
                Click to choose a photo or video
              </span>
              <span className="mt-1 text-xs">
                JPG / PNG · MP4 / MOV / AVI / MKV / WEBM · stays on localhost
              </span>
            </>
          )}
        </label>
        <input
          id="lab-file"
          type="file"
          accept="image/*,video/*"
          className="sr-only"
          onChange={onFile}
        />
        <div className="mt-3 flex gap-2">
          <Button onClick={analyze} disabled={!file || busy}>
            {busy
              ? isVideo
                ? "Analyzing video…"
                : "Analyzing…"
              : isVideo
                ? "Analyze video"
                : "Analyze photo"}
          </Button>
          {file && (
            <span className="self-center truncate text-xs text-muted">
              {file.name}
              {isVideo && (
                <span className="ml-2 uppercase text-accent">video</span>
              )}
            </span>
          )}
        </div>
        {isVideo && (
          <p className="mt-2 text-[11px] text-muted">
            The clip is decoded locally and sampled at your configured analysis
            rate (max 120s / 60MB). No session is recorded and nothing is
            stored.
          </p>
        )}
        <div className="mt-3">
          <p className="mb-1.5 text-xs text-muted">
            Or try a sample (public-domain photo, served locally):{" "}
            <a
              href="/samples/README.md"
              target="_blank"
              rel="noreferrer"
              className="underline hover:text-foreground"
            >
              credits
            </a>
          </p>
          <div className="flex gap-2">
            {(
              [
                ["einstein.jpg", "Einstein"],
                ["obama.jpg", "Obama"],
                ["astronaut.png", "Astronaut"],
              ] as const
            ).map(([file_name, label]) => (
              <button
                key={file_name}
                type="button"
                className="rounded-md border border-border px-2 py-1 text-xs text-muted transition hover:border-accent hover:text-foreground"
                onClick={async () => {
                  try {
                    const res = await fetch(`/samples/${file_name}`);
                    const blob = await res.blob();
                    const chosen = new File(
                      [blob],
                      file_name,
                      { type: blob.type },
                    );
                    onFile({
                      target: { files: [chosen] },
                    } as unknown as ChangeEvent<HTMLInputElement>);
                  } catch {
                    setError("Failed to load sample image.");
                  }
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {error && (
          <p role="alert" className="mt-2 text-sm text-red-400">
            {error}
          </p>
        )}
      </Card>

      {/* analysis panel */}
      <Card title={video ? "Video analysis" : "Analysis"}>
        {!result && !video && (
          <p className="text-sm text-muted">
            Upload a photo or a video and run Analyze to see predictions,
            ranking, and evidence.
          </p>
        )}
        {result && !primary && !video && (
          <p className="text-sm text-muted">
            {error ??
              "No face was detected in this image. Try a clearer, front-facing photo with a larger face."}
          </p>
        )}
        {video && (
          <VideoResult result={video} />
        )}
        {result && primary && (
          <div className="space-y-4">
            <div className="flex items-center justify-between rounded-md border border-border bg-background p-3">
              <div>
                <p className="text-xs text-muted">{personLabel(primary.face_id)}</p>
                <p
                  className="text-2xl font-bold uppercase"
                  style={{ color: emotionColor(primary.emotion) }}
                >
                  {EMOTION_EMOJI[primary.emotion] ?? ""}{" "}
                  {EMOTION_LABEL[primary.emotion] ?? primary.emotion}
                </p>
              </div>
              <div className="space-y-1 text-right text-sm">
                <p>Confidence {pct(primary.confidence, 1)}</p>
                <p>
                  Intensity {Math.round(primary.intensity)} (
                  {primary.intensity_category})
                </p>
                <p className="uppercase text-muted">{primary.certainty}</p>
              </div>
            </div>

            <div>
              <h4 className="mb-2 text-xs font-semibold uppercase text-muted">
                Emotion ranking
              </h4>
              <EmotionSpectrum
                probabilities={primary.probabilities}
                ranked={primary.ranked_emotions}
                evidenceScores={primary.evidence_scores}
              />
            </div>

            <div className="rounded-md border border-border bg-background p-2 text-[11px] text-muted">
              Action units: omitted —{" "}
              <span className="uppercase">{primary.au_model_status}</span> (AU
              model untrained; ranking weight w_AU = 0). Multiple faces:{" "}
              {result.faces.length}.
            </div>

            {result.faces.length > 1 && (
              <div className="space-y-2">
                {result.faces.slice(1).map((face) => (
                  <div
                    key={face.face_id}
                    className="flex items-center justify-between rounded border border-border px-2 py-1 text-xs"
                  >
                    <span>{personLabel(face.face_id)}</span>
                    <span
                      style={{ color: emotionColor(face.emotion) }}
                      className="font-semibold uppercase"
                    >
                      {EMOTION_LABEL[face.emotion] ?? face.emotion}
                    </span>
                    <span>{pct(face.confidence)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* XAI inspector */}
      <Card title="XAI inspector — Grad-CAM">
        {!xai && (
          <p className="text-sm text-muted">
            Heatmap and region relevance appear after analysis (computed
            on-demand from the trained classifier).
          </p>
        )}
        {xai && (
          <div className="space-y-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${xai.heatmap}`}
              alt={`Grad-CAM heatmap for ${xai.emotion}`}
              className="mx-auto max-h-72 rounded border border-border"
            />
            <div className="flex justify-center gap-3 text-xs text-muted">
              <span>method: {xai.method}</span>
              <span>
                target: {EMOTION_LABEL[xai.emotion] ?? xai.emotion} ·{" "}
                {pct(xai.confidence, 1)}
              </span>
            </div>
            <RegionChart regions={xai.regions} />
            <p className="text-[11px] text-muted">
              Regions = share of Grad-CAM mass over geometric face areas (no
              landmarks). Region→emotion evidence reweights the final ranking.
            </p>
          </div>
        )}
      </Card>

      {/* model info */}
      <Card title="Model inspector">
        {modelsLoading ? (
          <div className="space-y-2">
            <div className="h-14 animate-pulse rounded-md bg-border/50" />
            <div className="h-14 animate-pulse rounded-md bg-border/50" />
            <div className="h-14 animate-pulse rounded-md bg-border/50" />
          </div>
        ) : modelsError ? (
          <div className="space-y-2 text-sm text-muted">
            <p>
              Model registry unavailable — is the API running on port 8765?
            </p>
            <Button variant="ghost" onClick={() => loadModels(token)}>
              Retry
            </Button>
          </div>
        ) : models.length === 0 ? (
          <p className="text-sm text-muted">
            No models registered yet. Run the training pipeline or seed the
            registry to populate this list.
          </p>
        ) : (
          <div className="space-y-3">
            {models.map((m) => {
              const metrics = m.metrics ?? {};
              const primary = metrics.test_acc ?? metrics.macro_f1_or_bacc;
              const secondary =
                metrics.test_bacc ?? metrics.test_macro_f1 ?? null;
              const isInUse = m.status === "deployed";
              return (
                <div
                  key={`${m.model_name}-${m.version}`}
                  className="rounded-md border border-border bg-background p-3 text-xs"
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="font-semibold">
                      {m.model_name} v{m.version}
                      {isInUse && (
                        <span className="ml-2 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium uppercase text-emerald-400">
                          In use
                        </span>
                      )}
                    </span>
                    <span
                      className={`rounded px-1.5 py-0.5 uppercase ${
                        m.status === "deployed"
                          ? "bg-emerald-500/15 text-emerald-400"
                          : m.status === "validated"
                            ? "bg-sky-500/15 text-sky-400"
                            : "bg-border text-muted"
                      }`}
                    >
                      {m.status}
                    </span>
                  </div>
                  <p className="text-muted">
                    {m.model_type} · {m.dataset_used}
                  </p>
                  {typeof primary === "number" && (
                    <div className="mt-2 space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-muted">
                          {metrics.test_acc !== undefined
                            ? "Accuracy"
                            : "F1 / BAcc"}
                        </span>
                        <span className="tabular-nums font-medium">
                          {primary.toFixed(2)}%
                        </span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-border">
                        <div
                          className={`h-full rounded-full ${
                            isInUse ? "bg-emerald-500" : "bg-accent"
                          }`}
                          style={{
                            width: `${Math.min(100, Math.max(0, primary))}%`,
                          }}
                        />
                      </div>
                      {typeof secondary === "number" && (
                        <div className="flex items-center justify-between text-[11px]">
                          <span className="text-muted">
                            {metrics.test_bacc !== undefined
                              ? "Balanced acc"
                              : "Macro F1"}
                          </span>
                          <span className="tabular-nums font-medium">
                            {secondary.toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
            <p className="pt-1 text-[11px] text-muted">
              RAF-DB test set · n=3,068 · metrics from local experiments
              (EXP-007/008/009/010), never fabricated. Only{" "}
              <span className="text-foreground">deployed</span> models serve
              live inference.
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}

/* ---------------- video results ---------------- */

function Stat({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2">
      <p className="text-[10px] uppercase tracking-widest text-muted">{label}</p>
      <p className="mt-0.5 text-sm font-medium tabular-nums">{value}</p>
    </div>
  );
}

function VideoResult({ result }: { result: VideoAnalysisOut }) {
  const peak = result.peak_frame;
  const dist = result.emotion_distribution;
  const maxShare = Math.max(...dist.map((d) => d.share), 0.0001);

  return (
    <div className="space-y-4">
      {/* headline */}
      {result.face_coverage < 0.5 && (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/[0.07] p-2 text-xs text-amber-200">
          A face was detected in only {pct(result.face_coverage, 0)} of sampled
          frames. Crop closer to the subject or use a higher-resolution clip —
          no classification is invented for frames without a face.
        </p>
      )}
      <div
        className="flex items-center justify-between rounded-md border border-border bg-background p-3"
        style={
          result.dominant_emotion
            ? {
                borderColor: `${emotionColor(result.dominant_emotion)}66`,
              }
            : undefined
        }
      >
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted">
            Dominant emotion
          </p>
          <p
            className="text-2xl font-bold uppercase"
            style={{
              color: result.dominant_emotion
                ? emotionColor(result.dominant_emotion)
                : undefined,
            }}
          >
            {result.dominant_emotion
              ? `${EMOTION_EMOJI[result.dominant_emotion] ?? ""} ${
                  EMOTION_LABEL[result.dominant_emotion] ?? result.dominant_emotion
                }`
              : "—"}
          </p>
        </div>
        <div className="space-y-0.5 text-right text-xs text-muted">
          <p>
            avg confidence{" "}
            <span className="text-foreground tabular-nums">
              {result.avg_confidence !== null
                ? pct(result.avg_confidence, 1)
                : "—"}
            </span>
          </p>
          <p>
            avg intensity{" "}
            <span className="text-foreground tabular-nums">
              {result.avg_intensity !== null
                ? Math.round(result.avg_intensity)
                : "—"}
            </span>
          </p>
          <p>
            transitions{" "}
            <span className="text-foreground tabular-nums">
              {result.transition_count}
            </span>
          </p>
        </div>
      </div>

      {/* run metadata */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Frames analysed" value={result.frames_analyzed} />
        <Stat label="Duration" value={`${result.duration_seconds.toFixed(1)}s`} />
        <Stat
          label="Sampling"
          value={`${result.sample_fps} fps (1/${result.sample_every_n_frames})`}
        />
        <Stat
          label="Face coverage"
          value={`${pct(result.face_coverage, 0)} (${result.frames_with_face}/${result.frames_analyzed})`}
        />
        <Stat label="Faces / frame" value={result.faces_per_frame_avg} />
      </div>

      {/* distribution */}
      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase text-muted">
          Emotion distribution across the clip
        </h4>
        <ul className="space-y-1.5">
          {dist.map((d) => (
            <li key={d.emotion} className="flex items-center gap-2 text-xs">
              <span className="w-16 shrink-0 uppercase" style={{ color: emotionColor(d.emotion) }}>
                {EMOTION_LABEL[d.emotion] ?? d.emotion}
              </span>
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-border/60">
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${Math.max(2, (d.share / maxShare) * 100)}%`,
                    background: emotionColor(d.emotion),
                  }}
                />
              </span>
              <span className="w-20 shrink-0 text-right tabular-nums text-muted">
                {pct(d.share, 0)} · {d.count}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* transitions */}
      {result.transitions.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase text-muted">
            Transitions
          </h4>
          <ul className="max-h-40 space-y-1 overflow-y-auto pr-1">
            {result.transitions.map((t, i) => (
              <li
                key={`${t.t}-${i}`}
                className="flex items-center justify-between rounded border border-border px-2 py-1 text-xs"
              >
                <span className="tabular-nums text-muted">{t.t.toFixed(1)}s</span>
                <span className="uppercase">
                  <span style={{ color: emotionColor(t.from_emotion ?? "") }}>
                    {EMOTION_LABEL[t.from_emotion ?? ""] ?? t.from_emotion ?? "—"}
                  </span>
                  <span className="mx-1.5 text-muted">→</span>
                  <span style={{ color: emotionColor(t.to_emotion ?? "") }}>
                    {EMOTION_LABEL[t.to_emotion ?? ""] ?? t.to_emotion ?? "—"}
                  </span>
                </span>
                <span className="tabular-nums text-muted">
                  {t.confidence !== null ? pct(t.confidence, 0) : ""}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* peak frame */}
      {peak && (
        <div className="rounded-md border border-border bg-background p-3">
          <h4 className="text-xs font-semibold uppercase text-muted">
            Peak frame
          </h4>
          <div className="mt-2 flex items-center justify-between text-xs">
            <span className="text-muted">
              t = {peak.t.toFixed(1)}s · frame {peak.index}
            </span>
            <span className="uppercase" style={{ color: emotionColor(peak.emotion) }}>
              {EMOTION_LABEL[peak.emotion] ?? peak.emotion} {pct(peak.confidence, 1)}
            </span>
          </div>
          <div className="mt-2">
            <EmotionSpectrum
              probabilities={peak.probabilities}
              ranked={peak.ranked_emotions}
              evidenceScores={peak.evidence_scores}
            />
          </div>
        </div>
      )}

      {/* narration */}
      {result.narration.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase text-muted">
            Narration
          </h4>
          <ul className="space-y-1 text-xs text-muted">
            {result.narration.map((n, i) => (
              <li key={i} className="border-l-2 border-border pl-2">
                {n.text}
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-[11px] text-muted">
        Offline run over {result.frames_analyzed} sampled frames · model{" "}
        {result.model} · AU branch untrained (w_AU = 0) · intensity is a
        distribution-extremity proxy, not a clinical measure. Video runs are not
        saved as sessions.
      </p>
    </div>
  );
}
