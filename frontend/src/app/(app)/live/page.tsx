"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Button from "../../../components/common/Button";
import Card from "../../../components/common/Card";
import PersonCard from "../../../components/emotion/PersonCard";
import EmotionSpectrum from "../../../components/emotion/EmotionSpectrum";
import RegionChart from "../../../components/xai/RegionChart";
import CommentaryPanel from "../../../components/narrator/CommentaryPanel";
import EmotionTimeline from "../../../components/timeline/EmotionTimeline";
import { WS_URL, apiFetch } from "../../../lib/api";
import {
  EMOTION_LABEL,
  emotionColor,
  personLabel,
  pct,
} from "../../../lib/emotions";
import type {
  FaceAnalysis,
  NarrationLine,
  TimelineSample,
  UserSettings,
  WsServerMessage,
  XaiPayload,
} from "../../../lib/types";
import { useAuth } from "../../../stores/auth";
import { toast } from "../../../stores/toast";
import { usePageTitle } from "../../../hooks/usePageTitle";
import { useEmotionHold, type HeldFace } from "../../../lib/emotionHold";
import {
  DEFAULT_FPS as CADENCE_DEFAULT_FPS,
  formatDuration,
  startCadence,
  type CadenceHandle,
} from "../../../lib/cadence";
import {
  computeMotionScore,
  createMotionGate,
  DEFAULT_MOTION_COOLDOWN_MS,
  DEFAULT_MOTION_THRESHOLD,
  MOTION_H,
  MOTION_POLL_HZ,
  MOTION_W,
  shouldTriggerMotion,
  type MotionGate,
} from "../../../lib/motion";
import {
  dominantRegion,
  REGION_FRAC,
  VERDICT_INSTANT_MIN,
  VERDICT_STALE_MS,
  VerdictBuffer,
  type Box,
  type Verdict,
} from "../../../lib/verdict";

const CAPTURE_W = 640;
const CAPTURE_H = 480;
const DEFAULT_FPS = CADENCE_DEFAULT_FPS; // fallback when settings have not loaded
const NO_INTERVAL = 100000; // effectively "never" — transition-driven XAI

type WsStatus = "closed" | "connecting" | "open" | "reconnecting";

function speak(text: string, rate: number, voiceURI: string): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = rate;
  if (voiceURI && voiceURI !== "default") {
    const voice = window.speechSynthesis
      .getVoices()
      .find((v) => v.voiceURI === voiceURI);
    if (voice) utterance.voice = voice;
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

const SHORTCUTS: [string, string][] = [
  ["Space", "Pause / resume sending frames"],
  ["M", "Mirror the camera view"],
  ["H", "Toggle heatmap"],
  ["S", "Save snapshot (PNG)"],
  ["?", "Show / hide this help"],
  ["Esc", "Close help"],
];

export default function LivePage() {
  usePageTitle("Live Observer");
  const router = useRouter();
  const token = useAuth((s) => s.token);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const sendRef = useRef<HTMLCanvasElement | null>(null);
  const displayRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewTimerRef = useRef<number | null>(null);
  const motionTimerRef = useRef<number | null>(null);
  const intentionalCloseRef = useRef(false);
  const retryRef = useRef(0);
  const pausedRef = useRef(false);
  const mirrorRef = useRef(false);
  const settingsRef = useRef<UserSettings | null>(null);
  const sessionTitleRef = useRef("");
  const xaiModeRef = useRef<"interval" | "transition">("interval");

  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [mirror, setMirror] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [holdSeconds, setHoldSeconds] = useState(0);
  const [wsStatus, setWsStatus] = useState<WsStatus>("closed");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionTitle, setSessionTitle] = useState("");
  const [xaiMode, setXaiMode] = useState<"interval" | "transition">("interval");
  const [faces, setFaces] = useState<FaceAnalysis[]>([]);
  const [selectedId, setSelectedId] = useState(0);
  const [timeline, setTimeline] = useState<TimelineSample[]>([]);
  const [narration, setNarration] = useState<NarrationLine[]>([]);
  const [xai, setXai] = useState<XaiPayload | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  const [frameId, setFrameId] = useState<number | null>(null);
  const [eventCount, setEventCount] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [voiceOn, setVoiceOn] = useState(true);
  const [helpOpen, setHelpOpen] = useState(false);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [verdictSamples, setVerdictSamples] = useState(0);
  const [verdictStale, setVerdictStale] = useState(false);

  const voiceOnRef = useRef(voiceOn);
  const verdictBufRef = useRef(new VerdictBuffer());
  const verdictBoxRef = useRef<Box | null>(null);
  const lastFaceAtRef = useRef<number | null>(null);
  const instantFlushRef = useRef(false);
  const isPeriodicRef = useRef(false);
  const demoModeRef = useRef(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    demoModeRef.current =
      new URLSearchParams(window.location.search).get("demo") === "verdict";
  }, []);
  useEffect(() => {
    voiceOnRef.current = voiceOn;
    if (!voiceOn && typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, [voiceOn]);

  pausedRef.current = paused;
  mirrorRef.current = mirror;
  sessionTitleRef.current = sessionTitle;
  xaiModeRef.current = xaiMode;

  /* ---------- settings (voice rate, xai interval, detail, fps, dwell) ---------- */
  useEffect(() => {
    let cancelled = false;
    apiFetch<UserSettings>("/api/settings", {}, token)
      .then((data) => {
        if (cancelled) return;
        settingsRef.current = data;
        setSettings(data);
        setHoldSeconds(data.emotion_hold_seconds ?? 0);
      })
      .catch(() => {
        /* defaults are fine */
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  /* ---------- drawing ---------- */
  const drawDisplay = useCallback(() => {
    const video = videoRef.current;
    const display = displayRef.current;
    if (!video || !display) return;
    const ctx = display.getContext("2d");
    if (!ctx) return;
    ctx.save();
    if (mirrorRef.current) {
      ctx.translate(CAPTURE_W, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    ctx.restore();
  }, []);

  const drawOverlay = useCallback(
    (
      list: FaceAnalysis[],
      held: HeldFace | null = null,
      v: Verdict | null = null,
      stale = false,
    ) => {
      const canvas = overlayRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (v) {
        // Verdict mode: faint live tracking frame under the pinned verdict so
        // continuous detection stays legible between flushes.
        const liveBox: Box | null = list[0]?.box ?? null;
        for (const face of list) {
          const [bx, by, w, h] = face.box;
          const x = mirrorRef.current ? CAPTURE_W - bx - w : bx;
          ctx.strokeStyle = "rgba(255,255,255,0.35)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, by, w, h);
        }

        // Chrome tracks the live box (lerped at preview rate) so brackets
        // always hug the face; the window-mean box stays analytics-only.
        const target = liveBox ?? v.box;
        const lerp = 0.25;
        const cur = verdictBoxRef.current;
        const bx = cur ? cur[0] + (target[0] - cur[0]) * lerp : target[0];
        const by = cur ? cur[1] + (target[1] - cur[1]) * lerp : target[1];
        const bw = cur ? cur[2] + (target[2] - cur[2]) * lerp : target[2];
        const bh = cur ? cur[3] + (target[3] - cur[3]) * lerp : target[3];
        verdictBoxRef.current = [bx, by, bw, bh];
        const x = mirrorRef.current ? CAPTURE_W - bx - bw : bx;

        const color = v.uncertain ? "#d4d4d8" : emotionColor(v.label);
        ctx.save();
        ctx.globalAlpha = stale ? 0.4 : 1;

        // Soft glow on the dominant evidence region (skipped when uncertain).
        if (!v.uncertain) {
          const region = dominantRegion(v.regions);
          const frac = region ? REGION_FRAC[region] : null;
          const pctRegion = region && v.regions ? (v.regions[region] ?? 0) : 0;
          const strength = Math.min(1, pctRegion / 40);
          if (frac && strength > 0.05) {
            for (const [y0, y1, x0, x1] of frac) {
              const cx = x + ((x0 + x1) / 2) * bw;
              const cy = by + ((y0 + y1) / 2) * bh;
              const rx = ((x1 - x0) / 2) * bw * 1.25;
              const ry = ((y1 - y0) / 2) * bh * 1.25;
              const g = ctx.createRadialGradient(
                cx,
                cy,
                0,
                cx,
                cy,
                Math.max(rx, ry),
              );
              g.addColorStop(0, color);
              g.addColorStop(1, "rgba(0,0,0,0)");
              ctx.fillStyle = g;
              ctx.globalAlpha = (stale ? 0.4 : 1) * 0.35 * strength;
              ctx.beginPath();
              ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.globalAlpha = stale ? 0.4 : 1;
          }
        }

        // Corner brackets.
        const r = 14;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        const rx0 = x;
        const ry0 = by;
        const rx1 = x + bw;
        const ry1 = by + bh;
        ctx.beginPath();
        ctx.moveTo(rx0, ry0 + r);
        ctx.lineTo(rx0, ry0);
        ctx.lineTo(rx0 + r, ry0);
        ctx.moveTo(rx1 - r, ry0);
        ctx.lineTo(rx1, ry0);
        ctx.lineTo(rx1, ry0 + r);
        ctx.moveTo(rx1, ry1 - r);
        ctx.lineTo(rx1, ry1);
        ctx.lineTo(rx1 - r, ry1);
        ctx.moveTo(rx0 + r, ry1);
        ctx.lineTo(rx0, ry1);
        ctx.lineTo(rx0, ry1 - r);
        ctx.stroke();

        // Trait chip: label + confidence, runner-up when the window was mixed.
        const label = v.uncertain
          ? "UNCERTAIN"
          : (EMOTION_LABEL[v.label] ?? v.label).toUpperCase();
        const runner = v.runnerUp
          ? ` · ${(EMOTION_LABEL[v.runnerUp.label] ?? v.runnerUp.label).toUpperCase()} ${pct(v.runnerUp.confidence)}`
          : "";
        const chip = `VERDICT · ${label} ${pct(v.confidence)}${runner}${
          stale ? " · STALE" : ""
        } · n=${v.sampleCount}`;
        ctx.font = "bold 12px sans-serif";
        const tw = ctx.measureText(chip).width + 14;
        const ly = by > 26 ? by - 26 : by + bh + 6;
        ctx.fillStyle = "rgba(0,0,0,0.78)";
        ctx.fillRect(x, ly, tw, 20);
        ctx.fillStyle = color;
        ctx.fillRect(x, ly, 3, 20);
        ctx.fillStyle = v.uncertain ? "rgba(255,255,255,0.88)" : color;
        ctx.fillText(chip, x + 9, ly + 14);
        ctx.restore();
        return;
      }

      for (const face of list) {
        // Single-person product mode: the dwell applies to the tracked subject.
        const shown: HeldFace =
          held && held.face_id === face.face_id
            ? held
            : { ...face, holding: false, holdRemaining: 0, holdProgress: 0 };
        const [bx, by, w, h] = shown.box;
        const x = mirrorRef.current ? CAPTURE_W - bx - w : bx;
        const color = emotionColor(shown.emotion);
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(x, by, w, h);
        const label = `${personLabel(shown.face_id)} ${(
          EMOTION_LABEL[shown.emotion] ?? shown.emotion
        ).toUpperCase()} ${pct(shown.confidence)}`;
        ctx.font = "bold 13px sans-serif";
        const tw = ctx.measureText(label).width + 10;
        const ly = by > 20 ? by - 22 : by + h + 4;
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.fillRect(x, ly, tw, 18);
        ctx.fillStyle = color;
        ctx.fillText(label, x + 5, ly + 13);

        // Dwell countdown: thin bar under the label while the value is held.
        if (shown.holding) {
          const bw = tw;
          const by2 = ly + 19;
          ctx.fillStyle = "rgba(0,0,0,0.65)";
          ctx.fillRect(x, by2, bw, 3);
          ctx.fillStyle = color;
          ctx.fillRect(x, by2, Math.round(bw * shown.holdProgress), 3);
          ctx.font = "10px sans-serif";
          ctx.fillStyle = "rgba(255,255,255,0.7)";
          ctx.fillText(`hold ${shown.holdRemaining.toFixed(1)}s`, x, by2 + 14);
        }
      }
    },
    [],
  );

  /* ---------- emotion dwell (display hold) ---------- */
  // Dwell is a Live-mode display trick; in periodic mode the verdict window
  // itself is the hold, so it is disabled there.
  const holdMs =
    settings?.analysis_mode === "periodic" ? 0 : holdSeconds * 1000;
  const heldFace = useEmotionHold(faces, holdMs);

  useEffect(() => {
    drawOverlay(faces, heldFace, verdict, verdictStale);
  }, [faces, heldFace, verdict, verdictStale, drawOverlay, mirror]);

  useEffect(() => {
    if (running) drawDisplay();
  }, [mirror, running, drawDisplay]);

  useEffect(() => {
    if (faces.length === 0) return;
    if (!faces.some((f) => f.face_id === selectedId)) {
      setSelectedId(faces[0].face_id);
    }
  }, [faces, selectedId]);

  /* ---------- frame capture loop ---------- */
  const captureAndSend = useCallback(() => {
    if (pausedRef.current) return;
    const video = videoRef.current;
    const send = sendRef.current;
    const ws = wsRef.current;
    if (!video || !send || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > 1_000_000) return; // backpressure
    drawDisplay();
    const ctx = send.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    send.toBlob(
      (blob) => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(blob);
        }
      },
      "image/jpeg",
      0.7,
    );
  }, [drawDisplay]);

  const analysisFps = useMemo(() => {
    const raw = settings?.analysis_fps ?? DEFAULT_FPS;
    return Math.min(15, Math.max(1, Math.round(raw) || DEFAULT_FPS));
  }, [settings?.analysis_fps]);

  const analysisMode = settings?.analysis_mode ?? "live";
  const isPeriodic = analysisMode === "periodic";
  const intervalMs = (settings?.analysis_interval_seconds ?? 120) * 1000;
  const motionEnabled = settings?.motion_trigger_enabled ?? true;
  isPeriodicRef.current = isPeriodic;

  /* ---------- local preview loop (independent of analysis cadence) ---------- */
  const startPreviewLoop = useCallback(() => {
    if (previewTimerRef.current !== null) return;
    previewTimerRef.current = window.setInterval(drawDisplay, 1000 / 30);
  }, [drawDisplay]);

  const stopPreviewLoop = useCallback(() => {
    if (previewTimerRef.current !== null) {
      window.clearInterval(previewTimerRef.current);
      previewTimerRef.current = null;
    }
  }, []);

  /* ---------- verdict (periodic mode) ---------- */
  const applyVerdict = useCallback((v: Verdict) => {
    setVerdict(v);
    setVerdictSamples(verdictBufRef.current.size);
    const s = settingsRef.current;
    const name = v.uncertain
      ? "Uncertain"
      : (EMOTION_LABEL[v.label] ?? v.label);
    setNarration(
      (prev) =>
        [
          {
            face_id: 0,
            event_type: "verdict",
            text: `Verdict — ${
              v.uncertain ? "UNCERTAIN" : name.toUpperCase()
            } ${pct(v.confidence)} · window n=${v.sampleCount}`,
            timestamp: Date.now() / 1000,
          },
          ...prev,
        ].slice(0, 80),
    );
    if (voiceOnRef.current) {
      speak(
        `Verdict: ${name}, ${pct(v.confidence)}`,
        s?.narrator_speed ?? 1,
        s?.narrator_voice ?? "default",
      );
    }
  }, []);

  const flushVerdict = useCallback(() => {
    const v = verdictBufRef.current.flush(Date.now());
    if (v) applyVerdict(v);
    setVerdictSamples(verdictBufRef.current.size);
  }, [applyVerdict]);

  /* ---------- cadence scheduler ---------- */
  const cadenceRef = useRef<CadenceHandle | null>(null);
  const verdictCadenceRef = useRef<CadenceHandle | null>(null);
  const lastAnalysisAtRef = useRef<number | null>(null);
  const [nextCheckSec, setNextCheckSec] = useState<number | null>(null);

  const tickCadence = useCallback(() => {
    // In periodic mode the honest countdown is to the next verdict flush.
    const handle = isPeriodicRef.current
      ? verdictCadenceRef.current
      : cadenceRef.current;
    setNextCheckSec(handle ? handle.secondsUntil(Date.now()) : null);
    const last = lastAnalysisAtRef.current;
    if (isPeriodicRef.current) {
      const stale = last !== null && Date.now() - last > VERDICT_STALE_MS;
      setVerdictStale((prev) => (prev === stale ? prev : stale));
    }
  }, []);

  const startSendTimer = useCallback(() => {
    cadenceRef.current?.stop();
    verdictCadenceRef.current?.stop();
    // Capture streams continuously in both modes; periodic mode pools the
    // stream into a verdict on its own interval cadence.
    cadenceRef.current = startCadence(
      { mode: "live", fps: analysisFps, intervalMs },
      () => {
        captureAndSend();
        lastAnalysisAtRef.current = Date.now();
        tickCadence();
      },
    );
    if (isPeriodic) {
      verdictCadenceRef.current = startCadence(
        { mode: "periodic", fps: analysisFps, intervalMs },
        () => {
          flushVerdict();
          tickCadence();
        },
      );
    }
    tickCadence();
  }, [
    captureAndSend,
    analysisFps,
    isPeriodic,
    intervalMs,
    tickCadence,
    flushVerdict,
  ]);

  const stopSendTimer = useCallback(() => {
    cadenceRef.current?.stop();
    cadenceRef.current = null;
    verdictCadenceRef.current?.stop();
    verdictCadenceRef.current = null;
    tickCadence();
  }, [tickCadence]);

  // Re-pace an in-flight session when the configured cadence changes.
  useEffect(() => {
    if (running && wsRef.current?.readyState === WebSocket.OPEN) {
      startSendTimer();
    }
  }, [analysisFps, isPeriodic, intervalMs, motionEnabled, running, startSendTimer]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(tickCadence, 1000);
    return () => window.clearInterval(id);
  }, [running, tickCadence]);

  /* ---------- motion watcher (periodic mode only) ---------- */
  const motionGateRef = useRef<MotionGate | null>(null);
  const motionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [motionLevel, setMotionLevel] = useState(0);

  const pollMotion = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    let canvas = motionCanvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.width = MOTION_W;
      canvas.height = MOTION_H;
      motionCanvasRef.current = canvas;
    }
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, MOTION_W, MOTION_H);
    const data = ctx.getImageData(0, 0, MOTION_W, MOTION_H).data;
    // Luma-ish: average the channels per pixel for a single-channel sample.
    const sample = new Uint8ClampedArray(MOTION_W * MOTION_H);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      sample[p] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    const gate = (motionGateRef.current ??= createMotionGate());
    const score = computeMotionScore(gate.last, sample);
    gate.last = sample;
    setMotionLevel(score);

    if (
      motionEnabled &&
      isPeriodic &&
      !pausedRef.current &&
      wsRef.current?.readyState === WebSocket.OPEN &&
      shouldTriggerMotion(
        score,
        DEFAULT_MOTION_THRESHOLD,
        Date.now(),
        gate,
        DEFAULT_MOTION_COOLDOWN_MS,
      )
    ) {
      // Motion wakes the reading: flush the current window into an instant
      // verdict (capture itself is already continuous).
      if (verdictBufRef.current.size >= VERDICT_INSTANT_MIN) {
        const v = verdictBufRef.current.flush(Date.now());
        if (v) applyVerdict(v);
        setVerdictSamples(0);
      }
      tickCadence();
    }
  }, [motionEnabled, isPeriodic, tickCadence, applyVerdict]);

  const startMotionWatch = useCallback(() => {
    if (motionTimerRef.current !== null) return;
    motionTimerRef.current = window.setInterval(
      pollMotion,
      1000 / MOTION_POLL_HZ,
    );
  }, [pollMotion]);

  const stopMotionWatch = useCallback(() => {
    if (motionTimerRef.current !== null) {
      window.clearInterval(motionTimerRef.current);
      motionTimerRef.current = null;
    }
    motionGateRef.current = null;
    motionCanvasRef.current = null;
    setMotionLevel(0);
  }, []);

  /* ---------- websocket ---------- */
  const handleMessage = useCallback((msg: WsServerMessage) => {
    switch (msg.type) {
      case "session":
        setSessionId(msg.session_id);
        break;
      case "analysis": {
        // Demo mode drives the overlay from synthetic samples.
        if (!demoModeRef.current) setFaces(msg.faces ?? []);
        setFps(msg.fps ?? null);
        setFrameId(msg.frame_id ?? null);
        if (msg.events && msg.events.length > 0) {
          setEventCount((c) => c + msg.events.length);
        }
        if (msg.timeline) {
          setTimeline((prev) => [...prev, msg.timeline as TimelineSample].slice(-300));
        }
        if (msg.xai) setXai(msg.xai);
        if (isPeriodicRef.current) {
          const face = (msg.faces ?? [])[0];
          const now = Date.now();
          const buf = verdictBufRef.current;
          if (face) {
            const last = lastFaceAtRef.current;
            const wasStale = last !== null && now - last > VERDICT_STALE_MS;
            // Face changed or came back after a gap: close the old window and
            // let the fresh one pin an instant verdict instead of waiting.
            if (wasStale || (buf.faceId !== null && buf.faceId !== face.face_id)) {
              const leftover = buf.flush(now);
              if (leftover) applyVerdict(leftover);
              instantFlushRef.current = true;
            }
            lastFaceAtRef.current = now;
            buf.push({
              faceId: face.face_id,
              ts: now,
              box: face.box,
              probabilities: face.probabilities,
              regions: face.regions ?? null,
            });
            if (instantFlushRef.current && buf.size >= VERDICT_INSTANT_MIN) {
              instantFlushRef.current = false;
              const v = buf.flush(now);
              if (v) applyVerdict(v);
              setVerdictSamples(0);
            } else {
              setVerdictSamples(buf.size);
            }
          }
        }
        if (msg.narration && msg.narration.length > 0) {
          setNarration((prev) => [...msg.narration, ...prev].slice(0, 80));
          // Periodic mode narrates per verdict, not per frame.
          if (voiceOnRef.current && !isPeriodicRef.current) {
            const s = settingsRef.current;
            for (const line of msg.narration) {
              speak(
                line.text,
                s?.narrator_speed ?? 1,
                s?.narrator_voice ?? "default",
              );
            }
          }
        }
        break;
      }
      case "xai":
        setXai(msg);
        setShowHeatmap(true);
        break;
      case "error":
        setError(msg.error);
        break;
      default:
        break;
    }
  }, [applyVerdict]);

  const wsUrl = useCallback(() => {
    const params = new URLSearchParams({ token: token ?? "" });
    const title = sessionTitleRef.current.trim();
    if (title) params.set("title", title.slice(0, 120));
    params.set("fps", String(analysisFps));
    params.set("mode", isPeriodic ? "periodic" : "live");
    params.set("interval", String(Math.round(intervalMs / 1000)));
    params.set("motion", motionEnabled ? "1" : "0");
    if (xaiModeRef.current === "transition") {
      params.set("xai_interval", String(NO_INTERVAL));
    }
    return `${WS_URL}/ws/emotion-stream?${params.toString()}`;
  }, [token, analysisFps, isPeriodic, intervalMs, motionEnabled]);

  const openWs = useCallback(() => {
    intentionalCloseRef.current = false;
    setWsStatus(retryRef.current > 0 ? "reconnecting" : "connecting");
    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        handleMessage(JSON.parse(event.data) as WsServerMessage);
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onopen = () => {
      retryRef.current = 0;
      setWsStatus("open");
      setError(null);
      if (!pausedRef.current) startSendTimer();
    };
    ws.onclose = (event) => {
      stopSendTimer();
      if (intentionalCloseRef.current) {
        setWsStatus("closed");
        return;
      }
      if (event.code === 4001) {
        setError("WebSocket auth failed — please sign in again.");
        setWsStatus("closed");
        return;
      }
      if (event.code === 1013) {
        setError("Too many live connections for this account.");
        setWsStatus("closed");
        return;
      }
      // auto-reconnect with backoff (keeps the camera running)
      if (retryRef.current < 5) {
        const attempt = retryRef.current;
        retryRef.current += 1;
        setWsStatus("reconnecting");
        window.setTimeout(() => {
          if (wsRef.current === ws && !intentionalCloseRef.current) {
            openWs();
          }
        }, Math.min(8000, 1000 * 2 ** attempt));
      } else {
        setWsStatus("closed");
        setError("Connection lost — press Start to reconnect.");
      }
    };
    ws.onerror = () => {
      if (retryRef.current === 0) {
        setError("WebSocket error — is the API running?");
      }
    };
  }, [wsUrl, handleMessage, startSendTimer, stopSendTimer]);

  /* ---------- start / stop ---------- */
  const teardown = useCallback(() => {
    intentionalCloseRef.current = true;
    retryRef.current = 0;
    stopSendTimer();
    stopPreviewLoop();
    stopMotionWatch();
    lastAnalysisAtRef.current = null;
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
    const stream = streamRef.current;
    streamRef.current = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    const video = videoRef.current;
    if (video) video.srcObject = null;
    overlayRef.current?.getContext("2d")?.clearRect(0, 0, CAPTURE_W, CAPTURE_H);
    verdictBufRef.current.reset();
    verdictBoxRef.current = null;
    lastFaceAtRef.current = null;
    instantFlushRef.current = false;
    setVerdict(null);
    setVerdictSamples(0);
    setVerdictStale(false);
    setRunning(false);
    setPaused(false);
    setFaces([]);
    setFps(null);
    setWsStatus("closed");
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, [stopSendTimer, stopPreviewLoop, stopMotionWatch]);

  const start = useCallback(async () => {
    if (!token) return;
    setError(null);
    setTimeline([]);
    setNarration([]);
    setXai(null);
    setSessionId(null);
    setEventCount(0);
    verdictBufRef.current.reset();
    verdictBoxRef.current = null;
    lastFaceAtRef.current = null;
    instantFlushRef.current = false;
    setVerdict(null);
    setVerdictSamples(0);
    setVerdictStale(false);
    retryRef.current = 0;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("camera element missing");
      video.srcObject = stream;
      await video.play();
      startPreviewLoop();
      if (isPeriodic && motionEnabled) startMotionWatch();
      openWs();
      setRunning(true);
    } catch (err) {
      const message =
        err instanceof Error && err.name === "NotAllowedError"
          ? "Camera permission denied."
          : `Could not start camera: ${err instanceof Error ? err.message : String(err)}`;
      setError(message);
      teardown();
    }
  }, [token, openWs, teardown, startPreviewLoop, startMotionWatch, isPeriodic, motionEnabled]);

  const stop = useCallback(() => {
    teardown();
  }, [teardown]);

  const togglePause = useCallback(() => {
    if (!running) return;
    setPaused((p) => {
      const next = !p;
      pausedRef.current = next;
      if (next) {
        stopSendTimer();
        toast.info("Paused — camera stays on, no frames are sent.");
      } else {
        if (wsRef.current?.readyState === WebSocket.OPEN) startSendTimer();
        toast.info("Resumed.");
      }
      return next;
    });
  }, [running, startSendTimer, stopSendTimer]);

  const snapshot = useCallback(() => {
    const display = displayRef.current;
    const overlay = overlayRef.current;
    if (!display) return;
    const out = document.createElement("canvas");
    out.width = CAPTURE_W;
    out.height = CAPTURE_H;
    const ctx = out.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(display, 0, 0);
    if (overlay) ctx.drawImage(overlay, 0, 0);
    const a = document.createElement("a");
    a.href = out.toDataURL("image/png");
    a.download = `hfd_snapshot_${Date.now()}.png`;
    a.click();
    toast.success("Snapshot saved.");
  }, []);

  /* ---------- shortcuts ---------- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (typing) return;
      if (e.key === "Escape") {
        setHelpOpen(false);
        return;
      }
      if (e.key === "?" || (e.key === "/" && e.shiftKey)) {
        e.preventDefault();
        setHelpOpen((v) => !v);
        return;
      }
      if (e.key === " " && running) {
        e.preventDefault();
        togglePause();
        return;
      }
      if (e.key.toLowerCase() === "m") setMirror((v) => !v);
      if (e.key.toLowerCase() === "h") setShowHeatmap((v) => !v);
      if (e.key.toLowerCase() === "s" && running) snapshot();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [running, togglePause, snapshot]);

  useEffect(() => teardown, [teardown]);

  /* ---------- demo hook (?demo=verdict): synthetic samples for review ---------- */
  useEffect(() => {
    if (typeof window === "undefined" || !running || !demoModeRef.current) {
      return;
    }
    const id = window.setInterval(() => {
      const t = Date.now() / 1000;
      const swing = (Math.sin(t / 7) + 1) / 2;
      const happy = 0.2 + swing * 0.65;
      const rest = 1 - happy;
      const probs: Record<string, number> = {
        happy,
        neutral: rest * 0.45,
        surprise: rest * 0.25,
        sad: rest * 0.12,
        angry: rest * 0.08,
        fear: rest * 0.06,
        disgust: rest * 0.04,
      };
      const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
      const box: Box = [200 + Math.sin(t / 5) * 24, 105, 235, 285];
      const regions: Record<string, number> = {
        mouth: 35 + swing * 30,
        cheeks: 18 + swing * 12,
        eyes: 22 - swing * 8,
        eyebrows: 12,
        nose: 6,
      };
      lastFaceAtRef.current = Date.now();
      verdictBufRef.current.push({
        faceId: 0,
        ts: Date.now(),
        box,
        probabilities: probs,
        regions,
      });
      setVerdictSamples(verdictBufRef.current.size);
      setFaces([
        {
          face_id: 0,
          box,
          emotion: top,
          confidence: probs[top],
          probabilities: probs,
          ranked_emotions: Object.keys(probs).sort((a, b) => probs[b] - probs[a]),
          evidence_scores: {},
          certainty: "certain",
          margin: 0.2,
          intensity: 55,
          intensity_category: "medium",
          action_units: null,
          au_model_status: "untrained",
          regions,
        },
      ]);
    }, 450);
    return () => window.clearInterval(id);
  }, [running]);

  /* ---------- controls ---------- */
  const sendControl = useCallback(
    (faceId: number) => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "xai", face_id: faceId }));
        setShowHeatmap(true);
      } else {
        setError("Start the stream first to request an explanation.");
      }
    },
    [],
  );

  const selected =
    faces.find((f) => f.face_id === selectedId) ?? faces[0] ?? null;

  const statusLabel =
    wsStatus === "open"
      ? paused
        ? "Paused"
        : isPeriodic
          ? "Verdict"
          : "Live"
      : wsStatus === "connecting"
        ? "Connecting…"
        : wsStatus === "reconnecting"
          ? "Reconnecting…"
          : "Offline";

  return (
    <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-3">
      <h1 className="sr-only">Live Observer</h1>
      {/* hidden helpers */}
      <video ref={videoRef} className="hidden" playsInline muted autoPlay aria-hidden />
      <canvas ref={sendRef} width={CAPTURE_W} height={CAPTURE_H} className="hidden" />

      {/* ---- camera + timeline ---- */}
      <div className="space-y-4 lg:col-span-2">
        <Card
          title="Live camera feed"
          action={
            <span className="flex items-center gap-2 text-xs text-muted">
              <span
                className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 ${
                  wsStatus === "open" && !paused
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                    : wsStatus === "reconnecting" || wsStatus === "connecting"
                      ? "border-amber-500/40 bg-amber-500/10 text-amber-400"
                      : "border-border bg-background"
                }`}
                role="status"
              >
                <span
                  aria-hidden
                  className={`h-1.5 w-1.5 rounded-full ${
                    wsStatus === "open" && !paused
                      ? "bg-emerald-400"
                      : wsStatus === "reconnecting" || wsStatus === "connecting"
                        ? "bg-amber-400 animate-pulse"
                        : "bg-muted"
                  }`}
                />
                {statusLabel}
              </span>
              {running &&
                (isPeriodic
                  ? `verdict mode · next ${nextCheckSec !== null ? formatDuration(nextCheckSec) : "—"} · window ${verdictSamples} frames${verdictStale ? " · face lost (verdict held)" : " · tracking"} · frame ${frameId ?? 0} · ${eventCount} events${motionEnabled ? ` · motion ${(motionLevel * 100).toFixed(1)}%` : ""}`
                  : `${fps ?? "—"} fps · set ${analysisFps} fps · frame ${frameId ?? 0} · ${eventCount} events`)}
              {!running && "idle"}
            </span>
          }
        >
          <div className="relative overflow-hidden rounded-md bg-black">
            <canvas
              ref={displayRef}
              width={CAPTURE_W}
              height={CAPTURE_H}
              className="block w-full"
              aria-label="Camera frame"
            />
            <canvas
              ref={overlayRef}
              width={CAPTURE_W}
              height={CAPTURE_H}
              className="absolute inset-0 h-full w-full"
              aria-hidden
            />
            {!running && (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">
                Press Start to open the camera
              </div>
            )}
            {paused && running && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-sm font-medium">
                Paused — press Space or Resume
              </div>
            )}
          </div>

          {error && (
            <p role="alert" className="mt-2 text-sm text-red-400">
              {error}
            </p>
          )}

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {!running ? (
              <Button onClick={start}>Start</Button>
            ) : (
              <>
                <Button variant="ghost" onClick={togglePause}>
                  {paused ? "Resume" : "Pause"}
                </Button>
                <Button variant="danger" onClick={stop}>
                  Stop
                </Button>
              </>
            )}
            <label className="flex items-center gap-1.5 text-xs text-muted">
              Session title
              <input
                type="text"
                value={sessionTitle}
                maxLength={120}
                placeholder="e.g. Team standup"
                onChange={(e) => setSessionTitle(e.target.value)}
                className="w-40 rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground"
                title="Saved as this session's name in Analytics (set before Start)"
              />
            </label>
            <label className="flex items-center gap-1.5 text-xs text-muted">
              Explain
              <select
                value={xaiMode}
                onChange={(e) =>
                  setXaiMode(e.target.value as "interval" | "transition")
                }
                className="rounded-md border border-border bg-background px-2 py-1 text-sm"
                title="When to compute Grad-CAM explanations"
              >
                <option value="interval">every N frames</option>
                <option value="transition">on transitions only</option>
              </select>
            </label>
            <Button
              variant="ghost"
              aria-pressed={voiceOn}
              onClick={() => setVoiceOn((v) => !v)}
            >
              Voice {voiceOn ? "on" : "off"}
            </Button>
            <Button
              variant="ghost"
              aria-pressed={mirror}
              onClick={() => setMirror((v) => !v)}
            >
              Mirror {mirror ? "on" : "off"}
            </Button>
            <Button variant="ghost" disabled={!running} onClick={snapshot}>
              Snapshot
            </Button>
            <Button
              variant="ghost"
              onClick={() => setHelpOpen(true)}
              title="Keyboard shortcuts (?)"
            >
              ? Help
            </Button>
            {sessionId && (
              <Button
                variant="ghost"
                onClick={() => router.push(`/analytics?session=${sessionId}`)}
              >
                Open session analytics
              </Button>
            )}
          </div>
          <p className="mt-2 text-[11px] text-muted">
            Title and explain-mode apply when you press Start (they configure
            the session). Frames are processed in-memory and never stored.
          </p>
        </Card>

        <Card title="Timeline">
          <EmotionTimeline samples={timeline} />
        </Card>

        <Card title="Narrator" action={
          <span className="text-xs text-muted">{narration.length} lines</span>
        }>
          <div aria-live="polite" aria-atomic="false">
            <CommentaryPanel lines={narration} />
          </div>
        </Card>
      </div>

      {/* ---- person panel ---- */}
      <div className="space-y-4">
        <Card title="Person">
          {faces.length === 0 ? (
            <p className="text-xs text-muted">
              Face readings appear here once the stream is running.
            </p>
          ) : (
            <div className="space-y-2">
              {faces.map((face) => (
                <PersonCard
                  key={face.face_id}
                  face={face}
                  selected={face.face_id === (selected?.face_id ?? -1)}
                  badge={isPeriodic ? "live frame" : undefined}
                  onSelect={() => setSelectedId(face.face_id)}
                  onWhy={() => sendControl(face.face_id)}
                  onHeatmap={() => {
                    setSelectedId(face.face_id);
                    sendControl(face.face_id);
                  }}
                />
              ))}
            </div>
          )}
        </Card>

        {selected && (
          <Card title="Ranking & evidence">
            <EmotionSpectrum
              probabilities={selected.probabilities}
              ranked={selected.ranked_emotions}
              evidenceScores={selected.evidence_scores}
            />
            <p className="mt-3 text-[11px] text-muted">
              Bars: model probability. White tick: ranking evidence score. AU
              outputs omitted (AU model untrained).
            </p>
          </Card>
        )}

        {(selected?.regions || xai) && (
          <Card title="Region evidence (Grad-CAM)">
            <RegionChart regions={xai?.regions ?? selected?.regions ?? null} />
          </Card>
        )}

        {showHeatmap && xai && (
          <Card
            title="Heatmap"
            action={
              <button
                type="button"
                className="text-xs text-muted hover:text-foreground"
                onClick={() => setShowHeatmap(false)}
              >
                hide
              </button>
            }
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${xai.heatmap}`}
              alt={`Grad-CAM heatmap for ${xai.emotion}`}
              className="mx-auto max-h-56 rounded border border-border"
            />
            <p className="mt-2 text-center text-xs text-muted">
              {EMOTION_LABEL[xai.emotion] ?? xai.emotion} ·{" "}
              {pct(xai.confidence)}
            </p>
          </Card>
        )}
      </div>

      {/* shortcuts help */}
      {helpOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Keyboard shortcuts"
          onClick={() => setHelpOpen(false)}
        >
          <div
            className="w-full max-w-sm rounded-lg border border-border bg-surface p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
                Keyboard shortcuts
              </h2>
              <button
                type="button"
                aria-label="Close"
                onClick={() => setHelpOpen(false)}
                className="text-muted hover:text-foreground"
              >
                ✕
              </button>
            </div>
            <dl className="space-y-2 text-sm">
              {SHORTCUTS.map(([key, desc]) => (
                <div key={key} className="flex items-center justify-between gap-4">
                  <dt>
                    <kbd className="rounded border border-border bg-background px-1.5 py-0.5 font-mono text-xs">
                      {key}
                    </kbd>
                  </dt>
                  <dd className="text-right text-muted">{desc}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
