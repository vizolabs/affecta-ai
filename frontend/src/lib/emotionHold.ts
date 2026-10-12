"use client";

import { useEffect, useRef, useState } from "react";
import type { FaceAnalysis } from "./types";

export interface HeldFace extends FaceAnalysis {
  /** True while the displayed emotion is being held after a change. */
  holding: boolean;
  /** Seconds left on the hold (0 when not holding). */
  holdRemaining: number;
  /** 0..1 progress of the hold countdown (1 = just locked). */
  holdProgress: number;
}

/**
 * Countdown resolution: fine enough to animate a 3s hold, cheap enough that a
 * 2-minute hold does not spin the timer ~1200 times.
 */
function tickFor(holdMs: number): number {
  return Math.min(1000, Math.max(100, Math.round(holdMs / 60)));
}

/**
 * Display dwell for the dominant face.
 *
 * When the dominant emotion changes, the previously displayed value stays on
 * screen for `holdMs` before the live one takes over. This removes per-frame
 * label flicker for viewers without touching inference, events, or narration
 * (those stay live and truthful).
 *
 * `holdMs <= 0` disables holding entirely — the live value is returned as-is.
 */
export function useEmotionHold(
  faces: FaceAnalysis[],
  holdMs: number,
): HeldFace | null {
  const [version, setVersion] = useState(0);
  // Face actually painted on screen: may lag the live one while holding.
  const shownRef = useRef<FaceAnalysis | null>(null);
  const lockedUntilRef = useRef(0);
  const holdMsRef = useRef(holdMs);

  holdMsRef.current = holdMs;

  // Face identity can change (subject leaves / a new one enters).
  const liveId = faces.length > 0 ? faces[0].face_id : null;

  const live = faces[0] ?? null;

  // Decide whether the on-screen value should stay locked.
  useEffect(() => {
    const now = Date.now();
    if (!live) {
      shownRef.current = null;
      lockedUntilRef.current = 0;
      return;
    }
    const shown = shownRef.current;
    if (!shown || shown.face_id !== live.face_id) {
      // New subject: show it immediately.
      shownRef.current = live;
      lockedUntilRef.current = 0;
      return;
    }
    if (shown.emotion === live.emotion) {
      // Same emotion: keep refreshing the live numbers, no hold.
      shownRef.current = live;
      return;
    }
    // Emotion changed. If nothing is being held yet, start holding the old one.
    if (lockedUntilRef.current === 0) {
      if (holdMsRef.current > 0) {
        lockedUntilRef.current = now + holdMsRef.current;
      } else {
        shownRef.current = live;
        lockedUntilRef.current = 0;
      }
    }
  }, [live, liveId, version]);

  // Countdown ticker while a hold is active.
  useEffect(() => {
    if (lockedUntilRef.current === 0) return;
    const id = window.setInterval(() => {
      if (Date.now() >= lockedUntilRef.current) {
        window.clearInterval(id);
        lockedUntilRef.current = 0;
      }
      setVersion((v) => v + 1);
    }, tickFor(holdMsRef.current));
    return () => window.clearInterval(id);
  }, [version, liveId]);

  if (!live) return null;

  const shown = shownRef.current;
  const now = Date.now();

  if (lockedUntilRef.current > 0 && now < lockedUntilRef.current && shown) {
    const remainingMs = lockedUntilRef.current - now;
    const hold = Math.max(1, holdMsRef.current);
    return {
      ...shown,
      holding: true,
      holdRemaining: remainingMs / 1000,
      holdProgress: Math.min(1, Math.max(0, remainingMs / hold)),
    };
  }

  // Hold expired (or disabled): fall through to the live value.
  shownRef.current = live;
  return { ...live, holding: false, holdRemaining: 0, holdProgress: 0 };
}