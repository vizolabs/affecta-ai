export interface Envelope<T> {
  status: "success" | "error";
  data: T;
  meta: unknown;
  error: string | null;
}

export interface User {
  id: string;
  email: string;
  full_name?: string | null;
  created_at: string;
}

export interface TokenOut {
  token: string;
  token_type: string;
  user: User;
}

export type Certainty = "certain" | "uncertain" | "ambiguous" | string;

export interface FaceAnalysis {
  face_id: number;
  box: [number, number, number, number];
  emotion: string;
  confidence: number; // 0..1
  probabilities: Record<string, number>;
  ranked_emotions: string[];
  evidence_scores: Record<string, number>;
  certainty: Certainty;
  margin: number;
  intensity: number; // 0..100
  intensity_category: "low" | "medium" | "high" | string;
  action_units: Record<string, number> | null;
  au_model_status: string;
  regions: Record<string, number> | null;
}

export interface StreamEvent {
  event_type: string;
  face_id: number;
  timestamp: number;
  from_emotion?: string | null;
  to_emotion?: string | null;
  confidence?: number | null;
  intensity?: number | null;
  certainty?: string | null;
}

export interface NarrationLine {
  face_id: number;
  event_type: string;
  text: string;
  timestamp: number;
}

export interface XaiPayload {
  face_id: number;
  method: string;
  emotion: string;
  confidence: number;
  heatmap: string; // base64 PNG (no prefix)
  regions: Record<string, number>;
}

export interface TimelineSample {
  t: number;
  faces: {
    id: number;
    emotion: string;
    intensity: number;
    confidence: number;
  }[];
}

export interface AnalysisOut {
  session_id?: string | null;
  frame_id?: number;
  faces: FaceAnalysis[];
  events: StreamEvent[];
  narration: NarrationLine[];
  xai: XaiPayload | null;
  fps?: number;
  model?: string;
  timeline?: TimelineSample | null;
  timestamp?: number;
}

/** Offline video run: aggregate over every sampled frame of the clip. */
export interface VideoAnalysisOut {
  kind: "video";
  model: string;
  frames_analyzed: number;
  native_fps: number;
  sample_fps: number;
  sample_every_n_frames: number;
  total_frames: number | null;
  duration_seconds: number;
  faces_per_frame_avg: number;
  frames_with_face: number;
  frames_without_face: number;
  /** Share of sampled frames where a face was detected (0..1). */
  face_coverage: number;
  emotion_distribution: { emotion: string; count: number; share: number }[];
  dominant_emotion: string | null;
  avg_confidence: number | null;
  avg_intensity: number | null;
  transitions: {
    t: number;
    from_emotion: string | null;
    to_emotion: string | null;
    confidence: number | null;
  }[];
  transition_count: number;
  timeline: TimelineSample[];
  narration: NarrationLine[];
  peak_frame: {
    index: number;
    t: number;
    emotion: string;
    confidence: number;
    intensity: number;
    certainty: string;
    probabilities: Record<string, number>;
    ranked_emotions: string[];
    evidence_scores: Record<string, number>;
  } | null;
  xai: XaiPayload | null;
}

export type WsServerMessage =
  | ({ type: "session" } & { session_id: string })
  | ({ type: "analysis" } & AnalysisOut)
  | ({ type: "xai" } & XaiPayload)
  | { type: "error"; error: string }
  | { type: "pong" };

export interface SessionOut {
  id: string;
  start_time: string;
  end_time?: string | null;
  duration_seconds?: number | null;
  total_frames: number;
  total_events: number;
  avg_confidence?: number | null;
  avg_intensity?: number | null;
  metadata?: Record<string, unknown> | null;
}

export interface EventOut {
  id: string;
  session_id?: string | null;
  face_id: number;
  timestamp: string;
  event_type: string;
  from_emotion?: string | null;
  to_emotion?: string | null;
  confidence?: number | null;
  intensity?: number | null;
  certainty?: string | null;
}

export interface SessionSummary {
  session_id: string;
  duration_seconds?: number | null;
  total_frames: number;
  total_events: number;
  avg_confidence?: number | null;
  avg_intensity?: number | null;
  model?: string | null;
  expression_distribution: Record<string, number>;
  event_type_counts: Record<string, number>;
  transition_count: number;
  transition_matrix: Record<string, number>;
  uncertainty_events: number;
  per_person: Record<
    string,
    { samples: number; emotions: Record<string, number>; dominant: string }
  >;
}

export interface AnalyticsTimeline {
  session_id: string;
  timeline: TimelineSample[];
  events: EventOut[];
}

export interface UserSettings {
  theme: string;
  narrator_enabled: boolean;
  narrator_speed: number;
  narrator_voice: string;
  narrator_detail: string;
  ema_alpha: number;
  xai_frame_interval: number;
  /** Frames analysed per second upstream (1-15). */
  analysis_fps: number;
  /** Display dwell: seconds a detected emotion stays on screen (0 = off). */
  emotion_hold_seconds: number;
  /** "live" = continuous capture; "periodic" = one analysis every N seconds. */
  analysis_mode: "live" | "periodic";
  analysis_interval_seconds: number;
  /** Periodic mode: significant movement forces an early re-analysis. */
  motion_trigger_enabled: boolean;
}

export interface HealthInfo {
  status: string;
  app?: string;
  version?: string;
  model?: string;
  model_loaded?: boolean;
}

export interface ModelInfo {
  model_name: string;
  version: string;
  model_type: string;
  dataset_used: string;
  metrics: Record<string, number> | null;
  checkpoint_path: string;
  status: string;
  created_at: string | null;
}
