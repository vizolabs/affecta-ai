"use client";

import { useEffect, useState, type FormEvent } from "react";
import Button from "../../../components/common/Button";
import Card from "../../../components/common/Card";
import InfoTip from "../../../components/common/InfoTip";
import { apiFetch, ApiError } from "../../../lib/api";
import { formatDuration, formatHold } from "../../../lib/cadence";
import type { UserSettings } from "../../../lib/types";
import { useAuth } from "../../../stores/auth";
import { useTheme } from "../../../stores/theme";
import { toast } from "../../../stores/toast";
import { usePageTitle } from "../../../hooks/usePageTitle";

/** Check-in cadence presets in seconds (30s … 5 min). */
const INTERVAL_PRESETS = [30, 60, 90, 120, 180, 300];

function speakSample(text: string, rate: number, voiceURI: string): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = rate;
  const voice = window.speechSynthesis.getVoices().find((v) => v.voiceURI === voiceURI);
  if (voice) u.voice = voice;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

export default function SettingsPage() {
  usePageTitle("Settings");
  const token = useAuth((s) => s.token);
  const user = useAuth((s) => s.user);
  const setTheme = useTheme((s) => s.setTheme);

  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [armedDelete, setArmedDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiFetch<UserSettings>("/api/settings", {}, token)
      .then((data) => {
        if (!cancelled) setSettings(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const load = () => setVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, []);

  useEffect(() => {
    if (!armedDelete) return;
    const id = window.setTimeout(() => setArmedDelete(false), 3000);
    return () => window.clearTimeout(id);
  }, [armedDelete]);

  const update = <K extends keyof UserSettings>(
    key: K,
    value: UserSettings[K],
  ) => {
    setSaved(false);
    setSettings((prev) => (prev ? { ...prev, [key]: value } : prev));
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!settings) return;
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      const data = await apiFetch<UserSettings>(
        "/api/settings",
        { method: "PUT", body: JSON.stringify(settings) },
        token,
      );
      setSettings(data);
      if (data.theme === "dark" || data.theme === "light") {
        setTheme(data.theme);
      }
      setSaved(true);
      toast.success("Settings saved.");
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Save failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  };

  const deleteAccount = async () => {
    if (!armedDelete) {
      setArmedDelete(true);
      toast.info("Click again within 3s to permanently delete this account.");
      return;
    }
    try {
      await apiFetch("/api/auth/me", { method: "DELETE" }, token);
      toast.success("Account deleted.");
      const { clear } = useAuth.getState();
      clear();
      window.location.href = "/login";
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Delete failed");
      setArmedDelete(false);
    }
  };

  if (!settings) {
    return (
      <Card title="Settings" className="mx-auto max-w-xl">
        <p className="text-sm text-muted">
          {error ?? "Loading settings…"}
        </p>
      </Card>
    );
  }

  return (
    <form onSubmit={submit} className="mx-auto max-w-xl space-y-4">
      <Card title="Account">
        <p className="text-sm">{user?.email ?? "—"}</p>
        <p className="text-xs text-muted">
          Accounts and sessions are stored only in the local SQLite database.
        </p>
      </Card>

      <Card title="Appearance">
        <label className="flex items-center justify-between text-sm">
          <span>Theme</span>
          <select
            value={settings.theme}
            onChange={(e) => update("theme", e.target.value)}
            className="rounded-md border border-border bg-background px-2 py-1.5"
          >
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </label>
      </Card>

      <Card title="Narrator">
        <label className="mb-3 flex items-center justify-between text-sm">
          <span>Enabled (server-side)</span>
          <input
            type="checkbox"
            checked={settings.narrator_enabled}
            onChange={(e) => update("narrator_enabled", e.target.checked)}
            className="h-4 w-4 accent-[var(--accent)]"
          />
        </label>
        <label className="mb-3 flex items-center justify-between text-sm">
          <span>
            Detail level
            <InfoTip text="Calm: enter/exit/changes only. Normal: adds intensity + ambiguity. Detailed: also speaks confidence drops." />
          </span>
          <select
            value={settings.narrator_detail}
            onChange={(e) => update("narrator_detail", e.target.value)}
            className="rounded-md border border-border bg-background px-2 py-1.5"
          >
            <option value="calm">Calm</option>
            <option value="normal">Normal</option>
            <option value="detailed">Detailed</option>
          </select>
        </label>
        <label className="mb-3 flex items-center justify-between text-sm">
          <span>Speed ×{settings.narrator_speed.toFixed(1)}</span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.1}
            value={settings.narrator_speed}
            onChange={(e) => update("narrator_speed", Number(e.target.value))}
            className="w-40"
          />
        </label>
        <div className="flex items-center justify-between gap-2 text-sm">
          <span>Voice</span>
          <div className="flex items-center gap-2">
            <select
              value={settings.narrator_voice}
              onChange={(e) => update("narrator_voice", e.target.value)}
              className="max-w-48 rounded-md border border-border bg-background px-2 py-1.5"
            >
              <option value="default">Default</option>
              {voices.map((v) => (
                <option key={v.voiceURI} value={v.voiceURI}>
                  {v.name}
                </option>
              ))}
            </select>
            <Button
              type="button"
              variant="ghost"
              onClick={() =>
                speakSample(
                  "Person 01 changed from Neutral to Happy.",
                  settings.narrator_speed,
                  settings.narrator_voice,
                )
              }
            >
              Test
            </Button>
          </div>
        </div>
        <p className="mt-2 text-xs text-muted">
          Voice uses your browser&apos;s built-in speech synthesis — nothing is
          sent to a server.
        </p>
      </Card>

      <Card title="Analysis smoothing">
        <label className="flex items-center justify-between text-sm">
          <span>
            EMA smoothing α{" "}
            <span className="text-muted">({settings.ema_alpha.toFixed(2)})</span>
            <InfoTip text="Lower α = calmer probabilities (more smoothing). Higher α = more responsive, more flicker." />
          </span>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={settings.ema_alpha}
            onChange={(e) => update("ema_alpha", Number(e.target.value))}
            className="w-40"
          />
        </label>
        <p className="mt-2 text-xs text-muted">
          Applies to live probability smoothing. New sessions pick this up on
          connect.
        </p>
      </Card>

      <Card title="Capture &amp; display">
        <label className="flex items-center justify-between text-sm">
          <span>
            Analysis mode{" "}
            <span className="uppercase text-accent">
              {settings.analysis_mode}
            </span>
            <InfoTip text="Live = continuous per-frame analysis. Periodic = a pinned verdict every N minutes, mean-pooled from a continuous stream of readings — built for unattended displays." />
          </span>
          <select
            value={settings.analysis_mode}
            onChange={(e) =>
              update("analysis_mode", e.target.value as UserSettings["analysis_mode"])
            }
            className="rounded-md border border-border bg-background px-2 py-1.5 text-sm"
          >
            <option value="live">Live (continuous)</option>
            <option value="periodic">Periodic (verdict)</option>
          </select>
        </label>

        {settings.analysis_mode === "live" ? (
          <>
            <label className="mt-4 flex items-center justify-between text-sm">
              <span>
                Analysis frame rate{" "}
                <span className="text-muted">({settings.analysis_fps} fps)</span>
                <InfoTip text="Frames per second captured from the camera and analysed. Lower = less CPU/heat and steadier numbers; higher = more responsive." />
              </span>
              <input
                type="range"
                min={1}
                max={15}
                step={1}
                value={settings.analysis_fps}
                onChange={(e) =>
                  update("analysis_fps", Math.min(15, Math.max(1, Number(e.target.value))))
                }
                className="w-40"
              />
            </label>
          </>
        ) : (
          <>
            <div className="mt-4">
              <div className="flex items-center justify-between text-sm">
                <span>
                  Verdict every{" "}
                  <span className="text-muted">
                    ({formatDuration(settings.analysis_interval_seconds)})
                  </span>
                  <InfoTip text="Frames stream continuously at your analysis rate; every window the readings are mean-pooled into one pinned verdict on the face. Range: 30s – 10 min." />
                </span>
              </div>
              <input
                type="range"
                min={30}
                max={600}
                step={10}
                value={settings.analysis_interval_seconds}
                onChange={(e) =>
                  update(
                    "analysis_interval_seconds",
                    Math.min(600, Math.max(30, Number(e.target.value))),
                  )
                }
                className="mt-2 w-full"
              />
              <div className="mt-2 flex flex-wrap gap-1.5">
                {INTERVAL_PRESETS.map((sec) => (
                  <button
                    key={sec}
                    type="button"
                    onClick={() => update("analysis_interval_seconds", sec)}
                    className={`rounded-md border px-2 py-1 font-mono text-[11px] transition ${
                      settings.analysis_interval_seconds === sec
                        ? "border-accent bg-accent/15 text-accent"
                        : "border-border text-muted hover:border-accent/60 hover:text-foreground"
                    }`}
                  >
                    {formatDuration(sec)}
                  </button>
                ))}
              </div>
            </div>

            <label className="mt-4 flex items-center justify-between text-sm">
              <span>
                Re-analyse on movement
                <InfoTip text="A cheap frame-difference check (no model inference). Significant movement flushes the current window into an instant verdict; a 5s cooldown prevents repeated triggers." />
              </span>
              <input
                type="checkbox"
                checked={settings.motion_trigger_enabled}
                onChange={(e) =>
                  update("motion_trigger_enabled", e.target.checked)
                }
              />
            </label>
          </>
        )}

        <label className="mt-4 flex items-center justify-between text-sm">
          <span>
            Emotion hold{" "}
            <span className="text-muted">
              ({formatHold(settings.emotion_hold_seconds)})
            </span>
            <InfoTip text="When a new emotion is detected, the previous label stays on screen for this long before switching. Off = switch immediately. Readings that arrive during a hold are not replayed late — only the newest is shown." />
          </span>
          <input
            type="range"
            min={0}
            max={600}
            step={5}
            value={settings.emotion_hold_seconds}
            onChange={(e) =>
              update(
                "emotion_hold_seconds",
                Math.min(600, Math.max(0, Number(e.target.value))),
              )
            }
            className="w-40"
          />
        </label>

        <p className="mt-2 text-xs text-muted">
          {settings.analysis_mode === "periodic"
            ? "Periodic mode: frames stream continuously; the face shows the pinned verdict for the current window (pooled from every reading in that window), and the Live bar counts down to the next one. The Person panel and timeline stay live per-frame."
            : "Frame rate applies immediately to a running session; the hold takes effect on the next analysis frame. Held labels are marked with a countdown bar — narration and the Person panel always show live values."}
        </p>
      </Card>

      <Card title="XAI">
        <label className="flex items-center justify-between text-sm">
          <span>
            Grad-CAM every N frames{" "}
            <span className="text-muted">({settings.xai_frame_interval})</span>
          </span>
          <input
            type="number"
            min={1}
            max={120}
            value={settings.xai_frame_interval}
            onChange={(e) =>
              update("xai_frame_interval", Math.max(1, Number(e.target.value)))
            }
            className="w-20 rounded-md border border-border bg-background px-2 py-1.5"
          />
        </label>
        <p className="mt-2 text-xs text-muted">
          Lower values compute explanations more often (higher CPU load). Live
          page can also run in explain-on-transition mode.
        </p>
      </Card>

      <Card title="Danger zone">
        <p className="mb-2 text-xs text-muted">
          Deleting your account removes all local sessions, events, XAI
          snapshots, and settings. This cannot be undone.
        </p>
        <Button
          type="button"
          variant="danger"
          onClick={deleteAccount}
          className={armedDelete ? "ring-2 ring-red-400" : ""}
        >
          {armedDelete ? "Confirm delete — click again" : "Delete account"}
        </Button>
      </Card>

      {error && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}
      {saved && !error && (
        <p className="text-sm text-emerald-400" role="status">
          Settings saved.
        </p>
      )}

      <Button type="submit" disabled={busy}>
        {busy ? "Saving…" : "Save settings"}
      </Button>
    </form>
  );
}
