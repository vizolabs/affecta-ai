"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Button from "../../../components/common/Button";
import Card from "../../../components/common/Card";
import InfoTip from "../../../components/common/InfoTip";
import Skeleton from "../../../components/common/Skeleton";
import { apiBlob, apiFetch, ApiError, saveBlob } from "../../../lib/api";
import {
  EMOTIONS,
  EMOTION_LABEL,
  emotionColor,
  personLabel,
  pct,
} from "../../../lib/emotions";
import { formatSessionDuration, sessionLabel, slug } from "../../../lib/sessions";
import type {
  AnalyticsTimeline,
  SessionOut,
  SessionSummary,
} from "../../../lib/types";
import { useAuth } from "../../../stores/auth";
import { toast } from "../../../stores/toast";
import { usePageTitle } from "../../../hooks/usePageTitle";

/** Proxy valence — directional cue only, NOT a clinical measurement. */
const VALENCE: Record<string, number> = {
  happy: 0.7,
  surprise: 0.2,
  neutral: 0,
  sad: -0.3,
  fear: -0.4,
  disgust: -0.5,
  angry: -0.6,
};

function isoDay(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function dateStamp(d: Date): string {
  return d.toISOString().slice(0, 10).replace(/-/g, "");
}

function TransitionMatrix({ matrix }: { matrix: Record<string, number> }) {
  const pairs = Object.entries(matrix);
  if (pairs.length === 0) {
    return <p className="text-sm text-muted">No transitions recorded.</p>;
  }
  const max = Math.max(...pairs.map(([, n]) => n));
  const grid = new Map<string, number>();
  for (const [pair, n] of pairs) {
    const [from, to] = pair.split("->");
    if (from && to) grid.set(`${from}|${to}`, n);
  }
  return (
    <div>
      <div
        className="grid gap-0.5 overflow-x-auto text-[10px]"
        style={{
          gridTemplateColumns: `64px repeat(${EMOTIONS.length}, minmax(28px, 1fr))`,
        }}
        role="table"
        aria-label="Emotion transition matrix"
      >
        <span aria-hidden />
        {EMOTIONS.map((to) => (
          <span
            key={to}
            className="pb-1 text-center font-medium text-muted"
            title={EMOTION_LABEL[to]}
          >
            {EMOTION_LABEL[to].slice(0, 3)}
          </span>
        ))}
        {EMOTIONS.map((from) => (
          <div key={from} className="contents" role="row">
            <span
              className="flex items-center pr-1 text-right font-medium text-muted"
              title={EMOTION_LABEL[from]}
            >
              {EMOTION_LABEL[from]}
            </span>
            {EMOTIONS.map((to) => {
              const n = grid.get(`${from}|${to}`) ?? 0;
              const intensity = n > 0 ? 0.15 + 0.85 * (n / max) : 0;
              return (
                <span
                  key={to}
                  title={`${EMOTION_LABEL[from]} → ${EMOTION_LABEL[to]}: ${n}`}
                  className={`flex h-7 items-center justify-center rounded-sm tabular-nums ${
                    n === 0 ? "bg-border/30 text-muted/50" : ""
                  }`}
                  style={
                    n > 0
                      ? {
                          backgroundColor: `color-mix(in srgb, ${emotionColor(to)} ${intensity * 100}%, transparent)`,
                          color: intensity > 0.5 ? "#fff" : "var(--text)",
                        }
                      : undefined
                  }
                >
                  {n || "·"}
                </span>
              );
            })}
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-muted">
        Rows = from, columns = to · cell shade ∝ count (max {max}).
      </p>
    </div>
  );
}

function EmotionBands({ series }: { series: { t: number; emotion: string }[] }) {
  if (series.length === 0) {
    return <p className="text-sm text-muted">No timeline samples.</p>;
  }
  const t0 = series[0].t;
  const t1 = series[series.length - 1].t;
  const span = Math.max(0.001, t1 - t0);
  // collapse runs of identical emotions
  const runs: { start: number; end: number; emotion: string }[] = [];
  for (const row of series) {
    const last = runs[runs.length - 1];
    if (last && last.emotion === row.emotion) {
      last.end = row.t;
    } else {
      runs.push({ start: row.t, end: row.t, emotion: row.emotion });
    }
  }
  return (
    <div>
      <div className="relative h-6 overflow-hidden rounded border border-border bg-background">
        {runs.map((r, i) => (
          <span
            key={`${i}-${r.emotion}`}
            title={`${EMOTION_LABEL[r.emotion] ?? r.emotion} · ${(
              (r.start - t0) / span * 100
            ).toFixed(0)}–${((r.end - t0) / span * 100).toFixed(0)}%`}
            className="absolute inset-y-0"
            style={{
              left: `${((r.start - t0) / span) * 100}%`,
              width: `${Math.max(0.4, ((r.end - r.start) / span) * 100)}%`,
              backgroundColor: emotionColor(r.emotion),
            }}
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted tabular-nums">
        <span>0s</span>
        <span>{(span).toFixed(0)}s</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
        {EMOTIONS.map((e) => (
          <span key={e} className="inline-flex items-center gap-1 text-muted">
            <span
              className="inline-block h-2 w-2 rounded-sm"
              style={{ backgroundColor: emotionColor(e) }}
            />
            {EMOTION_LABEL[e]}
          </span>
        ))}
      </div>
    </div>
  );
}

function AnalyticsInner() {
  usePageTitle("Analytics");
  const token = useAuth((s) => s.token);
  const searchParams = useSearchParams();
  const initialSession = searchParams.get("session");

  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(initialSession);
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [prevSummary, setPrevSummary] = useState<SessionSummary | null>(null);
  const [compareOn, setCompareOn] = useState(false);
  const [timeline, setTimeline] = useState<AnalyticsTimeline | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [exporting, setExporting] = useState<string | null>(null);
  const [armedDelete, setArmedDelete] = useState(false);
  const [eventFilter, setEventFilter] = useState<string | null>(null);
  const [drillEmotion, setDrillEmotion] = useState<string | null>(null);
  const [timelineMode, setTimelineMode] = useState<"metrics" | "valence">(
    "metrics",
  );
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const selectedSession = useMemo(
    () => sessions.find((s) => s.id === selected) ?? null,
    [sessions, selected],
  );

  const filteredSessions = useMemo(() => {
    return sessions.filter((s) => {
      const day = isoDay(new Date(s.start_time));
      if (dateFrom && day < dateFrom) return false;
      if (dateTo && day > dateTo) return false;
      return true;
    });
  }, [sessions, dateFrom, dateTo]);

  const loadSessions = useCallback(async () => {
    setListLoading(true);
    try {
      const data = await apiFetch<SessionOut[]>("/api/sessions", {}, token);
      setSessions(data);
      setListError(null);
      setSelected((current) => {
        if (current) return current;
        return data[0]?.id ?? null;
      });
    } catch (err) {
      setListError(
        err instanceof ApiError ? err.message : "Failed to load sessions",
      );
    } finally {
      setListLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (!selected) {
      setSummary(null);
      setTimeline(null);
      setPrevSummary(null);
      return;
    }
    let cancelled = false;
    setDetailError(null);
    setSummary(null);
    setTimeline(null);
    setEventFilter(null);
    setDrillEmotion(null);
    Promise.all([
      apiFetch<SessionSummary>(
        `/api/analytics/summary?session_id=${selected}`,
        {},
        token,
      ),
      apiFetch<AnalyticsTimeline>(
        `/api/analytics/timeline?session_id=${selected}`,
        {},
        token,
      ),
    ])
      .then(([sum, tl]) => {
        if (cancelled) return;
        setSummary(sum);
        setTimeline(tl);
      })
      .catch((err) => {
        if (cancelled) return;
        setDetailError(
          err instanceof ApiError ? err.message : "Failed to load session",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [selected, token]);

  /* comparison: previous session in the list */
  useEffect(() => {
    if (!compareOn || !selected) {
      setPrevSummary(null);
      return;
    }
    const idx = sessions.findIndex((s) => s.id === selected);
    const prev = sessions[idx + 1];
    if (!prev) {
      setPrevSummary(null);
      return;
    }
    let cancelled = false;
    apiFetch<SessionSummary>(
      `/api/analytics/summary?session_id=${prev.id}`,
      {},
      token,
    )
      .then((s) => {
        if (!cancelled) setPrevSummary(s);
      })
      .catch(() => {
        if (!cancelled) setPrevSummary(null);
      });
    return () => {
      cancelled = true;
    };
  }, [compareOn, selected, sessions, token]);

  useEffect(() => {
    if (!armedDelete) return;
    const id = window.setTimeout(() => setArmedDelete(false), 3000);
    return () => window.clearTimeout(id);
  }, [armedDelete]);

  const pieData = useMemo(
    () =>
      Object.entries(summary?.expression_distribution ?? {}).map(
        ([name, value]) => ({
          name: EMOTION_LABEL[name] ?? name,
          value,
          key: name,
        }),
      ),
    [summary],
  );

  const events = useMemo(() => timeline?.events ?? [], [timeline]);

  const eventTypeData = useMemo(() => {
    const counts = new Map<string, number>();
    for (const ev of events) {
      counts.set(ev.event_type, (counts.get(ev.event_type) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) => ({ key: type, name: type, value: count }));
  }, [events]);

  const certaintyData = useMemo(() => {
    const counts = new Map<string, number>();
    for (const ev of events) {
      const c = ev.certainty ?? "certain";
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([certainty, count]) => ({ key: certainty, name: certainty, value: count }));
  }, [events]);

  const visibleEvents = useMemo(() => {
    let list = events;
    if (eventFilter) list = list.filter((ev) => ev.event_type === eventFilter);
    if (drillEmotion) {
      list = list.filter(
        (ev) =>
          ev.from_emotion === drillEmotion || ev.to_emotion === drillEmotion,
      );
    }
    return list;
  }, [events, eventFilter, drillEmotion]);

  /* primary person for the time series: most-sampled id */
  const { series, personId } = useMemo(() => {
    const samples = timeline?.timeline ?? [];
    const counts = new Map<number, number>();
    for (const s of samples) {
      for (const f of s.faces) counts.set(f.id, (counts.get(f.id) ?? 0) + 1);
    }
    let pid = 0;
    let best = -1;
    for (const [id, n] of counts) {
      if (n > best) {
        best = n;
        pid = id;
      }
    }
    if (samples.length === 0) return { series: [], personId: 0 };
    const t0 = samples[0].t;
    const rows = samples
      .map((s) => {
        const face = s.faces.find((f) => f.id === pid) ?? s.faces[0];
        if (!face) return null;
        return {
          t: Number((s.t - t0).toFixed(1)),
          confidence: Number((face.confidence * 100).toFixed(1)),
          intensity: Number(face.intensity.toFixed(1)),
          valence: VALENCE[face.emotion] ?? 0,
          emotion: face.emotion,
        };
      })
      .filter((r): r is NonNullable<typeof r> => r !== null);
    return { series: rows, personId: pid };
  }, [timeline]);

  const histData = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({
      bucket: `${i * 10}–${i * 10 + 10}`,
      mid: i * 10 + 5,
      count: 0,
    }));
    for (const row of series) {
      const idx = Math.min(9, Math.floor(row.confidence / 10));
      buckets[idx].count += 1;
    }
    return buckets;
  }, [series]);

  const download = async (format: "pdf" | "csv" | "json") => {
    if (!selected || !selectedSession) return;
    setExporting(format);
    try {
      const blob = await apiBlob(
        `/api/analytics/export/${selected}?format=${format}`,
        token,
      );
      const name = `hfd_${slug(sessionLabel(selectedSession))}_${dateStamp(
        new Date(selectedSession.start_time),
      )}.${format}`;
      saveBlob(blob, name);
      toast.success(`Exported ${name}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Export failed";
      setDetailError(msg);
      toast.error(msg);
    } finally {
      setExporting(null);
    }
  };

  const removeSession = async () => {
    if (!selected) return;
    if (!armedDelete) {
      setArmedDelete(true);
      toast.info("Click again within 3s to delete this session.");
      return;
    }
    try {
      await apiFetch(`/api/sessions/${selected}`, { method: "DELETE" }, token);
      toast.success("Session deleted.");
      setSelected(null);
      setSummary(null);
      setTimeline(null);
      setArmedDelete(false);
      await loadSessions();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Delete failed");
      setArmedDelete(false);
    }
  };

  return (
    <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-4">
      <h1 className="sr-only">Session Analytics</h1>
      {/* session list */}
      <Card title="Sessions" className="lg:col-span-1">
        {listError && (
          <p className="mb-2 text-xs text-red-400">{listError}</p>
        )}
        <div className="mb-2 flex gap-1.5 text-xs">
          <input
            type="date"
            aria-label="From date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="w-full rounded border border-border bg-background px-1.5 py-1"
          />
          <input
            type="date"
            aria-label="To date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="w-full rounded border border-border bg-background px-1.5 py-1"
          />
        </div>
        {listLoading && (
          <div className="space-y-1.5">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        )}
        {!listLoading && filteredSessions.length === 0 && !listError && (
          <p className="text-xs text-muted">
            {sessions.length === 0
              ? "No sessions yet — run a Live Observer session first."
              : "No sessions in this date range."}
          </p>
        )}
        <ul className="space-y-1.5">
          {filteredSessions.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                onClick={() => setSelected(s.id)}
                aria-current={selected === s.id}
                className={`w-full rounded-md border px-2 py-2 text-left text-xs transition ${
                  selected === s.id
                    ? "border-accent bg-accent/10"
                    : "border-border hover:bg-border/40"
                }`}
              >
                <span className="block font-medium">
                  {sessionLabel(s)}
                </span>
                <span className="text-muted">
                  {new Date(s.start_time).toLocaleString()} ·{" "}
                  {formatSessionDuration(s.duration_seconds)} · {s.total_events}{" "}
                  events
                </span>
              </button>
            </li>
          ))}
        </ul>
        {selected && (
          <Button
            variant="danger"
            className={`mt-3 w-full ${armedDelete ? "ring-2 ring-red-400" : ""}`}
            onClick={removeSession}
          >
            {armedDelete ? "Confirm delete — click again" : "Delete session"}
          </Button>
        )}
      </Card>

      {/* stats + charts */}
      <div className="space-y-4 lg:col-span-3">
        {detailError && (
          <p role="alert" className="text-sm text-red-400">
            {detailError}
          </p>
        )}
        {!selected && (
          <Card title="Statistics">
            <p className="text-sm text-muted">
              Select a session to view expression distribution, timeline,
              transitions, and exports.
            </p>
          </Card>
        )}

        {selected && !summary && !detailError && (
          <Card title="Statistics">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Skeleton className="h-14" />
              <Skeleton className="h-14" />
              <Skeleton className="h-14" />
              <Skeleton className="h-14" />
            </div>
          </Card>
        )}

        {selected && summary && (
          <>
            <Card
              title={selectedSession ? sessionLabel(selectedSession) : "Statistics"}
              action={
                <div className="flex flex-wrap gap-1.5">
                  <Button
                    variant="ghost"
                    aria-pressed={compareOn}
                    onClick={() => setCompareOn((v) => !v)}
                    title="Compare against the previous session"
                  >
                    {compareOn ? "Comparing" : "Compare"}
                  </Button>
                  <Button
                    variant="ghost"
                    disabled={exporting !== null}
                    onClick={() => download("pdf")}
                  >
                    {exporting === "pdf" ? "…" : "Export PDF"}
                  </Button>
                  <Button
                    variant="ghost"
                    disabled={exporting !== null}
                    onClick={() => download("csv")}
                  >
                    {exporting === "csv" ? "…" : "Export CSV"}
                  </Button>
                  <Button
                    variant="ghost"
                    disabled={exporting !== null}
                    onClick={() => download("json")}
                  >
                    {exporting === "json" ? "…" : "JSON"}
                  </Button>
                </div>
              }
            >
              {compareOn && !prevSummary && (
                <p className="mb-2 text-xs text-muted">
                  No previous session to compare against.
                </p>
              )}
              <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                {(
                  [
                    ["Duration", formatSessionDuration(summary.duration_seconds), formatSessionDuration(prevSummary?.duration_seconds)],
                    ["Frames", String(summary.total_frames), prevSummary ? String(prevSummary.total_frames) : null],
                    ["Events", String(summary.total_events), prevSummary ? String(prevSummary.total_events) : null],
                    ["Model", summary.model ?? "—", prevSummary?.model ?? null],
                    [
                      "Avg confidence",
                      summary.avg_confidence != null ? pct(summary.avg_confidence, 1) : "—",
                      prevSummary?.avg_confidence != null ? pct(prevSummary.avg_confidence, 1) : null,
                    ],
                    [
                      "Avg intensity",
                      summary.avg_intensity != null ? summary.avg_intensity.toFixed(1) : "—",
                      prevSummary?.avg_intensity != null ? prevSummary.avg_intensity.toFixed(1) : null,
                    ],
                    ["Transitions", String(summary.transition_count), prevSummary ? String(prevSummary.transition_count) : null],
                    ["Uncertainty events", String(summary.uncertainty_events), prevSummary ? String(prevSummary.uncertainty_events) : null],
                  ] as [string, string, string | null][]
                ).map(([label, value, prev]) => (
                  <div key={label}>
                    <dt className="text-xs text-muted">{label}</dt>
                    <dd className="font-semibold">
                      {value}
                      {compareOn && prev != null && (
                        <span className="ml-1 text-[11px] font-normal text-muted">
                          (was {prev})
                        </span>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </Card>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card
                title="Expression distribution"
                action={
                  drillEmotion ? (
                    <button
                      type="button"
                      className="rounded bg-accent/15 px-2 py-0.5 text-xs text-accent"
                      onClick={() => setDrillEmotion(null)}
                    >
                      {EMOTION_LABEL[drillEmotion] ?? drillEmotion} ✕
                    </button>
                  ) : (
                    <span className="text-[11px] text-muted">click a slice</span>
                  )
                }
              >
                {pieData.length === 0 ? (
                  <p className="text-sm text-muted">No timeline samples.</p>
                ) : (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          dataKey="value"
                          nameKey="name"
                          outerRadius={90}
                          label={(entry) => entry.name}
                          onClick={(entry: unknown) => {
                            const key = (entry as { key?: string })?.key;
                            if (key) {
                              setDrillEmotion((cur) => (cur === key ? null : key));
                            }
                          }}
                        >
                          {pieData.map((entry) => (
                            <Cell
                              key={entry.key}
                              fill={emotionColor(entry.key)}
                              opacity={
                                !drillEmotion || drillEmotion === entry.key ? 1 : 0.35
                              }
                              style={{ cursor: "pointer" }}
                            />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                )}
                {drillEmotion && (
                  <p className="text-xs text-muted">
                    {visibleEvents.length} of {events.length} events involve{" "}
                    {EMOTION_LABEL[drillEmotion] ?? drillEmotion} (see Events
                    below).
                  </p>
                )}
              </Card>

              <Card
                title={
                  series.length > 0
                    ? `Timeline — ${personLabel(personId)}`
                    : "Timeline"
                }
                action={
                  <div className="flex gap-1 text-[11px]">
                    <button
                      type="button"
                      onClick={() => setTimelineMode("metrics")}
                      className={`rounded px-2 py-0.5 ${
                        timelineMode === "metrics"
                          ? "bg-accent/15 text-accent"
                          : "text-muted hover:text-foreground"
                      }`}
                    >
                      Metrics
                    </button>
                    <button
                      type="button"
                      onClick={() => setTimelineMode("valence")}
                      className={`rounded px-2 py-0.5 ${
                        timelineMode === "valence"
                          ? "bg-accent/15 text-accent"
                          : "text-muted hover:text-foreground"
                      }`}
                    >
                      Valence
                    </button>
                  </div>
                }
              >
                {series.length === 0 ? (
                  <p className="text-sm text-muted">No timeline samples.</p>
                ) : (
                  <>
                    <EmotionBands series={series} />
                    <div className="mt-3 h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={series}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                          <XAxis
                            dataKey="t"
                            stroke="#9ca3af"
                            label={{
                              value: "seconds",
                              position: "insideBottomRight",
                              offset: -2,
                            }}
                          />
                          {timelineMode === "metrics" ? (
                            <YAxis domain={[0, 100]} stroke="#9ca3af" />
                          ) : (
                            <YAxis domain={[-1, 1]} stroke="#9ca3af" />
                          )}
                          <Tooltip />
                          <Legend />
                          {timelineMode === "metrics" ? (
                            <>
                              <Line
                                type="monotone"
                                dataKey="confidence"
                                stroke="#3b82f6"
                                dot={false}
                                isAnimationActive={false}
                              />
                              <Line
                                type="monotone"
                                dataKey="intensity"
                                stroke="#f59e0b"
                                dot={false}
                                isAnimationActive={false}
                              />
                            </>
                          ) : (
                            <Line
                              type="monotone"
                              dataKey="valence"
                              stroke="#a855f7"
                              dot={false}
                              isAnimationActive={false}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    {timelineMode === "valence" && (
                      <p className="mt-1 text-[11px] text-muted">
                        Proxy valence (−1..+1) mapped from the predicted
                        emotion — a directional cue only, not a clinical
                        measurement.
                      </p>
                    )}
                  </>
                )}
              </Card>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Transitions">
                <TransitionMatrix matrix={summary.transition_matrix} />
              </Card>

              <Card title="Uncertainty">
                <div className="mb-2 flex items-baseline gap-3">
                  <span className="text-2xl font-bold tabular-nums">
                    {summary.uncertainty_events}
                  </span>
                  <span className="text-xs text-muted">
                    ambiguity / low-margin events
                    <InfoTip text="Derived from top-1 vs top-2 margin and certainty flags — the model surfacing its own doubt, not a user judgement." />
                  </span>
                </div>
                {certaintyData.length === 0 ? (
                  <p className="text-sm text-muted">No events recorded.</p>
                ) : (
                  <ul className="space-y-1 text-sm">
                    {certaintyData.map((row) => (
                      <li
                        key={row.key}
                        className="flex justify-between border-b border-border/60 pb-1"
                      >
                        <span className="uppercase text-muted">{row.key}</span>
                        <span className="tabular-nums">{row.value}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card
                title="Event types"
                action={
                  eventFilter ? (
                    <button
                      type="button"
                      className="rounded bg-accent/15 px-2 py-0.5 text-xs text-accent"
                      onClick={() => setEventFilter(null)}
                    >
                      {eventFilter} ✕
                    </button>
                  ) : (
                    <span className="text-[11px] text-muted">click to filter</span>
                  )
                }
              >
                {eventTypeData.length === 0 ? (
                  <p className="text-sm text-muted">No events recorded.</p>
                ) : (
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="h-40 w-40">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={eventTypeData}
                            dataKey="value"
                            nameKey="name"
                            outerRadius={60}
                            onClick={(entry: unknown) => {
                              const key = (entry as { key?: string })?.key;
                              if (key) {
                                setEventFilter((cur) => (cur === key ? null : key));
                              }
                            }}
                          >
                            {eventTypeData.map((entry, i) => (
                              <Cell
                                key={entry.key}
                                fill={
                                  ["#3b82f6", "#f59e0b", "#a855f7", "#22c55e", "#ec4899", "#94a3b8", "#ef4444"][i % 7]
                                }
                                opacity={
                                  !eventFilter || eventFilter === entry.key ? 1 : 0.3
                                }
                                style={{ cursor: "pointer" }}
                              />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <ul className="flex-1 space-y-1 text-xs">
                      {eventTypeData.map((row) => (
                        <li key={row.key}>
                          <button
                            type="button"
                            onClick={() =>
                              setEventFilter((cur) =>
                                cur === row.key ? null : row.key,
                              )
                            }
                            className={`w-full rounded px-1.5 py-0.5 text-left transition ${
                              eventFilter === row.key
                                ? "bg-accent/15 text-accent"
                                : "hover:bg-border/40"
                            }`}
                          >
                            {row.key}{" "}
                            <span className="text-muted">×{row.value}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </Card>

              <Card title="Confidence histogram">
                {series.length === 0 ? (
                  <p className="text-sm text-muted">No timeline samples.</p>
                ) : (
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={histData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
                        <XAxis dataKey="bucket" stroke="#9ca3af" fontSize={10} />
                        <YAxis stroke="#9ca3af" allowDecimals={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#3b82f6" isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
                <p className="mt-1 text-[11px] text-muted">
                  Distribution of per-frame top-1 confidence (%).
                </p>
              </Card>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Per-person analysis">
                {Object.keys(summary.per_person).length === 0 ? (
                  <p className="text-sm text-muted">No faces observed.</p>
                ) : (
                  <ul className="space-y-1 text-sm">
                    {Object.entries(summary.per_person).map(([pid, info]) => (
                      <li
                        key={pid}
                        className="flex justify-between border-b border-border/60 pb-1"
                      >
                        <span>{personLabel(Number(pid))}</span>
                        <span>
                          <span
                            style={{ color: emotionColor(info.dominant) }}
                            className="font-semibold uppercase"
                          >
                            {EMOTION_LABEL[info.dominant] ?? info.dominant}
                          </span>{" "}
                          <span className="text-muted">
                            ({info.samples} samples)
                          </span>
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
              <Card title="Legend">
                <ul className="grid grid-cols-2 gap-1 text-xs">
                  {EMOTIONS.map((e) => (
                    <li key={e} className="flex items-center gap-2">
                      <span
                        className="inline-block h-2.5 w-2.5 rounded-sm"
                        style={{ backgroundColor: emotionColor(e) }}
                      />
                      {EMOTION_LABEL[e]}
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-[11px] text-muted">
                  All numbers computed locally from this session only.
                </p>
              </Card>
            </div>

            <Card title={`Events (${visibleEvents.length}${visibleEvents.length !== events.length ? ` of ${events.length}` : ""})`}>
              {(eventFilter || drillEmotion) && (
                <div className="mb-2 flex flex-wrap gap-1.5">
                  {eventFilter && (
                    <button
                      type="button"
                      className="rounded bg-accent/15 px-2 py-0.5 text-xs text-accent"
                      onClick={() => setEventFilter(null)}
                    >
                      type: {eventFilter} ✕
                    </button>
                  )}
                  {drillEmotion && (
                    <button
                      type="button"
                      className="rounded bg-accent/15 px-2 py-0.5 text-xs text-accent"
                      onClick={() => setDrillEmotion(null)}
                    >
                      emotion: {EMOTION_LABEL[drillEmotion] ?? drillEmotion} ✕
                    </button>
                  )}
                </div>
              )}
              {visibleEvents.length === 0 ? (
                <p className="text-sm text-muted">
                  {events.length === 0
                    ? "No events recorded."
                    : "No events match the current filters."}
                </p>
              ) : (
                <div className="max-h-72 overflow-y-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-surface text-muted">
                      <tr>
                        <th className="py-1 pr-2">Time</th>
                        <th className="py-1 pr-2">Face</th>
                        <th className="py-1 pr-2">Event</th>
                        <th className="py-1 pr-2">From → To</th>
                        <th className="py-1">Certainty</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...visibleEvents].reverse().map((ev) => (
                        <tr key={ev.id} className="border-t border-border/50">
                          <td className="py-1 pr-2 tabular-nums">
                            {new Date(ev.timestamp).toLocaleTimeString()}
                          </td>
                          <td className="py-1 pr-2">
                            {personLabel(ev.face_id)}
                          </td>
                          <td className="py-1 pr-2">{ev.event_type}</td>
                          <td className="py-1 pr-2">
                            {ev.from_emotion || ""}
                            {ev.to_emotion ? ` → ${ev.to_emotion}` : ""}
                          </td>
                          <td className="py-1 uppercase">{ev.certainty}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  return (
    <Suspense
      fallback={
        <p className="p-4 text-sm text-muted">Loading analytics…</p>
      }
    >
      <AnalyticsInner />
    </Suspense>
  );
}
