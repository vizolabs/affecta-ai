"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Card from "../../../components/common/Card";
import { apiFetch } from "../../../lib/api";
import { sessionLabel } from "../../../lib/sessions";
import type { SessionOut } from "../../../lib/types";
import { useAuth } from "../../../stores/auth";
import { usePageTitle } from "../../../hooks/usePageTitle";

const CHECKLIST = [
  {
    href: "/live",
    label: "Start a live session",
    hint: "Camera → single-person analysis + narration",
  },
  {
    href: "/lab",
    label: "Try the Research Lab",
    hint: "Upload a photo · Grad-CAM · model inspector",
  },
  {
    href: "/analytics",
    label: "Review analytics",
    hint: "Session timelines, transitions, exports",
  },
  {
    href: "/settings",
    label: "Tune settings",
    hint: "Narration, smoothing, danger zone",
  },
];

export default function DashboardPage() {
  usePageTitle("Dashboard");
  const token = useAuth((s) => s.token);
  const [sessions, setSessions] = useState<SessionOut[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch<SessionOut[]>("/api/sessions?limit=5", {}, token)
      .then((data) => {
        if (!cancelled) setSessions(data);
      })
      .catch(() => {
        if (!cancelled) setSessions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  return (
    <div className="mx-auto grid max-w-5xl gap-4">
      <div className="mb-1 flex items-center gap-3">
          <span className="relative grid h-8 w-8 place-items-center">
            <span className="absolute inset-0 rounded-full border border-accent/50" />
            <span className="absolute inset-[5px] animate-pulse rounded-full border border-accent/25" />
            <span className="h-1 w-1 rounded-full bg-accent shadow-[0_0_8px_rgba(103,232,249,0.95)]" />
          </span>
          <div>
            <h1 className="text-xl font-semibold uppercase tracking-[0.12em] text-foreground">
              Welcome to Human Facial{" "}
              <span className="text-accent">Emotion Detection</span>
            </h1>
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
              Ranking-based explainable facial emotion detection — 100% local
            </p>
          </div>
        </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card title="Quick start">
          <ul className="space-y-2">
            {CHECKLIST.map((item) => (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className="group flex items-start gap-2 rounded-md border border-border bg-background p-2.5 transition hover:border-accent/60"
                >
                  <span className="mt-0.5 text-accent">→</span>
                  <span>
                    <span className="block text-sm font-medium group-hover:text-accent">
                      {item.label}
                    </span>
                    <span className="block text-xs text-muted">
                      {item.hint}
                    </span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Recent sessions">
          {sessions === null ? (
            <div className="space-y-2">
              <div className="h-10 animate-pulse rounded bg-border/50" />
              <div className="h-10 animate-pulse rounded bg-border/50" />
              <div className="h-10 animate-pulse rounded bg-border/50" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-sm text-muted">
              <p>No sessions yet.</p>
              <p className="mt-1 text-xs">
                Start one from{" "}
                <Link href="/live" className="text-accent">
                  Live Observer
                </Link>{" "}
                or analyze a photo in the{" "}
                <Link href="/lab" className="text-accent">
                  Research Lab
                </Link>
                .
              </p>
            </div>
          ) : (
            <ul className="space-y-2">
              {sessions.map((s) => (
                <li key={s.id}>
                  <Link
                    href="/analytics"
                    className="flex items-center justify-between rounded-md border border-border bg-background px-3 py-2 text-sm transition hover:border-accent/60"
                  >
                    <span className="truncate">{sessionLabel(s)}</span>
                    <span className="ml-2 shrink-0 text-xs text-muted tabular-nums">
                      {new Date(s.start_time).toLocaleDateString()}
                    </span>
                  </Link>
                </li>
              ))}
              <li>
                <Link
                  href="/analytics"
                  className="block pt-1 text-xs text-accent"
                >
                  View all in Analytics →
                </Link>
              </li>
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
