"use client";

import type { ReactNode } from "react";
import { useReducedMotion } from "framer-motion";
import PortalFace from "./PortalFace";

const TELEMETRY: [string, string][] = [
  ["Inference", "Local · On-device"],
  ["Frames", "Never stored"],
  ["Model", "DAN · Ensemble"],
];

/* Card chrome shared with the login page (sf-mono/sf-ui/keyframes come from PortalFace). */
const CARD_CSS = `
.sf-card {
  position: relative;
  background: rgba(16, 33, 44, 0.6);
  -webkit-backdrop-filter: blur(24px);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(125, 249, 255, 0.2);
  clip-path: polygon(
    0 0,
    calc(100% - 16px) 0,
    100% 16px,
    100% 100%,
    16px 100%,
    0 calc(100% - 16px)
  );
  box-shadow:
    inset 0 -1px 0 rgba(0, 240, 255, 0.4),
    inset 0 -30px 50px -30px rgba(0, 240, 255, 0.28);
}
.sf-card::before,
.sf-card::after {
  content: "";
  position: absolute;
  width: 23px;
  height: 1px;
  background: linear-gradient(90deg, rgba(0, 240, 255, 0.2), rgba(0, 240, 255, 0.95));
  pointer-events: none;
}
.sf-card::before {
  top: 0;
  right: -7px;
  transform-origin: left center;
  transform: rotate(45deg);
}
.sf-card::after {
  left: 0;
  bottom: 16px;
  transform-origin: left center;
  transform: rotate(45deg);
}
`;

export interface AuthShellProps {
  /** Two-line brand, first line plain, second line accented. */
  titleLines: [string, string];
  /** Sentence shown under the brand mark. */
  tagline: string;
  /** Telemetry rows shown in the left rail (desktop). */
  telemetry?: [string, string][];
  children: ReactNode;
  /** Status text in the footer, right side. */
  status?: string;
  /**
   * "grid" = the original instrument-panel frame.
   * "portal" = full-bleed animated face backdrop with a split layout.
   */
  variant?: "grid" | "portal";
}

/**
 * Shared HUD frame for the auth screens (login / register).
 * Owns the ambient atmosphere: grid, radial glows, scan sweep, corner brackets
 * and the left telemetry rail. Pages only supply their content.
 */
export default function AuthShell({
  titleLines,
  tagline,
  telemetry = TELEMETRY,
  children,
  status = "Offline ready",
  variant = "grid",
}: AuthShellProps) {
  const reduced = !!useReducedMotion();

  if (variant === "portal") {
    return (
      <main className="relative flex min-h-screen flex-col overflow-hidden bg-[#020A0E] text-white lg:block">
        <style>{CARD_CSS}</style>

        {/* Desktop: full-bleed animated face (same identity as the login portal) */}
        <div className="absolute inset-0 hidden lg:block">
          <div className="relative h-full w-full">
            <PortalFace reduced={reduced} dimmed />
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0"
              style={{
                background:
                  "linear-gradient(90deg, #020A0E 20%, rgba(2,10,14,0.86) 42%, rgba(2,10,14,0.25) 66%, transparent 85%)",
              }}
            />
          </div>
        </div>

        {/* Mobile: face as a top banner, fading into the page */}
        <div className="relative h-60 overflow-hidden sm:h-72 lg:hidden">
          <PortalFace reduced={reduced} />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                "linear-gradient(180deg, rgba(2,10,14,0.15) 0%, transparent 45%, #020A0E 100%)",
            }}
          />
        </div>

        {/* Chrome: corner brackets framing the viewport */}
        <div aria-hidden className="pointer-events-none absolute inset-4 sm:inset-6">
          {(
            [
              "left-0 top-0 border-l border-t",
              "right-0 top-0 border-r border-t",
              "left-0 bottom-0 border-l border-b",
              "right-0 bottom-0 border-r border-b",
            ] as const
          ).map((pos) => (
            <span key={pos} className={`absolute h-10 w-10 border-cyan-300/50 ${pos}`} />
          ))}
          <span className="absolute left-1/2 top-0 h-4 w-px -translate-x-1/2 bg-cyan-300/30" />
        </div>

        {/* Content: left-aligned column over the dimmed field */}
        <div className="relative z-10 mx-auto flex w-full max-w-7xl flex-1 items-center px-6 py-10 sm:px-10 lg:min-h-screen lg:px-16">
          <section className="w-full max-w-md">
            <header className="mb-7">
              <p className="font-mono text-[10px] uppercase tracking-[0.34em] text-cyan-300/70">
                Ranking-based · Explainable AI
              </p>
              <h1 className="mt-4 text-3xl font-extrabold uppercase leading-[1.05] tracking-wide text-white">
                {titleLines[0]}
                <br />
                <span className="text-cyan-300 drop-shadow-[0_0_14px_rgba(34,211,238,0.55)]">
                  {titleLines[1]}
                </span>
              </h1>
              <p className="mt-3 max-w-sm text-xs leading-relaxed text-white/45">
                {tagline}
              </p>
            </header>

            <div className="sf-card p-6 pt-7 sm:p-8 sm:pt-9 shadow-[0_0_30px_rgba(0,240,255,0.15)]">
              {children}
            </div>

            <div className="mt-8 flex items-center justify-between border-t border-white/[0.06] pt-4 font-mono text-[10px] uppercase tracking-[0.22em] text-white/25">
              <span>v1.0 · Local build</span>
              <span className="flex items-center gap-2">
                <span className="h-1 w-1 animate-pulse rounded-full bg-emerald-400" />
                {status}
              </span>
            </div>
          </section>
        </div>

        {/* Fine scanlines (screen texture) */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.35] mix-blend-overlay"
          style={{
            backgroundImage:
              "repeating-linear-gradient(0deg, rgba(255,255,255,0.028) 0px, rgba(255,255,255,0.028) 1px, transparent 1px, transparent 3px)",
          }}
        />
      </main>
    );
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#04070c] text-white">
      {/* Ambient field: faint technical grid + vignette */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage:
            "linear-gradient(rgba(103,232,249,0.045) 1px, transparent 1px), " +
            "linear-gradient(90deg, rgba(103,232,249,0.045) 1px, transparent 1px)",
          backgroundSize: "44px 44px",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(1100px 620px at 78% 40%, rgba(34,211,238,0.13), transparent 62%), " +
            "radial-gradient(760px 520px at 18% 82%, rgba(59,130,246,0.10), transparent 60%)",
        }}
      />

      {/* Vertical scan sweep (ambient, respects reduced-motion) */}
      <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="hf-scanline absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-300/20 to-transparent" />
      </div>

      {/* Chrome: corner brackets framing the viewport */}
      <div aria-hidden className="pointer-events-none absolute inset-4 sm:inset-6">
        {(
          [
            "left-0 top-0 border-l border-t",
            "right-0 top-0 border-r border-t",
            "left-0 bottom-0 border-l border-b",
            "right-0 bottom-0 border-r border-b",
          ] as const
        ).map((pos) => (
          <span key={pos} className={`absolute h-10 w-10 border-cyan-300/50 ${pos}`} />
        ))}
        <span className="absolute left-1/2 top-0 h-4 w-px -translate-x-1/2 bg-cyan-300/30" />
      </div>

      {/* Left rail: system telemetry (desktop only) */}
      <aside className="pointer-events-none absolute inset-y-0 left-0 hidden w-72 flex-col justify-between border-r border-white/[0.06] px-8 py-16 xl:flex">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.34em] text-cyan-300/70">
            Affect Analysis
          </p>
          <div className="mt-3 h-px w-12 bg-cyan-300/50" />
          <p className="mt-4 max-w-[13rem] text-[11px] leading-relaxed text-white/35">
            Real-time expression classification with ranked evidence, region
            attribution and calibrated confidence.
          </p>
        </div>

        <div className="space-y-4 font-mono text-[10px] uppercase tracking-widest text-white/30">
          {telemetry.map(([k, v]) => (
            <div key={k} className="flex items-center gap-3">
              <span className="inline-block h-1 w-1 rounded-full bg-cyan-300 shadow-[0_0_8px_rgba(103,232,249,0.9)]" />
              <span>{k}</span>
              <span className="ml-auto text-white/55">{v}</span>
            </div>
          ))}
        </div>
      </aside>

      {/* Center stage */}
      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-12">
        <section className="w-full max-w-[26.5rem]">
          <header className="mb-9">
            <div className="flex items-center gap-3">
              {/* Compact identity mark: scan-ring */}
              <span className="relative grid h-9 w-9 place-items-center">
                <span className="absolute inset-0 rounded-full border border-cyan-300/50" />
                <span className="absolute inset-[5px] animate-pulse rounded-full border border-cyan-300/25" />
                <span className="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_10px_rgba(103,232,249,0.95)]" />
              </span>
              <div>
                <p className="text-[13px] font-semibold uppercase leading-none tracking-[0.2em] text-white">
                  {titleLines[0]}
                </p>
                <p className="mt-1 text-[13px] font-semibold uppercase leading-none tracking-[0.2em] text-cyan-300">
                  {titleLines[1]}
                </p>
              </div>
            </div>
            <p className="mt-5 max-w-sm text-xs leading-relaxed text-white/45">
              {tagline}
            </p>
          </header>

          {children}

          <div className="mt-10 flex items-center justify-between border-t border-white/[0.06] pt-4 font-mono text-[10px] uppercase tracking-[0.22em] text-white/25">
            <span>v1.0 · Local build</span>
            <span className="flex items-center gap-2">
              <span className="h-1 w-1 animate-pulse rounded-full bg-emerald-400" />
              {status}
            </span>
          </div>
        </section>
      </div>

      {/* Fine scanlines (screen texture) */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.35] mix-blend-overlay"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, rgba(255,255,255,0.028) 0px, rgba(255,255,255,0.028) 1px, transparent 1px, transparent 3px)",
        }}
      />
    </main>
  );
}