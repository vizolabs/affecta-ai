"use client";

import { useEffect, useState, type FormEvent } from "react";
import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { apiFetch, ApiError } from "../lib/api";
import type { TokenOut } from "../lib/types";
import { useAuth } from "../stores/auth";
import { usePageTitle } from "../hooks/usePageTitle";
import PortalFace, { EASE } from "../components/auth/PortalFace";

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/** Ranking labels beside the face — decorative sample output (clearly captioned). */
const EMO = [
  { label: "JOY", value: 92.4 },
  { label: "SURPRISE", value: 45.1 },
  { label: "ANGER", value: 12.8 },
] as const;

/** Faint ambient code column on the right edge. */
const CODE_STRIP = [
  "0x8F21  rank.frame(sym)",
  "w_au <- 0.0  (untrained)",
  "loop: capture -> detect",
  "raf -> softmax -> rank",
  "region: brow · eye · mouth",
  "ece 0.018 | tta: on",
  "if face: attr(gradcam)",
  "store: none // local",
  "fps 8 | hold 0s",
  "ready.",
];

const NOISE_URL =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)'/%3E%3C/svg%3E\")";

/* Card chrome (sf-mono / sf-ui / keyframes live in PortalFace). */
const CSS = `
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

/* ------------------------------------------------------------------ */
/* Icons                                                               */
/* ------------------------------------------------------------------ */

function IconUser({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <circle cx="12" cy="8" r="4" />
      <path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
    </svg>
  );
}

function IconLock({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <rect x="5" y="11" width="14" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Ranking labels                                                      */
/* ------------------------------------------------------------------ */

function RankLabels({ reduced }: { reduced: boolean }) {
  return (
    <motion.aside
      aria-hidden
      className="pointer-events-none absolute left-[42%] top-1/2 z-20 hidden -translate-y-1/2 min-[1360px]:block"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.7, duration: 0.6, ease: EASE }}
    >
      <p className="sf-mono mb-5 whitespace-nowrap text-[9px] uppercase tracking-[0.32em] text-white/60">
        sample · illustrative
      </p>
      <div className="space-y-5">
        {EMO.map((e) => (
          <div key={e.label} className="relative w-[150px]">
            <div className="flex items-center justify-between">
              <span className="sf-mono text-[11px] tracking-[0.24em] text-[#00F0FF]">
                {e.label}
              </span>
              <span className="sf-mono text-[11px] text-[#00F0FF]/70">
                [{e.value.toFixed(1)}%]
              </span>
            </div>
            <div className="relative mt-2">
              <div className="h-[5px] w-[150px] overflow-hidden rounded-full border border-white/10 bg-white/[0.05]">
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    background:
                      "linear-gradient(90deg, #00F0FF 0%, #00F0FF 84%, #FF6A00 100%)",
                  }}
                  initial={{ width: "0%" }}
                  animate={{ width: `${e.value}%` }}
                  transition={{
                    delay: 0.8,
                    duration: reduced ? 0 : 1.2,
                    ease: EASE,
                  }}
                />
              </div>
              <span className="absolute left-[166px] top-1/2 h-px w-[78px] -translate-y-1/2 bg-gradient-to-r from-[#00F0FF]/50 to-[#00F0FF]/10" />
              <span className="absolute left-[242px] top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-[#00F0FF] shadow-[0_0_8px_rgba(0,240,255,0.9)]" />
            </div>
          </div>
        ))}
      </div>
    </motion.aside>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

const INPUT_CLS =
  "h-11 w-full rounded-lg border border-white/10 bg-white/[0.04] pl-11 pr-4 " +
  "text-sm text-white placeholder:text-white/30 outline-none transition-all duration-200 " +
  "focus:border-[#00F0FF] focus:bg-white/[0.06] " +
  "focus:shadow-[0_0_0_1px_rgba(0,240,255,0.3),0_0_18px_rgba(0,240,255,0.18)] " +
  "focus:scale-[1.01] focus-visible:outline-none disabled:cursor-not-allowed";

export default function PortalPage() {
  usePageTitle("Sign in");
  const router = useRouter();
  const token = useAuth((s) => s.token);
  const setAuth = useAuth((s) => s.setAuth);
  const reduced = !!useReducedMotion();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (token) router.replace("/dashboard");
  }, [token, router]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const data = await apiFetch<TokenOut>("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      setAuth(data.token, data.user);
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Sign in failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="sf-ui relative min-h-screen overflow-hidden bg-[#020A0E] text-white selection:bg-[#00F0FF]/30">
      <style>{CSS}</style>

      {/* ---------- Background: dots, glows, particles, wireframe face ---------- */}
      <motion.div
        aria-hidden
        className="absolute inset-0 z-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: EASE }}
      >
        <PortalFace reduced={reduced} dimmed />
      </motion.div>

      {/* ---------- Ranking labels ---------- */}
      <RankLabels reduced={reduced} />

      {/* ---------- Right-edge code strip ---------- */}
      <div
        aria-hidden
        className="sf-mono pointer-events-none absolute right-7 top-[26%] z-10 hidden w-48 text-[8px] leading-[1.9] tracking-[0.12em] text-[#7DF9FF] xl:block"
        style={{ opacity: 0.07 }}
      >
        {CODE_STRIP.map((line) => (
          <p key={line}>{line}</p>
        ))}
      </div>

      {/* ---------- Bottom HUD ---------- */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 bottom-0 z-10 hidden items-end justify-between px-9 pb-6 md:flex"
        style={{ opacity: 0.16 }}
      >
        <div className="sf-mono space-y-1 text-[10px] tracking-[0.22em] text-[#7DF9FF]">
          <p>▸ 0x4F2A :: AFFECT-CORE :: CHANNEL-08</p>
          <p>▸ 0110 1001 1100 0010 1111 0101 0011 1010</p>
          <p className="flex items-center gap-2">
            ▸ SIGNAL
            <span
              className="inline-block h-2 w-px bg-[#00F0FF]"
              style={{
                animation: "sf-blink 1.1s steps(2, jump-none) infinite",
              }}
            />
          </p>
        </div>
        <p className="sf-mono text-[10px] tracking-[0.28em] text-[#7DF9FF]">
          ▁▂▃▅▆█▅▃▂ ▪ ambient · illustrative
        </p>
      </div>

      {/* ---------- Vignette ---------- */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-30"
        style={{
          background:
            "radial-gradient(ellipse 90% 80% at 50% 45%, transparent 40%, rgba(2,10,14,0.55) 100%)",
        }}
      />

      {/* ---------- Login card ---------- */}
      <div className="relative z-40 flex min-h-screen items-center justify-center px-6 lg:justify-start lg:pl-[7vw]">
        <motion.section
          className="w-[420px] max-w-[calc(100vw_-_3rem)]"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduced ? 0 : 0.6, ease: EASE }}
        >
          <div className="shadow-[0_0_30px_rgba(0,240,255,0.15)]">
            <div className="sf-card p-6 pt-7 sm:p-8 sm:pt-9">
              <span
                aria-hidden
                className="absolute left-0 right-4 top-0 h-px bg-gradient-to-r from-[#00F0FF] via-[#00F0FF]/40 to-transparent"
              />

              <h1 className="text-[19px] font-bold uppercase leading-[1.15] tracking-[0.12em] text-[#E8F6F8] sm:text-[22px]">
                <span className="block">A Ranking-Based</span>
                <span className="block">Explainable AI Model</span>
              </h1>
              <p className="sf-mono mt-2.5 text-[11px] uppercase tracking-[0.3em] text-[#00F0FF]/60">
                For Human Facial Emotion Detection
              </p>

              <form onSubmit={submit} className="mt-7 space-y-3.5">
                <div className="relative">
                  <IconUser className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/50" />
                  <input
                    id="email"
                    type="email"
                    required
                    autoComplete="email"
                    aria-label="Email"
                    placeholder="you@local"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className={INPUT_CLS}
                  />
                </div>
                <div className="relative">
                  <IconLock className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/50" />
                  <input
                    id="password"
                    type="password"
                    required
                    autoComplete="current-password"
                    aria-label="Password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className={INPUT_CLS}
                  />
                </div>

                <div aria-live="polite" className="min-h-[1rem] pt-0.5">
                  {error && (
                    <p role="alert" className="text-[11px] text-rose-300">
                      {error}
                    </p>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={busy}
                  className="group relative mt-1 flex h-12 w-full items-center justify-center overflow-hidden border border-[#00F0FF] bg-transparent shadow-[inset_0_0_12px_rgba(0,240,255,0.3)] transition-colors duration-300 hover:bg-[#00F0FF] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <span
                    aria-hidden
                    className="absolute inset-y-0 -left-1/3 w-1/3 skew-x-[-18deg] bg-white/20 transition-transform duration-700 group-hover:translate-x-[420%]"
                  />
                  <span className="sf-mono relative text-[12px] font-semibold tracking-[0.34em] text-[#00F0FF] [text-shadow:0_0_12px_rgba(0,240,255,0.8)] group-hover:text-[#020A0E] group-hover:[text-shadow:none]">
                    {busy ? "VERIFYING" : "SIGN IN"}
                  </span>
                </button>

                <div className="mt-5 flex items-center justify-between">
                  <Link
                    href="/register"
                    className="sf-mono text-[10px] uppercase tracking-[0.24em] text-white/60 transition hover:text-[#00F0FF]"
                  >
                    Provision an account →
                  </Link>
                  <span className="sf-mono text-[10px] tracking-[0.2em] text-white/60">
                    local
                  </span>
                </div>
              </form>
            </div>
          </div>

          <p className="sf-mono mt-6 text-center text-[10px] tracking-[0.22em] text-white/60 lg:text-left">
            fully local · frames never leave this device
          </p>
        </motion.section>
      </div>

      {/* ---------- Film-grain noise ---------- */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 z-50 opacity-[0.02]"
        style={{ backgroundImage: NOISE_URL }}
      />
    </main>
  );
}
