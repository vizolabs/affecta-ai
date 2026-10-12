"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { apiFetch, ApiError } from "../lib/api";
import type { TokenOut } from "../lib/types";
import { useAuth } from "../stores/auth";
import { usePageTitle } from "../hooks/usePageTitle";

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];
const TAU = Math.PI * 2;
const ROT_BASE = -0.19; // slight 3/4 turn (~-11°)
const ROT_AMP = (5 * Math.PI) / 180; // breathing yaw: ±5°
const STATIC_T = 2.1; // composed frame for prefers-reduced-motion

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

/** Deterministic pseudo-random (no Math.random → SSR/hydration safe). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Particle {
  left: number;
  top: number;
  dx: number;
  dy: number;
  dur: number;
  delay: number;
  op: number;
  warm: boolean;
}

const PARTICLES: Particle[] = (() => {
  const rnd = mulberry32(20261011);
  return Array.from({ length: 60 }, () => ({
    left: +(rnd() * 100).toFixed(2),
    top: +(rnd() * 100).toFixed(2),
    dx: +(rnd() * 90 - 45).toFixed(1),
    dy: +(rnd() * -130 - 35).toFixed(1),
    dur: +(rnd() * 14 + 13).toFixed(1),
    delay: +(rnd() * -26).toFixed(1),
    op: +(rnd() * 0.4 + 0.22).toFixed(2),
    warm: rnd() > 0.72,
  }));
})();

const NOISE_URL =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)'/%3E%3C/svg%3E\")";

/* Per-file keyframes + card chrome (single-file spec: no extra modules). */
const CSS = `
@keyframes sf-drift {
  0% { transform: translate3d(0, 0, 0); opacity: 0; }
  14% { opacity: var(--po); }
  86% { opacity: var(--po); }
  100% { transform: translate3d(var(--dx), var(--dy), 0); opacity: 0; }
}
@keyframes sf-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.12; }
}
.sf-mono {
  font-family: var(--font-geist-mono), ui-monospace, SFMono-Regular, Menlo, monospace;
}
.sf-ui {
  font-family: var(--font-geist-sans), ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
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
/* Head geometry: base profile + facial height-map displacement        */
/* ------------------------------------------------------------------ */

type PT3 = [number, number, number];

/** Half-width of the head silhouette as a function of v (0 = crown, 1 = chin). */
const PROFILE: [number, number][] = [
  [0.0, 0.1],
  [0.03, 0.3],
  [0.07, 0.42],
  [0.12, 0.51],
  [0.19, 0.59],
  [0.28, 0.655],
  [0.38, 0.695],
  [0.48, 0.7],
  [0.56, 0.69],
  [0.64, 0.66],
  [0.72, 0.61],
  [0.79, 0.545],
  [0.85, 0.47],
  [0.9, 0.39],
  [0.94, 0.3],
  [0.97, 0.22],
  [1.0, 0.13],
];

function halfW(v: number): number {
  const t = Math.min(1, Math.max(0, v));
  for (let i = 1; i < PROFILE.length; i++) {
    const [v1, w1] = PROFILE[i];
    if (t <= v1) {
      const [v0, w0] = PROFILE[i - 1];
      const u = (t - v0) / Math.max(1e-6, v1 - v0);
      return w0 + (w1 - w0) * u;
    }
  }
  return PROFILE[PROFILE.length - 1][1];
}

function depthOf(v: number): number {
  return halfW(v) * 0.78;
}

function yOf(v: number): number {
  return -1 + 2 * v;
}

function clamp01(t: number): number {
  return Math.min(1, Math.max(0, t));
}

function g2(dx: number, sx: number, dy: number, sy: number): number {
  return Math.exp(-((dx / sx) ** 2) - ((dy / sy) ** 2));
}

/**
 * Facial height map — radial displacement applied to the front hemisphere.
 * Grid lines drawn over the displaced surface follow the topography:
 * nose ridge, brow ridge, recessed eye sockets, lips, chin, cheekbones.
 */
function featureH(x: number, y: number): number {
  let h = 0;
  // nose ridge (bridge → tip), fades into nostrils
  if (y > -0.14 && y < 0.34) {
    const up = clamp01((y + 0.14) / 0.36);
    const down = y <= 0.26 ? 1 : clamp01(1 - (y - 0.26) / 0.08);
    h += (0.03 + 0.15 * up ** 1.4) * down * Math.exp(-((x / 0.085) ** 2));
  }
  // nostril wings
  h += 0.055 * (g2(x - 0.09, 0.05, y - 0.26, 0.045) + g2(x + 0.09, 0.05, y - 0.26, 0.045));
  // brow ridge
  h += 0.05 * g2(x, 0.42, y + 0.165, 0.055);
  // eye sockets (recess)
  h -= 0.05 * (g2(x - 0.25, 0.14, y + 0.02, 0.085) + g2(x + 0.25, 0.14, y + 0.02, 0.085));
  // cheekbones
  h += 0.028 * (g2(x - 0.38, 0.16, y - 0.1, 0.12) + g2(x + 0.38, 0.16, y - 0.1, 0.12));
  // upper + lower lip, with mouth-line crease and under-lip recess
  h += 0.045 * g2(x, 0.16, y - 0.465, 0.032);
  h += 0.05 * g2(x, 0.14, y - 0.555, 0.038);
  h -= 0.035 * g2(x, 0.15, y - 0.508, 0.011);
  h -= 0.02 * g2(x, 0.12, y - 0.63, 0.035);
  // chin
  h += 0.04 * g2(x, 0.18, y - 0.82, 0.11);
  return h;
}

/** Ear bump at the sides (θ ≈ ±90°, v ≈ 0.42–0.62). */
function earBump(th: number, v: number): number {
  const dth = Math.abs(Math.abs(th) - Math.PI / 2);
  if (dth > 0.5 || v < 0.36 || v > 0.68) return 0;
  return 0.085 * Math.exp(-((dth / 0.22) ** 2)) * Math.exp(-(((v - 0.52) / 0.1) ** 2));
}

/** Surface point above head-space (x, y) on the front hemisphere. */
function sf(x: number, y: number): PT3 {
  const v = (y + 1) / 2;
  const w = Math.max(halfW(v), 1e-4);
  const c = Math.sqrt(Math.max(0, 1 - (x / w) ** 2));
  return [x, y, (depthOf(v) + featureH(x, y)) * c];
}

/** Point on the displaced surface at spherical coords (theta, v). */
function surfTheta(th: number, v: number): PT3 {
  const y = yOf(v);
  const eb = earBump(th, v);
  const w = halfW(v) + eb;
  const c = Math.cos(th);
  const x = w * Math.sin(th);
  const h = c > 0 ? featureH(x, y) : 0;
  return [x, y, (depthOf(v) + h) * c + eb * 0.4 * Math.max(0, c)];
}

function project(
  x: number,
  y: number,
  z: number,
  rot: number,
  cx: number,
  cy: number,
  s: number,
): [number, number] {
  const c = Math.cos(rot);
  const si = Math.sin(rot);
  const xr = x * c + z * si;
  const zr = z * c - x * si;
  const k = 3.4 / (3.4 - zr);
  return [cx + xr * k * s, cy + y * k * s];
}

/* Dense grid: 29 meridians (±105°), 26 rings */
const TH_DENSE = Array.from(
  { length: 29 },
  (_, i) => -1.8326 + (i * 3.6652) / 28,
);
const RING_V = Array.from({ length: 26 }, (_, i) => 0.07 + (i * 0.88) / 25);

/** Node set: every 3rd grid intersection + seeded extras → [theta, v] pairs. */
const NODES: [number, number][] = (() => {
  const out: [number, number][] = [];
  for (let i = 1; i < TH_DENSE.length; i += 3)
    for (let j = 1; j < RING_V.length; j += 3) out.push([TH_DENSE[i], RING_V[j]]);
  const rnd = mulberry32(77);
  for (let k = 0; k < 22; k++)
    out.push([-1.45 + rnd() * 2.9, 0.1 + rnd() * 0.8]);
  return out;
})();

/** Constellation edges between nearby nodes (precomputed in head space). */
const EDGES: [number, number][] = (() => {
  const pts = NODES.map(([th, v]) => surfTheta(th, v));
  const out: [number, number][] = [];
  for (let i = 0; i < pts.length; i++) {
    let deg = 0;
    for (let j = i + 1; j < pts.length && deg < 3; j++) {
      const dx = pts[i][0] - pts[j][0];
      const dy = pts[i][1] - pts[j][1];
      const dz = pts[i][2] - pts[j][2];
      if (dx * dx + dy * dy + dz * dz < 0.165 ** 2) {
        out.push([i, j]);
        deg++;
      }
    }
  }
  return out;
})();

const MESH_ALPHA = [0.06, 0.12, 0.2, 0.3, 0.44, 0.58];

/* ------------------------------------------------------------------ */
/* Canvas painting                                                     */
/* ------------------------------------------------------------------ */

/** Fitted glowing ellipse between two surface anchors (screen space). */
function zoneEllipse(
  ctx: CanvasRenderingContext2D,
  P: (x: number, y: number, z: number) => [number, number],
  a: PT3,
  b: PT3,
  ry: number,
  alpha: number,
  scale = 1,
): void {
  const [ax, ay] = P(a[0], a[1], a[2]);
  const [bx, by] = P(b[0], b[1], b[2]);
  const cx = (ax + bx) / 2;
  const cy = (ay + by) / 2;
  const rx = (Math.hypot(bx - ax, by - ay) / 2) * scale;
  const angle = Math.atan2(by - ay, bx - ax);
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
  g.addColorStop(0, `rgba(255, 236, 150, ${0.72 * alpha})`);
  g.addColorStop(0.3, `rgba(255, 150, 40, ${0.62 * alpha})`);
  g.addColorStop(0.6, `rgba(255, 64, 32, ${0.38 * alpha})`);
  g.addColorStop(1, "rgba(214, 28, 74, 0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry * scale, 0, 0, TAU);
  ctx.fill();
  ctx.restore();
}

function neckW(y: number): number {
  return 0.19 + 0.13 * Math.min(1, (y - 1) / 0.55);
}

function render(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  t: number,
): void {
  ctx.clearRect(0, 0, w, h);

  const mobile = w < 1024;
  const headH = mobile
    ? Math.min(h * 0.5, w * 0.92)
    : Math.min(h * 0.64, w * 0.46);
  const s = (headH / 2) * (1 + 0.008 * Math.sin((TAU * t) / 4));
  const cx = mobile ? w * 0.5 : w * 0.68;
  const cy = (mobile ? h * 0.42 : h * 0.5) + 4 * Math.sin((TAU * t) / 7);
  const rot = ROT_BASE + ROT_AMP * Math.sin((TAU * t) / 8);
  const P = (x: number, y: number, z: number) => project(x, y, z, rot, cx, cy, s);
  const facing = (th: number) => Math.max(0, Math.cos(th + rot));

  const stroke3 = (pts: PT3[], color: string, lw: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = P(pts[i][0], pts[i][1], pts[i][2]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  };

  /* --- Heat zones: filled under the mesh, pulsing 0.6 → 1.0 → 0.6 --- */
  const zA = (phase: number) =>
    0.8 - 0.2 * Math.cos((TAU * (t - phase)) / 2.5);
  ctx.globalCompositeOperation = "lighter";
  {
    const ab = zA(0);
    zoneEllipse(ctx, P, sf(-0.13, -0.19), sf(-0.4, -0.175), 0.036 * s, ab);
    zoneEllipse(ctx, P, sf(0.13, -0.19), sf(0.4, -0.175), 0.036 * s, ab);
    const ae = zA(0.45);
    zoneEllipse(ctx, P, sf(-0.14, -0.02), sf(-0.36, -0.015), 0.055 * s, ae);
    zoneEllipse(ctx, P, sf(0.14, -0.02), sf(0.36, -0.015), 0.055 * s, ae);
    const al = zA(0.9);
    zoneEllipse(ctx, P, sf(-0.15, 0.505), sf(0.15, 0.505), 0.062 * s, al);
  }
  ctx.globalCompositeOperation = "source-over";

  /* --- Neck (behind the face) --------------------------------------- */
  {
    const pts: PT3[] = [];
    for (let i = 0; i <= 10; i++) {
      const y = 1.0 + (i / 10) * 0.55;
      const nw = neckW(y);
      pts.push([-nw, y, 0]);
    }
    for (let i = 10; i >= 0; i--) {
      const y = 1.0 + (i / 10) * 0.55;
      const nw = neckW(y);
      pts.push([nw, y, 0]);
    }
    stroke3(pts, "rgba(125, 249, 255, 0.3)", 1.2);
    for (let k = 1; k <= 5; k++) {
      const y = 1.0 + (k / 5) * 0.5;
      const nw = neckW(y);
      const ring: PT3[] = [];
      for (let i = 0; i <= 16; i++) {
        const th = -1.35 + (i / 16) * 2.7;
        ring.push([nw * Math.sin(th), y, nw * 0.85 * Math.cos(th)]);
      }
      stroke3(ring, `rgba(125, 249, 255, ${(0.08 + 0.2 * facing(Math.PI / 2)).toFixed(3)})`, 1);
    }
    for (const th of [-1.0, -0.5, 0, 0.5, 1.0]) {
      const line: PT3[] = [];
      for (let i = 0; i <= 8; i++) {
        const y = 1.0 + (i / 8) * 0.55;
        const nw = neckW(y);
        line.push([nw * Math.sin(th), y, nw * 0.85 * Math.cos(th)]);
      }
      stroke3(line, `rgba(125, 249, 255, ${(0.06 + 0.22 * facing(th)).toFixed(3)})`, 1);
    }
  }

  /* --- Dense wireframe mesh, batched into 6 facing-alpha buckets ---- */
  {
    const buckets: number[][] = Array.from({ length: 6 }, () => []);
    const pushSeg = (th: number, a: PT3, b: PT3) => {
      const f = facing(th);
      const bi = Math.min(5, Math.floor(f * 6));
      const [ax, ay] = P(a[0], a[1], a[2]);
      const [bx, by] = P(b[0], b[1], b[2]);
      buckets[bi].push(ax, ay, bx, by);
    };
    for (const th of TH_DENSE) {
      for (let i = 0; i < 44; i++) {
        pushSeg(th, surfTheta(th, i / 44), surfTheta(th, (i + 1) / 44));
      }
    }
    for (const lv of RING_V) {
      for (let i = 0; i < 36; i++) {
        const t0 = -1.8326 + (i / 36) * 3.6652;
        const t1 = -1.8326 + ((i + 1) / 36) * 3.6652;
        pushSeg((t0 + t1) / 2, surfTheta(t0, lv), surfTheta(t1, lv));
      }
    }
    ctx.lineWidth = 1;
    for (let bi = 0; bi < 6; bi++) {
      const segs = buckets[bi];
      if (!segs.length) continue;
      ctx.strokeStyle = `rgba(125, 249, 255, ${MESH_ALPHA[bi]})`;
      ctx.beginPath();
      for (let i = 0; i < segs.length; i += 4) {
        ctx.moveTo(segs[i], segs[i + 1]);
        ctx.lineTo(segs[i + 2], segs[i + 3]);
      }
      ctx.stroke();
    }
  }

  /* --- Silhouette + ears + facial line work ------------------------- */
  for (const side of [Math.PI / 2, -Math.PI / 2]) {
    const pts: PT3[] = [];
    for (let i = 0; i <= 48; i++) pts.push(surfTheta(side, i / 48));
    stroke3(pts, "rgba(125, 249, 255, 0.55)", 1.3);
  }
  for (const side of [-1, 1]) {
    const ear: PT3[] = [];
    const inner: PT3[] = [];
    const th = side * (Math.PI / 2 - 0.06);
    for (let i = 0; i <= 22; i++) {
      const a = (i / 22) * TAU;
      const v = 0.52 + 0.105 * Math.sin(a);
      const dth = 0.16 * Math.cos(a);
      ear.push(surfTheta(th + dth * side, v));
      inner.push(surfTheta(th + dth * side * 0.9, 0.52 + 0.065 * Math.sin(a)));
    }
    const f = 0.15 + 0.4 * facing(side * Math.PI / 2);
    stroke3(ear, `rgba(125, 249, 255, ${f.toFixed(3)})`, 1.2);
    stroke3(inner, `rgba(125, 249, 255, ${(f * 0.7).toFixed(3)})`, 1);
  }

  for (const side of [-1, 1]) {
    const brow: PT3[] = [];
    for (let i = 0; i <= 10; i++) {
      const u = i / 10;
      brow.push(
        sf(side * (0.13 + 0.27 * u), -0.185 - 0.06 * Math.sin(Math.PI * u)),
      );
    }
    stroke3(brow, "rgba(185, 252, 248, 0.75)", 1.2);
  }
  for (const side of [-1, 1]) {
    const upper: PT3[] = [];
    const lower: PT3[] = [];
    for (let i = 0; i <= 12; i++) {
      const u = i / 12;
      const x = side * 0.25 + (u - 0.5) * 0.22;
      upper.push(sf(x, -0.02 - 0.055 * Math.sin(Math.PI * u)));
      lower.push(sf(x, -0.02 + 0.048 * Math.sin(Math.PI * u)));
    }
    stroke3(upper, "rgba(185, 252, 248, 0.75)", 1.2);
    stroke3(lower, "rgba(185, 252, 248, 0.75)", 1.2);
    const [px, py] = P(...sf(side * 0.25, -0.02));
    ctx.fillStyle = "rgba(210, 255, 250, 0.7)";
    ctx.beginPath();
    ctx.arc(px, py, 0.022 * s, 0, TAU);
    ctx.fill();
  }
  {
    const bridge: PT3[] = [
      sf(0, -0.05),
      sf(0.006, 0.08),
      sf(0.012, 0.18),
      sf(0.008, 0.24),
    ];
    stroke3(bridge, "rgba(185, 252, 248, 0.7)", 1.2);
    const base: PT3[] = [];
    for (let i = 0; i <= 8; i++) {
      const u = i / 8;
      base.push(sf(-0.085 + 0.17 * u, 0.25 + 0.035 * Math.sin(Math.PI * u)));
    }
    stroke3(base, "rgba(185, 252, 248, 0.7)", 1.2);
    for (const side of [-1, 1])
      stroke3(
        [sf(side * 0.075, 0.245), sf(side * 0.108, 0.228)],
        "rgba(125, 249, 255, 0.55)",
        1,
      );
  }
  {
    const lip: PT3[] = [];
    for (let i = 0; i <= 14; i++) {
      const u = i / 14;
      const x = -0.16 + 0.32 * u;
      lip.push(sf(x, 0.51 - 0.03 * (1 - (x / 0.16) ** 2)));
    }
    stroke3(lip, "rgba(185, 252, 248, 0.75)", 1.2);
    const up: PT3[] = [];
    const lo: PT3[] = [];
    for (let i = 0; i <= 10; i++) {
      const u = i / 10;
      const x = -0.13 + 0.26 * u;
      up.push(sf(x, 0.468 - 0.012 * Math.sin(Math.PI * u)));
      lo.push(sf(x, 0.552 + 0.022 * Math.sin(Math.PI * u)));
    }
    stroke3(up, "rgba(125, 249, 255, 0.55)", 1);
    stroke3(lo, "rgba(125, 249, 255, 0.55)", 1);
  }

  /* --- Constellation edges + twinkling nodes ------------------------ */
  {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(100, 240, 235, 0.13)";
    ctx.beginPath();
    for (const [i, j] of EDGES) {
      const [thA, vA] = NODES[i];
      const [thB, vB] = NODES[j];
      const [ax, ay] = P(...surfTheta(thA, vA));
      const [bx, by] = P(...surfTheta(thB, vB));
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();

    let n = 0;
    for (const [th, v] of NODES) {
      const [nx, ny] = P(...surfTheta(th, v));
      const pulse = 0.5 + 0.5 * Math.sin(t * 2.1 + (n * 0.618) * TAU);
      const a = 0.55 + 0.45 * pulse;
      const r = 1.5 * (1 + 0.5 * pulse);
      ctx.fillStyle = `rgba(0, 240, 255, ${a * 0.25})`;
      ctx.beginPath();
      ctx.arc(nx, ny, r * 2.7, 0, TAU);
      ctx.fill();
      ctx.fillStyle = `rgba(200, 255, 252, ${a})`;
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, TAU);
      ctx.fill();
      n++;
    }
  }

  /* --- Iris scanning rings (rotating dashed borders) ---------------- */
  for (const side of [-1, 1]) {
    const [ex, ey] = P(...sf(side * 0.25, -0.02));
    const r = 0.095 * s;
    ctx.save();
    ctx.setLineDash([r * 0.8, r * 0.55]);
    ctx.lineDashOffset = -t * r * 3.2;
    ctx.strokeStyle = "rgba(0, 240, 255, 0.85)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(ex, ey, r, 0, TAU);
    ctx.stroke();
    ctx.restore();
  }

  /* --- Zone halo: soft glow bleeding over the wireframe ------------- */
  ctx.globalCompositeOperation = "lighter";
  {
    const ab = zA(0) * 0.16;
    zoneEllipse(ctx, P, sf(-0.13, -0.19), sf(-0.4, -0.175), 0.036 * s, ab, 1.5);
    zoneEllipse(ctx, P, sf(0.13, -0.19), sf(0.4, -0.175), 0.036 * s, ab, 1.5);
    const ae = zA(0.45) * 0.16;
    zoneEllipse(ctx, P, sf(-0.14, -0.02), sf(-0.36, -0.015), 0.055 * s, ae, 1.5);
    zoneEllipse(ctx, P, sf(0.14, -0.02), sf(0.36, -0.015), 0.055 * s, ae, 1.5);
    const al = zA(0.9) * 0.16;
    zoneEllipse(ctx, P, sf(-0.15, 0.505), sf(0.15, 0.505), 0.062 * s, al, 1.5);
  }
  ctx.globalCompositeOperation = "source-over";

  /* --- Scan line (4s top→bottom sweep) ------------------------------ */
  {
    const p = (t % 4) / 4;
    const env = Math.min(1, Math.sin(Math.PI * p) * 1.8);
    const y = cy - s * 1.12 + p * s * 2.24;
    const span = s * 0.95;
    ctx.globalCompositeOperation = "lighter";
    const band = ctx.createLinearGradient(0, y - 22, 0, y + 22);
    band.addColorStop(0, "rgba(0, 240, 255, 0)");
    band.addColorStop(0.5, `rgba(0, 240, 255, ${0.09 * env})`);
    band.addColorStop(1, "rgba(0, 240, 255, 0)");
    ctx.fillStyle = band;
    ctx.fillRect(cx - span, y - 22, span * 2, 44);
    const line = ctx.createLinearGradient(cx - span, 0, cx + span, 0);
    line.addColorStop(0, "rgba(0, 240, 255, 0)");
    line.addColorStop(0.5, `rgba(125, 249, 255, ${0.95 * env})`);
    line.addColorStop(1, "rgba(0, 240, 255, 0)");
    ctx.fillStyle = line;
    ctx.fillRect(cx - span, y - 0.75, span * 2, 1.5);
    ctx.globalCompositeOperation = "source-over";
  }
}

function FaceCanvas({ reduced }: { reduced: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0;
    let h = 0;
    let raf = 0;
    let start = performance.now();

    const size = () => {
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (reduced) render(ctx, w, h, STATIC_T);
    };

    const frame = (now: number) => {
      raf = 0;
      render(ctx, w, h, (now - start) / 1000);
      raf = requestAnimationFrame(frame);
    };
    const play = () => {
      if (raf || reduced || document.hidden) return;
      start = performance.now();
      raf = requestAnimationFrame(frame);
    };
    const stop = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    };
    const onVis = () => (document.hidden ? stop() : play());

    size();
    play();
    window.addEventListener("resize", size);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      stop();
      window.removeEventListener("resize", size);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [reduced]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

/* ------------------------------------------------------------------ */
/* Ranking labels                                                       */
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
      <p className="sf-mono mb-5 whitespace-nowrap text-[9px] uppercase tracking-[0.32em] text-white/30">
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
/* Icons                                                                */
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
/* Page                                                                 */
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

      {/* ---------- Background: dots, glows, particles ---------- */}
      <motion.div
        aria-hidden
        className="absolute inset-0 z-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: EASE }}
      >
        <div
          className="absolute inset-0 opacity-[0.15]"
          style={{
            backgroundImage:
              "radial-gradient(rgba(255,255,255,0.55) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />
        <div className="absolute -left-40 -top-48 h-[560px] w-[560px] rounded-full bg-[#00F0FF] opacity-[0.15] blur-[120px]" />
        <div className="absolute -bottom-52 -right-44 h-[620px] w-[620px] rounded-full bg-[#0A4A8A] opacity-[0.15] blur-[120px]" />
        {PARTICLES.map((p, i) => (
          <span
            key={i}
            className="absolute rounded-full"
            style={
              {
                left: `${p.left}%`,
                top: `${p.top}%`,
                width: 2,
                height: 2,
                background: p.warm ? "#FFB35C" : "#00F0FF",
                boxShadow: p.warm
                  ? "0 0 6px rgba(255,179,92,0.8)"
                  : "0 0 6px rgba(0,240,255,0.8)",
                animation: `sf-drift ${p.dur}s linear ${p.delay}s infinite`,
                "--dx": `${p.dx}px`,
                "--dy": `${p.dy}px`,
                "--po": p.op,
              } as React.CSSProperties
            }
          />
        ))}
      </motion.div>

      {/* ---------- Animated wireframe face ---------- */}
      <div
        aria-hidden
        className="absolute inset-0 z-[5] opacity-30 lg:opacity-100"
      >
        <motion.div
          className="absolute inset-0"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, ease: EASE }}
        >
          <FaceCanvas reduced={reduced} />
        </motion.div>
      </div>

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
                    minLength={6}
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
                    className="sf-mono text-[10px] uppercase tracking-[0.24em] text-white/40 transition hover:text-[#00F0FF]"
                  >
                    Provision an account →
                  </Link>
                  <span className="sf-mono text-[10px] tracking-[0.2em] text-white/25">
                    local
                  </span>
                </div>
              </form>
            </div>
          </div>

          <p className="sf-mono mt-6 text-center text-[10px] tracking-[0.22em] text-white/30 lg:text-left">
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
