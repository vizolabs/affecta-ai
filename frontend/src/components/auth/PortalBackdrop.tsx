"use client";

import { useEffect, useRef } from "react";

/**
 * Procedural "analysed face" portal backdrop for the auth screens.
 *
 * Original artwork drawn live in code (no image asset): a particle wireframe
 * face built from parametric curves, ringed by a rotating sci-fi portal, swept
 * by a scan beam, with warm analysis zones and abstract region bars. The HUD
 * numbers-free by design — telemetry here is ambient ornament, never
 * fabricated metrics.
 *
 * Performance: one rAF loop, DPR clamped at 2, ~700 draw ops/frame. Sleeps
 * when the tab is hidden; renders a single static frame when the user prefers
 * reduced motion.
 */

const CYAN = "103,232,249";
const CYAN_DEEP = "34,211,238";
const AMBER = "245,158,11";

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Pt {
  x: number;
  y: number;
}

interface Feature {
  pts: Pt[];
}

/** Sample the abstract face as parametric curves. */
function buildFace(cx: number, cy: number, R: number): Feature[] {
  const features: Feature[] = [];
  const ell = (
    ex: number,
    ey: number,
    rx: number,
    ry: number,
    n: number,
    a0 = 0,
    a1 = Math.PI * 2,
  ): Pt[] => {
    const out: Pt[] = [];
    for (let i = 0; i < n; i++) {
      const a = a0 + ((a1 - a0) * i) / Math.max(1, n - 1);
      out.push({ x: ex + Math.cos(a) * rx, y: ey + Math.sin(a) * ry });
    }
    return out;
  };

  // Head oval.
  features.push({ pts: ell(cx, cy, R, R * 1.28, 170) });

  const browY = cy - 0.16 * R;
  const eyeY = cy + 0.02 * R;
  const eyeDX = 0.38 * R;
  for (const s of [-1, 1]) {
    // Brow arc.
    const brow: Pt[] = [];
    for (let i = 0; i <= 22; i++) {
      const t = i / 22 - 0.5;
      brow.push({
        x: cx + s * eyeDX + t * 0.34 * R,
        y: browY - Math.cos(t * Math.PI) * 0.035 * R - 0.02 * R,
      });
    }
    features.push({ pts: brow });
    // Eye ellipse.
    features.push({ pts: ell(cx + s * eyeDX, eyeY, 0.13 * R, 0.075 * R, 26) });
  }

  // Nose bridge + base.
  const nose: Pt[] = [];
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    nose.push({ x: cx + t * 0.03 * R, y: cy - 0.05 * R + t * 0.33 * R });
  }
  nose.push({ x: cx - 0.05 * R, y: cy + 0.28 * R });
  nose.push({ x: cx + 0.08 * R, y: cy + 0.28 * R });
  features.push({ pts: nose });

  // Mouth ellipse.
  features.push({ pts: ell(cx, cy + 0.46 * R, 0.22 * R, 0.085 * R, 30) });

  return features;
}

interface Layout {
  w: number;
  h: number;
  cx: number;
  cy: number;
  R: number;
  desktop: boolean;
}

function computeLayout(w: number, h: number): Layout {
  const desktop = w >= 1024;
  const R = Math.min(w, h) * (desktop ? 0.3 : 0.32);
  const cx = desktop ? w * 0.7 : w * 0.5;
  const cy = Math.min(Math.max(h * (desktop ? 0.44 : 0.4), R * 1.7), h - R * 1.8);
  return { w, h, cx, cy, R, desktop };
}

export default function PortalBackdrop() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const rng = mulberry32(20261011);

    let W = 0;
    let H = 0;
    let layout: Layout = computeLayout(1, 1);
    let strokes: Pt[][] = [];
    let nodes: { p: Pt; seed: number }[] = [];

    const rebuild = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      W = Math.max(1, Math.round(rect.width));
      H = Math.max(1, Math.round(rect.height));
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      layout = computeLayout(W, H);
      const feats = buildFace(layout.cx, layout.cy, layout.R);
      strokes = feats.map((f) => f.pts);
      // Landmark nodes: every ~9th point, jittered seed for pulse phase.
      nodes = [];
      let k = 0;
      for (const s of strokes) {
        for (let i = 0; i < s.length; i += 9) {
          nodes.push({ p: s[i], seed: rng() * Math.PI * 2 });
          if (++k >= 44) break;
        }
        if (k >= 44) break;
      }
    };

    const drawRings = (t: number) => {
      const { cx, cy, R } = layout;
      const rings: [number, number[], number, number][] = [
        [R * 1.55, [2, 7], 0.5, t / 60000],
        [R * 1.78, [14, 5, 2, 5], 0.32, -t / 90000],
        [R * 2.0, [1, 4], 0.22, 0],
      ];
      for (const [r, dash, alpha, rot] of rings) {
        ctx.save();
        ctx.strokeStyle = `rgba(${CYAN},${alpha})`;
        ctx.lineWidth = 1;
        ctx.setLineDash(dash);
        ctx.lineDashOffset = rot * Math.PI * 2 * r;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      // Tick marks on the outer ring.
      ctx.save();
      ctx.strokeStyle = `rgba(${CYAN},0.4)`;
      ctx.lineWidth = 1;
      for (let i = 0; i < 60; i++) {
        const a = (i / 60) * Math.PI * 2;
        const long = i % 5 === 0;
        const r0 = R * 2.0 + 3;
        const r1 = r0 + (long ? 7 : 3.5);
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
        ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
        ctx.stroke();
      }
      ctx.restore();
    };

    const drawHeat = (t: number) => {
      const { cx, cy, R } = layout;
      const zones: [number, number, number, number][] = [
        [cx - 0.36 * R, cy - 0.14 * R, R * 0.3, 0], // brow
        [cx + 0.38 * R, cy + 0.02 * R, R * 0.26, 2.1], // eye
        [cx, cy + 0.46 * R, R * 0.34, 4.2], // mouth
      ];
      for (const [x, y, r, phase] of zones) {
        const breathe = 0.5 + 0.5 * Math.sin((t / 5000) * Math.PI * 2 + phase);
        const alpha = 0.09 + breathe * 0.15;
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, `rgba(${AMBER},${alpha})`);
        g.addColorStop(1, `rgba(${AMBER},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const drawFace = (scanY: number) => {
      // Strokes.
      ctx.save();
      ctx.strokeStyle = `rgba(${CYAN},0.52)`;
      ctx.lineWidth = 1.1;
      for (const s of strokes) {
        ctx.beginPath();
        s.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.stroke();
      }
      ctx.restore();
      // Dots.
      ctx.save();
      ctx.fillStyle = `rgba(${CYAN_DEEP},0.95)`;
      for (const s of strokes) {
        for (const p of s) ctx.fillRect(p.x - 1, p.y - 1, 2, 2);
      }
      ctx.restore();
      // Landmark nodes; pulse as the scan beam passes.
      for (const { p, seed } of nodes) {
        const d = Math.abs(scanY - p.y);
        const lock = Math.max(0, 1 - d / 26);
        const tw = 0.5 + 0.5 * Math.sin(seed + performance.now() / 900);
        const r = 2.5 + lock * 3 + tw * 0.8;
        const a = 0.35 + lock * 0.55;
        ctx.save();
        ctx.strokeStyle = `rgba(${CYAN},${a})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(224,251,255,${0.5 + lock * 0.5})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.3, 0, Math.PI * 2);
        ctx.fill();
        // Crosshair for locked nodes.
        if (lock > 0.55) {
          ctx.strokeStyle = `rgba(${CYAN},${lock})`;
          ctx.beginPath();
          ctx.moveTo(p.x - r - 3, p.y);
          ctx.lineTo(p.x - r + 1, p.y);
          ctx.moveTo(p.x + r - 1, p.y);
          ctx.lineTo(p.x + r + 3, p.y);
          ctx.stroke();
        }
        ctx.restore();
      }
    };

    const drawScan = (t: number): number => {
      const { cx, cy, R } = layout;
      const span = R * 1.7;
      const phase = (t % 7000) / 7000;
      const y = cy - span + phase * span * 2;
      // Trail.
      const trail = ctx.createLinearGradient(0, y - 70, 0, y);
      trail.addColorStop(0, `rgba(${CYAN_DEEP},0)`);
      trail.addColorStop(1, `rgba(${CYAN_DEEP},0.10)`);
      ctx.fillStyle = trail;
      ctx.fillRect(cx - R * 2.1, y - 70, R * 4.2, 70);
      // Beam.
      ctx.save();
      ctx.shadowColor = `rgba(${CYAN_DEEP},0.9)`;
      ctx.shadowBlur = 18;
      ctx.fillStyle = `rgba(207,244,255,0.85)`;
      ctx.fillRect(cx - R * 2.1, y - 1, R * 4.2, 2);
      ctx.restore();
      return y;
    };

    const drawHud = (t: number) => {
      if (!layout.desktop) return;
      const { cx, cy, R, w, h } = layout;
      const bx = Math.min(cx + R * 1.05, w - 240);
      const by = Math.min(cy + R * 0.9, h - 150);
      const rows: [string, number][] = [
        ["BROW REGION", 0],
        ["EYE REGION", 2.4],
        ["MOUTH REGION", 4.8],
      ];
      ctx.save();
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      // Dim chip behind the readout so ring lines never fight the text.
      ctx.fillStyle = "rgba(4,7,12,0.55)";
      ctx.fillRect(bx - 10, by - 26, 210, 3 * 30 + 26);
      rows.forEach(([label, phase], ri) => {
        const y = by + ri * 30;
        const cyc = (t / 1000 + phase) % 8;
        const fill = 3 + Math.round((Math.sin(cyc * 0.9) * 0.5 + 0.5) * 5); // 3..8
        ctx.fillStyle = `rgba(224,251,255,0.55)`;
        ctx.fillText(label, bx, y - 8);
        for (let s = 0; s < 8; s++) {
          const on = s < fill;
          ctx.fillStyle =
            s === 0
              ? `rgba(${AMBER},${on ? 0.9 : 0.2})`
              : `rgba(${CYAN_DEEP},${on ? 0.75 : 0.15})`;
          ctx.fillRect(bx + s * 22, y, 18, 5);
        }
      });
      ctx.fillStyle = "rgba(224,251,255,0.30)";
      ctx.fillText("ambient · illustrative", bx, by + 3 * 30 + 2);
      ctx.restore();
    };

    const frame = (t: number) => {
      ctx.clearRect(0, 0, W, H);
      drawRings(t);
      drawHeat(t);
      const scanY = drawScan(t);
      drawFace(scanY);
      drawHud(t);
    };

    rebuild();
    let raf = 0;
    let timer: number | null = null;

    const onResize = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(rebuild, 150);
    };
    window.addEventListener("resize", onResize);

    const onVis = () => {
      if (document.hidden) {
        cancelAnimationFrame(raf);
        raf = 0;
      } else if (!raf && !reduced) {
        loop();
      }
    };

    const loop = () => {
      frame(performance.now());
      raf = requestAnimationFrame(loop);
    };

    if (reduced) {
      frame(2600); // one static composition
    } else {
      document.addEventListener("visibilitychange", onVis);
      loop();
    }

    return () => {
      cancelAnimationFrame(raf);
      if (timer !== null) window.clearTimeout(timer);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return <canvas ref={ref} aria-hidden className="absolute inset-0 h-full w-full" />;
}
