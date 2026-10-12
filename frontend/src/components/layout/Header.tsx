"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Button from "../common/Button";
import HealthChip from "../common/HealthChip";
import { apiFetch } from "../../lib/api";
import { useAuth } from "../../stores/auth";
import { useTheme } from "../../stores/theme";

const NAV = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/live", label: "Live Observer" },
  { href: "/lab", label: "Research Lab" },
  { href: "/analytics", label: "Analytics" },
];

export default function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, clear } = useAuth();
  const { theme, setTheme } = useTheme();
  const [aboutOpen, setAboutOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  // Close the mobile menu on navigation.
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  // Escape closes the About dialog (matches the Live help dialog).
  useEffect(() => {
    if (!aboutOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setAboutOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [aboutOpen]);

  const logout = async () => {
    try {
      const token = useAuth.getState().token;
      await apiFetch("/api/auth/logout", { method: "POST" }, token);
    } catch {
      // logout is best-effort locally
    }
    clear();
    router.replace("/login");
  };

  const navLinkCls = (active: boolean) =>
    `relative rounded-md px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] transition ${
      active
        ? "bg-accent/10 text-accent"
        : "text-muted hover:bg-border/40 hover:text-foreground"
    }`;

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/85 px-4 py-2.5 backdrop-blur-md">
      <div className="mx-auto flex max-w-[100rem] flex-wrap items-center gap-x-5 gap-y-2">
        <Link
          href="/dashboard"
          className="flex items-center gap-2.5"
          aria-label="Human Facial Emotion Detection — dashboard"
        >
          {/* Compact identity mark: scan-ring (shared with auth screens) */}
          <span className="relative grid h-7 w-7 place-items-center">
            <span className="absolute inset-0 rounded-full border border-accent/50" />
            <span className="absolute inset-[4px] rounded-full border border-accent/25" />
            <span className="h-1 w-1 rounded-full bg-accent shadow-[0_0_8px_rgba(103,232,249,0.95)]" />
          </span>
          <span className="text-[11px] font-semibold uppercase leading-none tracking-[0.18em] text-foreground">
            Human Facial
            <span className="ml-1 text-accent">Emotion Detection</span>
          </span>
        </Link>

        <span aria-hidden className="hidden h-5 w-px bg-border lg:block" />

        <nav aria-label="Modes" className="hidden items-center gap-1 lg:flex">
          {NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={navLinkCls(active)}
              >
                {active && (
                  <span
                    aria-hidden
                    className="absolute inset-x-2 -bottom-[7px] h-px bg-accent/70 shadow-[0_0_8px_rgba(103,232,249,0.7)]"
                  />
                )}
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <HealthChip />
          <button
            type="button"
            aria-label="Toggle theme"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-md border border-border px-2 py-1.5 text-xs text-muted transition hover:border-accent/40 hover:text-accent"
          >
            {theme === "dark" ? "☾" : "☀"}
          </button>
          <button
            type="button"
            onClick={() => setAboutOpen(true)}
            className="rounded-md px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] text-muted transition hover:bg-border/40 hover:text-foreground"
          >
            About
          </button>
          <Link
            href="/settings"
            aria-label="Settings"
            className={navLinkCls(pathname.startsWith("/settings"))}
          >
            Settings
          </Link>
          {user && (
            <span className="hidden font-mono text-[11px] text-muted lg:inline">
              {user.email}
            </span>
          )}
          <Button
            variant="ghost"
            onClick={logout}
            className="border-border text-muted"
          >
            Log out
          </Button>
          {/* Mobile menu toggle */}
          <button
            type="button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((v) => !v)}
            className="rounded-md border border-border px-2 py-1.5 text-xs text-muted transition hover:border-accent/40 hover:text-accent lg:hidden"
          >
            {menuOpen ? "✕" : "☰"}
          </button>
        </div>
      </div>
      {/* Mobile dropdown nav */}
      {menuOpen && (
        <nav
          aria-label="Modes (mobile)"
          className="mx-auto mt-2 flex max-w-[100rem] flex-col gap-1 border-t border-border pt-2 lg:hidden"
        >
          {NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={navLinkCls(active)}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      )}
      <p className="mx-auto mt-1 max-w-[100rem] font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
        Fully local · webcam frames never leave this machine · no cloud inference
      </p>

      {aboutOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="About Human Facial Emotion Detection"
          onClick={() => setAboutOpen(false)}
        >
          <div
            className="relative w-full max-w-md rounded-xl border border-accent/25 bg-surface/95 p-6 text-sm"
            style={{ boxShadow: "0 0 60px rgba(34,211,238,0.15)" }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* bracket accent */}
            <span
              aria-hidden
              className="pointer-events-none absolute -inset-px border border-accent/10"
            />
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-foreground">
                Human Facial Emotion Detection
                <span className="mt-1 block font-mono text-[10px] font-normal uppercase tracking-[0.2em] text-accent/70">
                  ranking-based explainable emotion detection
                </span>
              </h2>
              <button
                type="button"
                aria-label="Close"
                onClick={() => setAboutOpen(false)}
                className="rounded px-2 text-muted hover:text-foreground"
              >
                ✕
              </button>
            </div>
            <div className="h-px w-full bg-border" />
            <ul className="mt-4 list-disc space-y-2 pl-5 text-[13px] leading-relaxed text-muted">
              <li>
                Single-person tracking: one subject, pinned, debounce-stable
                transitions.
              </li>
              <li>
                Emotion ranking (RAF-DB, n=3,068) with Grad-CAM region evidence
                — model ships its own uncertainty.
              </li>
              <li>
                Deterministic offline narration; frames are processed in-memory
                and never stored or uploaded.
              </li>
              <li>
                Research metrics come from local experiments; AU branch
                untrained (w_AU = 0); intensity is a distribution-extremity
                proxy — not clinical.
              </li>
            </ul>
            <p className="mt-4 border-t border-border pt-3 font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
              Not a medical or psychological diagnostic tool.
            </p>
          </div>
        </div>
      )}
    </header>
  );
}
