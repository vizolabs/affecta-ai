"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { apiFetch, ApiError } from "../../lib/api";
import { useAuth } from "../../stores/auth";

/**
 * Client-side auth gate. Renders children only when a token exists AND the
 * server still accepts it (catches expired/revoked tokens before any page
 * renders a misleading empty state). Network failures do not lock the user
 * out of a local app — page-level error states handle those.
 */
export default function Protected({ children }: { children: ReactNode }) {
  const token = useAuth((s) => s.token);
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [validated, setValidated] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && !token) {
      setValidated(false);
      // "/" is the login portal — skips the extra /login -> / redirect hop.
      router.replace("/");
    }
  }, [mounted, token, router]);

  useEffect(() => {
    if (!mounted || !token) return;
    let cancelled = false;
    setValidated(false);
    apiFetch("/api/auth/me", {}, token)
      .then(() => {
        if (!cancelled) setValidated(true);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) {
          useAuth.getState().clear();
          router.replace("/");
          return;
        }
        // API unreachable — let the app render; pages surface their own errors.
        setValidated(true);
      });
    return () => {
      cancelled = true;
    };
  }, [mounted, token, router]);

  if (!mounted || !token || !validated) {
    return (
      <div className="flex min-h-screen items-center justify-center text-muted">
        Loading…
      </div>
    );
  }
  return <>{children}</>;
}
