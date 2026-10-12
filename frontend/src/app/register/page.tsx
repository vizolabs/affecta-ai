"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import AuthShell from "../../components/auth/AuthShell";
import { AuthError, AuthField, NeonButton } from "../../components/auth/AuthControls";
import { apiFetch, ApiError } from "../../lib/api";
import type { TokenOut } from "../../lib/types";
import { useAuth } from "../../stores/auth";
import { usePageTitle } from "../../hooks/usePageTitle";

export default function RegisterPage() {
  usePageTitle("Register");
  const router = useRouter();
  const setAuth = useAuth((s) => s.setAuth);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const data = await apiFetch<TokenOut>("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({
          email,
          password,
          full_name: fullName || null,
        }),
      });
      setAuth(data.token, data.user);
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Registration failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthShell
      variant="portal"
      titleLines={["Human Facial", "Emotion Detection"]}
      tagline="A Ranking-Based Explainable AI Model — evidence-ranked expression analysis, running 100% on this machine. Provision a local account to begin."
      status="Provisioning local"
    >
      <form onSubmit={submit} className="space-y-5">
        <div className="space-y-4">
          <AuthField
            id="full_name"
            label="Operator name (optional)"
            type="text"
            autoComplete="name"
            placeholder="optional"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
          />
          <AuthField
            id="email"
            label="Identity"
            type="email"
            required
            autoComplete="email"
            placeholder="you@local"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <AuthField
            id="password"
            label="Passphrase"
            type="password"
            required
            minLength={6}
            autoComplete="new-password"
            placeholder="min. 6 characters"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        <AuthError message={error} />

        <NeonButton disabled={busy}>{busy ? "Provisioning" : "Provision account"}</NeonButton>
      </form>

      <p className="mt-7 text-center text-xs text-white/40">
        Already provisioned?{" "}
        <Link
          href="/login"
          className="font-medium text-cyan-300 underline decoration-cyan-300/40 underline-offset-4 transition hover:text-cyan-200 hover:decoration-cyan-200"
        >
          Sign in
        </Link>
      </p>
    </AuthShell>
  );
}