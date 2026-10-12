import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-5 bg-background px-4 text-center">
      <span className="relative grid h-12 w-12 place-items-center" aria-hidden>
        <span className="absolute inset-0 rounded-full border border-accent/50" />
        <span className="absolute inset-[6px] rounded-full border border-accent/25" />
        <span className="h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_10px_rgba(103,232,249,0.95)]" />
      </span>
      <div>
        <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-muted">
          signal lost
        </p>
        <h1 className="mt-2 text-2xl font-semibold uppercase tracking-[0.12em] text-foreground">
          404 — Page not found
        </h1>
        <p className="mt-2 text-sm text-muted">
          That route does not exist in this build.
        </p>
      </div>
      <div className="flex gap-3">
        <Link
          href="/dashboard"
          className="rounded-md border border-accent/60 bg-accent/10 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-accent transition hover:bg-accent/20"
        >
          Dashboard
        </Link>
        <Link
          href="/live"
          className="rounded-md border border-border px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-muted transition hover:border-accent/40 hover:text-foreground"
        >
          Live Observer
        </Link>
      </div>
      <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.18em] text-muted/70">
        Human Facial Emotion Detection · 100% local · frames never leave this
        machine
      </p>
    </main>
  );
}
