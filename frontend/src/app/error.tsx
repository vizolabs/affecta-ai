"use client";

import { useEffect } from "react";

/**
 * Route-level error boundary. Rendered inside the app layout, so a thrown
 * error on one page no longer blanks the whole session.
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // In a local-first app there is no error reporting service; surface it
    // in the console so a developer/examiner can inspect the cause.
    console.error(error);
  }, [error]);

  return (
    <main className="mx-auto flex min-h-[60vh] max-w-lg flex-col items-center justify-center gap-4 px-4 text-center">
      <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-muted">
        fault
      </p>
      <h1 className="text-xl font-semibold uppercase tracking-[0.12em] text-foreground">
        Something went wrong
      </h1>
      <p className="text-sm text-muted">
        This page hit an unexpected error.
        {error.digest ? ` (ref ${error.digest})` : ""}
      </p>
      <button
        type="button"
        onClick={reset}
        className="rounded-md border border-accent/60 bg-accent/10 px-4 py-2 font-mono text-[11px] uppercase tracking-[0.16em] text-accent transition hover:bg-accent/20"
      >
        Try again
      </button>
    </main>
  );
}
