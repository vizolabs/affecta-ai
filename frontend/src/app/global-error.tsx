"use client";

/**
 * Last-resort error boundary (replaces the root layout when it itself
 * fails). Kept minimal and self-contained — no app imports.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#04070c",
          color: "#e8f1f5",
          fontFamily: "ui-sans-serif, system-ui, sans-serif",
          textAlign: "center",
        }}
      >
        <div style={{ padding: "2rem" }}>
          <p
            style={{
              fontSize: 11,
              letterSpacing: "0.3em",
              textTransform: "uppercase",
              color: "#7c93a3",
              margin: 0,
            }}
          >
            system fault
          </p>
          <h1
            style={{
              fontSize: 20,
              textTransform: "uppercase",
              letterSpacing: "0.12em",
              margin: "0.5rem 0",
            }}
          >
            Application error
          </h1>
          <p style={{ fontSize: 13, color: "#7c93a3" }}>
            The app failed to start.
            {error.digest ? ` (ref ${error.digest})` : ""}
          </p>
          <button
            type="button"
            onClick={reset}
            style={{
              marginTop: "1.25rem",
              padding: "0.5rem 1.25rem",
              background: "transparent",
              color: "#22d3ee",
              border: "1px solid #22d3ee",
              borderRadius: 6,
              fontSize: 11,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              cursor: "pointer",
            }}
          >
            Reload
          </button>
        </div>
      </body>
    </html>
  );
}
