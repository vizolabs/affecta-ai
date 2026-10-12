"use client";

import { useToast } from "../../stores/toast";

const KIND_STYLE = {
  info: "border-border bg-surface text-foreground",
  success: "border-emerald-500/50 bg-emerald-500/10 text-emerald-300",
  error: "border-red-500/50 bg-red-500/10 text-red-300",
} as const;

/** Global toast stack (mounted once in the app layout). */
export default function Toaster() {
  const { toasts, dismiss } = useToast();
  if (toasts.length === 0) return null;
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-72 flex-col gap-2"
    >
      {toasts.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => dismiss(t.id)}
          className={`pointer-events-auto rounded-md border px-3 py-2 text-left text-sm shadow-lg ${KIND_STYLE[t.kind]}`}
        >
          {t.text}
        </button>
      ))}
    </div>
  );
}
