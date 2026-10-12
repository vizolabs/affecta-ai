"use client";

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

/** Shared surface styles for the auth HUD screens. */

export const authField =
  "w-full rounded-md border border-white/15 bg-white/[0.04] px-4 py-2.5 text-sm text-white " +
  "placeholder:text-white/35 backdrop-blur-sm outline-none transition " +
  "focus:border-cyan-300/70 focus:bg-white/[0.07] " +
  "focus:shadow-[0_0_0_1px_rgba(103,232,249,0.35),0_0_22px_rgba(34,211,238,0.25)]";

export const authLabel =
  "mb-1.5 block font-mono text-[10px] uppercase tracking-[0.24em] text-white/40";

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  id: string;
}

export function AuthField({ label, id, className = "", ...props }: FieldProps) {
  return (
    <div>
      <label htmlFor={id} className={authLabel}>
        {label}
      </label>
      <input id={id} {...props} className={`${authField} ${className}`} />
    </div>
  );
}

interface AuthErrorProps {
  message: string | null;
}

/** Framed alert readout; reserves height so the layout never jumps. */
export function AuthError({ message }: AuthErrorProps) {
  return (
    <div aria-live="polite" className="min-h-[1.25rem]">
      {message && (
        <p
          role="alert"
          className="flex items-start gap-2 border-l-2 border-rose-400/70 bg-rose-500/[0.07] py-2 pl-3 pr-2 text-xs text-rose-200"
        >
          <span className="font-mono">!</span>
          {message}
        </p>
      )}
    </div>
  );
}

interface NeonButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
}

/** Bracket-cornered neon bar with a light sweep on hover. */
export function NeonButton({ children, className = "", ...props }: NeonButtonProps) {
  return (
    <div className="relative pt-1">
      <span
        aria-hidden
        className="pointer-events-none absolute -inset-px border border-cyan-300/25"
      />
      <span
        aria-hidden
        className="pointer-events-none absolute -left-px -top-1.5 h-2 w-px bg-cyan-300/60"
      />
      <span
        aria-hidden
        className="pointer-events-none absolute -right-px -top-1.5 h-2 w-px bg-cyan-300/60"
      />
      <button
        type="submit"
        {...props}
        className={`group relative w-full overflow-hidden border border-cyan-300/40 bg-cyan-400/[0.07] py-3 text-[12px] font-semibold uppercase tracking-[0.32em] text-cyan-100 transition
          hover:bg-cyan-400/[0.14] hover:text-white hover:shadow-[0_0_26px_rgba(34,211,238,0.35)]
          focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-200
          disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      >
        <span
          aria-hidden
          className="absolute inset-y-0 -left-1/3 w-1/3 skew-x-[-18deg] bg-cyan-200/20 transition-transform duration-500 group-hover:translate-x-[420%]"
        />
        <span className="relative">{children}</span>
      </button>
    </div>
  );
}