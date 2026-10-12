"use client";

interface Props {
  text: string;
  label?: string;
}

/** Small inline help affordance (hover/focus reveals the full text). */
export default function InfoTip({ text, label }: Props) {
  return (
    <span
      tabIndex={0}
      role="note"
      aria-label={label ?? text}
      title={text}
      className="ml-1 inline-flex h-4 w-4 cursor-help select-none items-center justify-center rounded-full border border-border align-middle text-[10px] font-medium text-muted hover:border-accent hover:text-foreground"
    >
      ?
    </span>
  );
}
