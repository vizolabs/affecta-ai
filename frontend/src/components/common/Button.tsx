import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "ghost" | "danger";

const styles: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:opacity-90 disabled:opacity-40",
  ghost:
    "bg-transparent border border-border text-foreground hover:bg-surface disabled:opacity-40",
  danger:
    "bg-red-600/90 text-white hover:bg-red-600 disabled:opacity-40",
};

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

export default function Button({
  variant = "primary",
  className = "",
  ...props
}: Props) {
  return (
    <button
      aria-label={props["aria-label"] ?? (props.title as string)}
      {...props}
      className={`rounded-md px-3 py-2 text-sm font-medium transition
        focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent
        ${styles[variant]} ${className}`}
    />
  );
}
