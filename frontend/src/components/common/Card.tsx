import type { HTMLAttributes, ReactNode } from "react";

interface Props extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  action?: ReactNode;
  children: ReactNode;
}

export default function Card({
  title,
  action,
  children,
  className = "",
  ...rest
}: Props) {
  return (
    <div
      {...rest}
      className={`rounded-lg border border-border bg-surface p-4 ${className}`}
    >
      {(title || action) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title ? (
            <h3 className="text-sm font-semibold uppercase tracking-wide text-muted">
              {title}
            </h3>
          ) : (
            <span />
          )}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}
