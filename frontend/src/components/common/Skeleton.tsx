interface Props {
  className?: string;
}

export default function Skeleton({ className = "" }: Props) {
  return (
    <div
      aria-hidden
      className={`animate-pulse rounded-md bg-border/50 ${className}`}
    />
  );
}
