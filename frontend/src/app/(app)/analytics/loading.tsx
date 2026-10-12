export default function Loading() {
  return (
    <div className="mx-auto max-w-7xl space-y-4 p-4" aria-busy="true">
      <div className="h-8 w-48 animate-pulse rounded bg-border/50" />
      <div className="grid gap-4 lg:grid-cols-4">
        <div className="h-64 animate-pulse rounded-lg bg-border/40 lg:col-span-1" />
        <div className="h-64 animate-pulse rounded-lg bg-border/40 lg:col-span-3" />
      </div>
    </div>
  );
}
