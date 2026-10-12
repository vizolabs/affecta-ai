export default function Loading() {
  return (
    <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-3" aria-busy="true">
      <div className="h-96 animate-pulse rounded-lg bg-border/40 lg:col-span-2" />
      <div className="space-y-4">
        <div className="h-24 animate-pulse rounded-lg bg-border/40" />
        <div className="h-48 animate-pulse rounded-lg bg-border/40" />
      </div>
    </div>
  );
}
