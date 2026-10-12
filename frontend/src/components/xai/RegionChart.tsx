"use client";

/** Horizontal bars for Grad-CAM region mass (%). */
export default function RegionChart({
  regions,
}: {
  regions: Record<string, number> | null | undefined;
}) {
  if (!regions || Object.keys(regions).length === 0) {
    return <p className="text-xs text-muted">No region data yet.</p>;
  }
  const entries = Object.entries(regions).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <ul className="space-y-1.5">
      {entries.map(([region, value]) => (
        <li key={region} className="flex items-center gap-2 text-xs">
          <span className="w-20 shrink-0 capitalize text-muted">{region}</span>
          <div className="h-3 flex-1 overflow-hidden rounded bg-border">
            <div
              className="h-full rounded bg-accent"
              style={{ width: `${(value / max) * 100}%` }}
            />
          </div>
          <span className="w-12 shrink-0 text-right tabular-nums">
            {value.toFixed(1)}%
          </span>
        </li>
      ))}
    </ul>
  );
}
