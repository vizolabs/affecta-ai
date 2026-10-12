"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { NarrationLine } from "../../lib/types";

/** Rolling narrator commentary feed (newest at top). */
export default function CommentaryPanel({ lines }: { lines: NarrationLine[] }) {
  return (
    <div
      aria-live="polite"
      aria-label="Narrator commentary"
      className="h-40 overflow-y-auto rounded-md border border-border bg-background p-2"
    >
      {lines.length === 0 && (
        <p className="text-xs text-muted">
          Narration appears here when events occur…
        </p>
      )}
      <AnimatePresence initial={false}>
        {lines.map((line, i) => (
          <motion.p
            key={`${line.timestamp}-${i}`}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-b border-border/60 py-1.5 text-xs last:border-0"
          >
            <span className="text-muted">
              {new Date(line.timestamp * 1000).toLocaleTimeString()}
            </span>{" "}
            {line.text}
          </motion.p>
        ))}
      </AnimatePresence>
    </div>
  );
}
