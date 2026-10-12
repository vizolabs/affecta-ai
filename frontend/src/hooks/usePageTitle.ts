"use client";

import { useEffect } from "react";

/** Sets document.title per page (client pages can't export Next Metadata). */
export function usePageTitle(page: string): void {
  useEffect(() => {
    const prev = document.title;
    document.title = `${page} · Human Facial Emotion Detection`;
    return () => {
      document.title = prev;
    };
  }, [page]);
}
