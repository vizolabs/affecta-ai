"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

type Theme = "dark" | "light";

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") return;
  document.documentElement.classList.toggle("light", theme === "light");
  document.documentElement.classList.toggle("dark", theme !== "light");
}

/** One-time migration: carry the pre-rename key forward so sessions survive. */
if (typeof window !== "undefined" && !localStorage.getItem("hfd-theme")) {
  const legacy = localStorage.getItem("affecta-theme");
  if (legacy) localStorage.setItem("hfd-theme", legacy);
}

export const useTheme = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: "dark",
      setTheme: (theme) => {
        applyTheme(theme);
        set({ theme });
        void get();
      },
    }),
    {
      name: "hfd-theme",
      onRehydrateStorage: () => (state) => {
        if (state) applyTheme(state.theme);
      },
    },
  ),
);
