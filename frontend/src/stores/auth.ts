"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "../lib/types";

interface AuthState {
  token: string | null;
  user: User | null;
  setAuth: (token: string, user: User) => void;
  clear: () => void;
}

/** One-time migration: carry the pre-rename key forward so sessions survive. */
if (typeof window !== "undefined" && !localStorage.getItem("hfd-auth")) {
  const legacy = localStorage.getItem("affecta-auth");
  if (legacy) localStorage.setItem("hfd-auth", legacy);
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      setAuth: (token, user) => set({ token, user }),
      clear: () => set({ token: null, user: null }),
    }),
    { name: "hfd-auth" },
  ),
);
