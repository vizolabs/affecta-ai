import type { Envelope } from "./types";
import { useAuth } from "../stores/auth";
import { toast } from "../stores/toast";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8765";

export const WS_URL = (
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8765"
).replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

/** Endpoints where a 401 is an expected business answer (bad password, expired logout). */
function isAuthEndpoint(path: string): boolean {
  return (
    path.startsWith("/api/auth/login") ||
    path.startsWith("/api/auth/register") ||
    path.startsWith("/api/auth/logout")
  );
}

let authFailureShown = false;
/** Clear the session and land on the login portal exactly once per failure. */
export function handleAuthFailure(): void {
  if (typeof window === "undefined") return;
  useAuth.getState().clear();
  if (!authFailureShown) {
    authFailureShown = true;
    toast.error("Session expired — please sign in again.");
  }
  if (!window.location.pathname.startsWith("/login")) {
    window.location.assign("/login");
  }
}

/** Extract a readable message from FastAPI / Human Facial Emotion Detection error bodies. */
function flattenDetail(detail: unknown): string | null {
  if (typeof detail === "string" && detail.length > 0) {
    return detail.replace(/^Value error,\s*/, "");
  }
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "msg" in part) {
          return String((part as { msg: unknown }).msg).replace(
            /^Value error,\s*/,
            "",
          );
        }
        return null;
      })
      .filter((m): m is string => m !== null && m.length > 0);
    if (msgs.length > 0) return msgs.join(" ");
    return null;
  }
  if (detail && typeof detail === "object" && "msg" in detail) {
    return String((detail as { msg: unknown }).msg).replace(
      /^Value error,\s*/,
      "",
    );
  }
  return null;
}

export async function readError(res: Response): Promise<string> {
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) {
    try {
      const body = (await res.json()) as {
        detail?: unknown;
        error?: unknown;
        message?: unknown;
      } | null;
      if (body && typeof body === "object") {
        const msg =
          flattenDetail(body.detail) ??
          flattenDetail(body.error) ??
          (typeof body.message === "string" ? body.message : null);
        if (msg) return msg;
      }
    } catch {
      /* fall through to status text */
    }
  }
  return res.statusText || `HTTP ${res.status}`;
}

/** POST/GET JSON helper — unwraps the Human Facial Emotion Detection envelope, throws ApiError. */
export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  token?: string | null,
): Promise<T> {
  const headers = new Headers(init.headers);
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (
    init.body &&
    typeof init.body === "string" &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (res.status === 401 && !isAuthEndpoint(path)) {
    handleAuthFailure();
    throw new ApiError("Session expired — please sign in again.", 401);
  }
  if (!res.ok) {
    throw new ApiError(await readError(res), res.status);
  }
  const envelope = (await res.json()) as Envelope<T>;
  if (envelope.status === "error") {
    throw new ApiError(envelope.error ?? "Request failed", res.status);
  }
  return envelope.data;
}

/** Binary download (PDF/CSV) — attaches auth, returns a Blob. */
export async function apiBlob(
  path: string,
  token: string | null | undefined,
): Promise<Blob> {
  const headers: HeadersInit = token
    ? { Authorization: `Bearer ${token}` }
    : {};
  const res = await fetch(`${API_URL}${path}`, { headers });
  if (res.status === 401 && !isAuthEndpoint(path)) {
    handleAuthFailure();
    throw new ApiError("Session expired — please sign in again.", 401);
  }
  if (!res.ok) {
    throw new ApiError(await readError(res), res.status);
  }
  return res.blob();
}

export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
