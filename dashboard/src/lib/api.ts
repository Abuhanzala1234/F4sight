/** Typed API client (BUILD_SPEC §8). No `any`, per CLAUDE.md. */

import type {
  AlertDetail,
  AlertPage,
  Camera,
  Health,
  Session,
  Site,
  StreamInfo,
  Verification,
  Zone,
} from '@/types';

const BASE = '/api/v1';
const STORAGE_KEY = 'ibvap.session';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    // A corrupt entry must not brick the login screen.
    return null;
  }
}

export function saveSession(session: Session): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  localStorage.removeItem(STORAGE_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const session = loadSession();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((init.headers as Record<string, string>) ?? {}),
  };
  if (session) headers['Authorization'] = `Bearer ${session.access_token}`;

  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch (cause) {
    // Distinguish "the API said no" from "the API is not there". An operator
    // needs to know whether the system is refusing or absent.
    throw new ApiError(0, `cannot reach the API — is it running? (${String(cause)})`);
  }

  if (response.status === 401) {
    // A 401 with no prior session is a REJECTED LOGIN (wrong username or
    // password) -- the backend's own detail already says so correctly. Only a
    // 401 on a call that carried a token is an actual expired/invalid
    // session. Collapsing both into "session expired" (as this used to do)
    // reports a typo'd password as a timeout, which sends someone hunting for
    // a bug that was actually just a wrong keystroke.
    if (session) {
      clearSession();
      throw new ApiError(401, 'session expired — sign in again');
    }
    let detail = 'invalid username or password';
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; the generic message will do */
    }
    throw new ApiError(401, detail);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; the status text will do */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  async login(username: string, password: string): Promise<Session> {
    const session = await request<Session>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    saveSession(session);
    return session;
  },

  health: () => request<Health>('/health'),
  sites: () => request<Site[]>('/sites'),
  cameras: (siteId?: string) =>
    request<Camera[]>(`/cameras${siteId ? `?site_id=${encodeURIComponent(siteId)}` : ''}`),
  stream: (cameraId: string) => request<StreamInfo>(`/cameras/${cameraId}/stream`),
  connectCamera: (cameraId: string, ip: string, port = 8080, path = 'h264_ulaw.sdp') =>
    request<Camera>(`/cameras/${cameraId}/connect`, {
      method: 'POST',
      body: JSON.stringify({ ip, port, path }),
    }),
  disconnectCamera: (cameraId: string) =>
    request<Camera>(`/cameras/${cameraId}/disconnect`, { method: 'POST' }),
  zones: (cameraId: string) => request<Zone[]>(`/cameras/${cameraId}/zones`),

  alerts: (params: Record<string, string> = {}) => {
    const query = new URLSearchParams(params).toString();
    return request<AlertPage>(`/alerts${query ? `?${query}` : ''}`);
  },
  alert: (id: string) => request<AlertDetail>(`/alerts/${id}`),
  acknowledge: (id: string) => request<unknown>(`/alerts/${id}/ack`, { method: 'POST' }),
  adjudicate: (id: string, verdict: string, note?: string) =>
    request<unknown>(`/alerts/${id}/adjudicate`, {
      method: 'POST',
      body: JSON.stringify({ verdict, note: note ?? null }),
    }),
  verify: (id: string) => request<Verification>(`/alerts/${id}/verify`),
  verifyDocument: (document: Record<string, unknown>, expectedHash?: string) =>
    request<Verification>('/verify/document', {
      method: 'POST',
      body: JSON.stringify({ document, expected_hash: expectedHash ?? null }),
    }),
  evidenceUrl: (alertId: string, itemId: string) =>
    request<{ url: string; sha256: string; kind: string; expires_in: number }>(
      `/alerts/${alertId}/evidence/${itemId}`,
    ),
};
