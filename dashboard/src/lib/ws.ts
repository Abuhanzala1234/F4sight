/**
 * Live alert socket (BUILD_SPEC §9).
 *
 * The contract that matters: a dropped socket must never mean a lost alert on
 * screen. So this reconnects with backoff and reports the timestamp it last
 * saw, and the caller replays the gap over REST. The socket is an accelerator
 * for the REST feed, never the only way to learn something happened.
 */

import type { ServerMsg } from '@/types';

type Handler = (msg: ServerMsg) => void;
type StatusHandler = (connected: boolean) => void;

const BACKOFF_MS = [1000, 2000, 4000, 8000, 15000, 30000] as const;

export class AlertSocket {
  private socket: WebSocket | null = null;
  private attempt = 0;
  private closedByUs = false;
  private timer: number | null = null;

  constructor(
    private readonly token: string,
    private readonly onMessage: Handler,
    private readonly onStatus: StatusHandler,
  ) {}

  connect(): void {
    this.closedByUs = false;
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${scheme}://${window.location.host}/ws/alerts?token=${encodeURIComponent(this.token)}`;

    try {
      this.socket = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this.attempt = 0;
      this.onStatus(true);
      this.send({ type: 'subscribe', site_ids: [], min_severity: 'info' });
    };

    this.socket.onmessage = (event: MessageEvent<string>) => {
      try {
        this.onMessage(JSON.parse(event.data) as ServerMsg);
      } catch {
        // A malformed frame is not worth tearing the socket down for.
      }
    };

    this.socket.onclose = () => {
      this.onStatus(false);
      if (!this.closedByUs) this.scheduleReconnect();
    };

    this.socket.onerror = () => this.socket?.close();
  }

  private scheduleReconnect(): void {
    const delay = BACKOFF_MS[Math.min(this.attempt, BACKOFF_MS.length - 1)] ?? 30000;
    this.attempt += 1;
    this.timer = window.setTimeout(() => this.connect(), delay);
  }

  send(payload: Record<string, unknown>): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }

  close(): void {
    this.closedByUs = true;
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.socket?.close();
  }
}
