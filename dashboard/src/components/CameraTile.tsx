import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import type { Camera } from '@/types';
import { api } from '@/lib/api';
import { StatusDot } from './Primitives';
import { DetectionOverlay } from './DetectionOverlay';

/**
 * One camera on the live wall.
 *
 * Video arrives as HLS from MediaMTX, never RTSP — no browser can play RTSP,
 * and that surprises people on demo day (blocker #4). Safari plays HLS
 * natively; everything else needs hls.js.
 */
export function CameraTile({
  camera,
  onSelect,
  onConnect,
  onDisconnect,
}: {
  camera: Camera;
  onSelect?: () => void;
  /** Slot has no live source bound yet — clicking it should ask for an IP
   * instead of trying (and failing) to open a stream. */
  onConnect?: () => void;
  /** Bound camera — tear it back down to an empty slot. */
  onDisconnect?: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<'connecting' | 'live' | 'down'>('connecting');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!camera.enabled) return undefined;
    let hls: Hls | null = null;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;

    // A camera that was just connected (ConnectCameraModal) has its row
    // flipped to enabled the moment MediaMTX *accepts* the source -- MediaMTX
    // still needs a beat to actually finish the RTSP handshake with the
    // camera and start producing segments. Landing here before that beat
    // gets a 404 on the manifest / an empty first HLS response, which used
    // to be treated as fatal and left the tile on "stream unavailable"
    // forever with no way to recover short of a full page reload. Retry with
    // backoff for a while first; only give up and show 'down' once the
    // stream has had a real chance to come up.
    const MAX_ATTEMPTS = 6;
    const RETRY_DELAY_MS = 2500;

    function fail(message: string) {
      if (cancelled) return;
      attempt += 1;
      if (attempt < MAX_ATTEMPTS) {
        retryTimer = setTimeout(connect, RETRY_DELAY_MS);
      } else {
        setState('down');
        setError(message);
      }
    }

    function connect() {
      if (cancelled) return;
      hls?.destroy();
      hls = null;
      void (async () => {
        try {
          const info = await api.stream(camera.id);
          if (cancelled || !videoRef.current) return;
          const video = videoRef.current;

          if (video.canPlayType('application/vnd.apple.mpegurl')) {
            // Native HLS (Safari, and some current Chrome builds). Wait for
            // the browser to actually confirm playable media before calling
            // this 'live' -- setting .src succeeds even when nothing is
            // published at that MediaMTX path, and a green dot over a blank
            // tile is worse than a slow one: it teaches an operator to trust
            // a dead camera.
            video.src = info.hls_url;
            video.addEventListener(
              'loadedmetadata',
              () => {
                if (!cancelled) setState('live');
              },
              { once: true },
            );
            video.addEventListener('error', () => fail('stream unavailable'), { once: true });
          } else if (Hls.isSupported()) {
            hls = new Hls({
              lowLatencyMode: true,
              backBufferLength: 10,
              maxBufferLength: 6,
              // Defaults target ~3 segments behind the live edge -- fine for
              // a VOD-style safety margin, but on a live security feed it
              // reads as several extra seconds of lag on top of whatever the
              // network is already adding. Hug the edge instead, and let it
              // catch up fast (up to 5x speed, capped once within half a
              // second of live) rather than settling into a steady-state
              // delay after any rebuffer.
              liveSyncDurationCount: 1,
              liveMaxLatencyDurationCount: 3,
              maxLiveSyncPlaybackRate: 5,
            });
            hls.loadSource(info.hls_url);
            hls.attachMedia(video);
            hls.on(Hls.Events.MANIFEST_PARSED, () => {
              if (!cancelled) setState('live');
            });
            hls.on(Hls.Events.ERROR, (_e, data) => {
              if (data.fatal) fail(data.details);
            });
          } else {
            setState('down');
            setError('HLS unsupported in this browser');
          }
        } catch (err) {
          fail(err instanceof Error ? err.message : 'stream unavailable');
        }
      })();
    }

    connect();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      hls?.destroy();
    };
  }, [camera.id, camera.enabled]);

  if (!camera.enabled) {
    return (
      <button
        type="button"
        onClick={onConnect}
        className="group relative flex aspect-video flex-col items-center justify-center gap-2
                   border border-dashed border-rule2 bg-panel/40 text-dim transition-colors
                   hover:border-signal hover:text-signal"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="h-7 w-7">
          <path d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2.6l1-1.6A1.5 1.5 0 0 1 9.4 4.7h5.2a1.5 1.5 0 0 1 1.3.7l1 1.6h2.6A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5v-9Z" />
          <circle cx="12" cy="13" r="3.6" />
          <path d="M12 11v4M10 13h4" />
        </svg>
        <span className="font-mono text-2xs uppercase tracking-[0.16em]">Connect camera</span>
        <span className="font-mono text-2xs text-dim/70">{camera.code}</span>
      </button>
    );
  }

  return (
    <div
      role={onSelect ? 'button' : undefined}
      tabIndex={onSelect ? 0 : undefined}
      onClick={onSelect}
      className="panel group relative aspect-video overflow-hidden text-left transition-colors hover:border-signal/60"
    >
      {onDisconnect && (
        <button
          type="button"
          title="Disconnect camera"
          onClick={(e) => {
            e.stopPropagation();
            onDisconnect();
          }}
          className="absolute right-2 top-2 z-10 flex h-6 w-6 items-center justify-center
                     border border-rule2 bg-void/80 font-mono text-2xs text-dim opacity-0
                     transition-opacity hover:border-alarm hover:text-alarm group-hover:opacity-100"
        >
          ✕
        </button>
      )}

      <video
        ref={videoRef}
        muted
        playsInline
        autoPlay
        className="h-full w-full bg-void object-cover opacity-90 transition-opacity group-hover:opacity-100"
      />

      <DetectionOverlay active={state === 'live'} camera={camera} />

      {state !== 'live' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-void/85">
          <span className="font-mono text-2xs uppercase tracking-[0.2em] text-dim">
            {state === 'connecting' ? 'acquiring signal' : 'no signal'}
          </span>
          {error && <span className="px-4 text-center font-mono text-2xs text-alarm">{error}</span>}
        </div>
      )}

      {/* Overlay bar. Kept to one line so it never eats the picture. */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-void via-void/80 to-transparent px-2.5 py-1.5">
        <span className="flex items-center gap-2">
          <StatusDot state={state === 'live' ? 'live' : state === 'down' ? 'down' : 'warn'} />
          <span className="font-display text-xs font-semibold tracking-[0.1em] text-bright">
            {camera.code}
          </span>
          <span className="truncate font-mono text-2xs text-dim">{camera.name}</span>
        </span>
        <span className="font-mono text-2xs tabular-nums text-dim">
          {camera.analytics_fps.toFixed(0)} fps
        </span>
      </div>
    </div>
  );
}
