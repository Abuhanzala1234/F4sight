import { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import type { Camera } from '@/types';
import { api } from '@/lib/api';
import { StatusDot } from './Primitives';

/**
 * One camera on the live wall.
 *
 * Video arrives as HLS from MediaMTX, never RTSP — no browser can play RTSP,
 * and that surprises people on demo day (blocker #4). Safari plays HLS
 * natively; everything else needs hls.js.
 */
export function CameraTile({ camera, onSelect }: { camera: Camera; onSelect?: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<'connecting' | 'live' | 'down'>('connecting');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let hls: Hls | null = null;
    let cancelled = false;

    void (async () => {
      try {
        const info = await api.stream(camera.id);
        if (cancelled || !videoRef.current) return;
        const video = videoRef.current;

        if (video.canPlayType('application/vnd.apple.mpegurl')) {
          // Native HLS (Safari, and some current Chrome builds). Wait for the
          // browser to actually confirm playable media before calling this
          // 'live' -- setting .src succeeds even when nothing is published at
          // that MediaMTX path, and a green dot over a blank tile is worse
          // than a slow one: it teaches an operator to trust a dead camera.
          video.src = info.hls_url;
          video.addEventListener('loadedmetadata', () => setState('live'), { once: true });
          video.addEventListener(
            'error',
            () => {
              setState('down');
              setError('stream unavailable');
            },
            { once: true },
          );
        } else if (Hls.isSupported()) {
          hls = new Hls({ lowLatencyMode: true, backBufferLength: 10, maxBufferLength: 6 });
          hls.loadSource(info.hls_url);
          hls.attachMedia(video);
          hls.on(Hls.Events.MANIFEST_PARSED, () => setState('live'));
          hls.on(Hls.Events.ERROR, (_e, data) => {
            if (data.fatal) {
              setState('down');
              setError(data.details);
            }
          });
        } else {
          setState('down');
          setError('HLS unsupported in this browser');
        }
      } catch (err) {
        if (!cancelled) {
          setState('down');
          setError(err instanceof Error ? err.message : 'stream unavailable');
        }
      }
    })();

    return () => {
      cancelled = true;
      hls?.destroy();
    };
  }, [camera.id]);

  return (
    <button
      type="button"
      onClick={onSelect}
      className="panel group relative aspect-video overflow-hidden text-left transition-colors hover:border-signal/60"
    >
      <video
        ref={videoRef}
        muted
        playsInline
        autoPlay
        className="h-full w-full bg-void object-cover opacity-90 transition-opacity group-hover:opacity-100"
      />

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
    </button>
  );
}
