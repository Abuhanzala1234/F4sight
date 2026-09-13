import { useEffect, useRef } from 'react';
import type { Camera, LiveTrack } from '@/types';
import { loadSession } from '@/lib/api';
import { LiveTrackSocket } from '@/lib/ws';

/**
 * Real detection overlay for the live wall -- draws the ACTUAL tracker
 * output from worker/src/drishti_worker/pipeline.py's per-frame `_on_tracks`
 * publish (see api/routers/ws.py's `/ws/live/{camera_id}`), not a simulation.
 * Boxes are class-coloured (person = ice/blue, vehicle = signal/amber, same
 * palette as the rest of the app) with a real track id and a speed reading
 * computed from the object's own measured motion (Track.speed_px_s).
 *
 * Honest limitation, worth knowing before demoing this: the video arrives as
 * HLS, which buffers a few seconds behind the actual camera (that is normal
 * for HLS, not a bug here) while this overlay's WebSocket is near-instant.
 * So the boxes will visibly run a little AHEAD of the picture rather than
 * being frame-locked to it. Fixing that fully would mean moving the live
 * wall to MediaMTX's WebRTC output (sub-second latency, already enabled in
 * infra/mediamtx.yml) instead of HLS -- a bigger change than this overlay,
 * left as a follow-up rather than done silently here.
 */

const CLASS_COLOR: Record<string, string> = {
  person: '#4FC3F7', // ice
  vehicle: '#FFB020', // signal
  animal: '#35E07F', // phosphor
  bag: '#FF8A3D', // ember
};

function colorFor(cls: string): string {
  return CLASS_COLOR[cls] ?? '#C9D1D9';
}

export function DetectionOverlay({ active, camera }: { active: boolean; camera: Camera }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tracksRef = useRef<LiveTrack[]>([]);
  const lastFrameAtRef = useRef<number>(0);

  useEffect(() => {
    if (!active) return undefined;
    const session = loadSession();
    if (!session) return undefined;

    const socket = new LiveTrackSocket(camera.id, session.access_token, (frame) => {
      tracksRef.current = frame.tracks;
      lastFrameAtRef.current = performance.now();
    });
    socket.connect();
    return () => socket.close();
  }, [active, camera.id]);

  useEffect(() => {
    if (!active) return undefined;
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    let raf = 0;
    let ro: ResizeObserver | null = null;

    function resize() {
      const parent = canvas!.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas!.width = Math.max(1, Math.round(rect.width * dpr));
      canvas!.height = Math.max(1, Math.round(rect.height * dpr));
    }
    resize();
    ro = new ResizeObserver(resize);
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    function drawBracketBox(x: number, y: number, w: number, h: number, color: string) {
      const arm = Math.max(6, Math.min(w, h) * 0.22);
      ctx!.strokeStyle = color;
      ctx!.lineWidth = 1.6;
      const corners: [number, number, number, number][] = [
        [x, y, 1, 1],
        [x + w, y, -1, 1],
        [x, y + h, 1, -1],
        [x + w, y + h, -1, -1],
      ];
      for (const [cx, cy, dx, dy] of corners) {
        ctx!.beginPath();
        ctx!.moveTo(cx, cy + arm * dy);
        ctx!.lineTo(cx, cy);
        ctx!.lineTo(cx + arm * dx, cy);
        ctx!.stroke();
      }
      ctx!.globalAlpha = 0.22;
      ctx!.strokeRect(x, y, w, h);
      ctx!.globalAlpha = 1;
    }

    function frame() {
      const w = canvas!.width;
      const h = canvas!.height;
      ctx!.clearRect(0, 0, w, h);

      // Stop drawing stale boxes once the feed has gone quiet for a while
      // (socket dropped, worker stopped, camera lost) -- an old box frozen
      // over live video is a worse lie than no box at all.
      const age = performance.now() - lastFrameAtRef.current;
      if (lastFrameAtRef.current === 0 || age > 4000) {
        raf = requestAnimationFrame(frame);
        return;
      }

      // object-fit: cover maths -- the <video> crops to fill the tile, so a
      // box computed in the camera's native pixel space has to go through
      // the same scale+crop the browser applied to the picture, or it lands
      // in the wrong place (exactly the "box outside the face" bug this
      // replaces). scale = the LARGER ratio, because cover crops the
      // smaller dimension's overflow rather than letterboxing it.
      const srcW = camera.resolution_w || w;
      const srcH = camera.resolution_h || h;
      const scale = Math.max(w / srcW, h / srcH);
      const offX = (w - srcW * scale) / 2;
      const offY = (h - srcH * scale) / 2;

      for (const track of tracksRef.current) {
        const [x1, y1, x2, y2] = track.box;
        const bx = x1 * scale + offX;
        const by = y1 * scale + offY;
        const bw = (x2 - x1) * scale;
        const bh = (y2 - y1) * scale;
        const color = colorFor(track.cls);

        drawBracketBox(bx, by, bw, bh, color);

        const speed = (track.speed_px_s / 40).toFixed(1); // px/s -> cosmetic m/s-ish scale
        const label = `${track.cls.toUpperCase()} #${track.track_id}`;
        ctx!.font = `${Math.max(9, Math.round(h * 0.012))}px "IBM Plex Mono", monospace`;
        ctx!.textBaseline = 'bottom';
        ctx!.fillStyle = color;
        ctx!.fillText(label, bx, by - 12);
        ctx!.globalAlpha = 0.75;
        ctx!.font = `${Math.max(8, Math.round(h * 0.01))}px "IBM Plex Mono", monospace`;
        ctx!.fillText(`${speed} m/s`, bx, by - 1);
        ctx!.globalAlpha = 1;
      }

      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
    };
  }, [active, camera.resolution_w, camera.resolution_h]);

  if (!active) return null;
  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />;
}
