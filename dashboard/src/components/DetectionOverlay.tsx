import { useEffect, useRef } from 'react';
import type { Camera, LiveTrack } from '@/types';
import { loadSession } from '@/lib/api';
import { LiveTrackSocket } from '@/lib/ws';

/**
 * Real detection overlay for the live wall -- draws the ACTUAL tracker
 * output from worker/src/ibvap_worker/pipeline.py's per-frame `_on_tracks`
 * publish (see api/routers/ws.py's `/ws/live/{camera_id}`), not a simulation.
 * Boxes are class-coloured (person = ice/blue, vehicle = signal/amber, same
 * palette as the rest of the app) with a real track id and a speed reading
 * computed from the object's own measured motion (Track.speed_px_s).
 *
 * On alignment, since two separate things used to break it:
 *
 * 1. Scale. Boxes are in the worker's own frame pixel space and the tile
 *    renders the video with object-fit: cover, so they need the same
 *    scale-and-crop the browser applied. That needs the TRUE frame size,
 *    which now travels on every WebSocket frame (`frame_w`/`frame_h`). It
 *    used to read camera.resolution_w/h -- a seed-time 1280x720 default that
 *    nothing updates on connect, so any camera that was not actually 720p
 *    (a phone in portrait, most obviously) had every box transformed wrong.
 *
 * 2. Time. Boxes arrive over a near-instant WebSocket, so if the video is
 *    delayed the boxes lead the picture no matter how right the maths is.
 *    CameraTile plays WebRTC (a few hundred ms) in preference to HLS
 *    (seconds), which closes most of that gap. A tile that has fallen back
 *    to HLS will still show the lead -- that is the fallback being visible,
 *    not this overlay being wrong, and the tile labels which transport it
 *    got (RTC/HLS) so the difference is never a mystery.
 */

const CLASS_COLOR: Record<string, string> = {
  person: '#4FC3F7', // ice
  vehicle: '#FFB020', // signal
  animal: '#35E07F', // phosphor
  bag: '#FF8A3D', // ember
};

// Same alarm red the rest of the app already uses for a fired alert -- a
// weapon box needs to read as "the threat", not just "another tracked
// object", so it does not share the class palette above.
const WEAPON_COLOR = '#F16565';

function colorFor(cls: string): string {
  return CLASS_COLOR[cls] ?? '#C9D1D9';
}

export function DetectionOverlay({ active, camera }: { active: boolean; camera: Camera }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tracksRef = useRef<LiveTrack[]>([]);
  const lastFrameAtRef = useRef<number>(0);
  // The frame size the boxes were actually measured in, as reported by the
  // worker on every frame. Seeded from the camera row only so the very first
  // paint has something; every real frame overwrites it.
  const srcSizeRef = useRef<[number, number]>([camera.resolution_w, camera.resolution_h]);

  useEffect(() => {
    if (!active) return undefined;
    const session = loadSession();
    if (!session) return undefined;

    const socket = new LiveTrackSocket(camera.id, session.access_token, (frame) => {
      tracksRef.current = frame.tracks;
      if (frame.frame_w && frame.frame_h) {
        srcSizeRef.current = [frame.frame_w, frame.frame_h];
      }
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

    // Deliberately NOT the bracket style above -- brackets mean "a tracked
    // object", a solid box means "the specific threat region inside it",
    // matching how a weapon detector's own reference imagery draws it (a
    // small solid box on the weapon, nested inside the person's own box).
    function drawSolidBox(x: number, y: number, w: number, h: number, label: string) {
      ctx!.strokeStyle = WEAPON_COLOR;
      ctx!.lineWidth = 2;
      ctx!.strokeRect(x, y, w, h);

      const fontPx = Math.max(9, Math.round(h * 0.5));
      ctx!.font = `700 ${fontPx}px "IBM Plex Mono", monospace`;
      const textW = ctx!.measureText(label).width;
      const padX = 4;
      const chipH = fontPx + 6;
      // Chip sits above the box, flush left with it -- if that would run off
      // the top of the tile, drop it just inside the box instead.
      const chipY = y - chipH >= 0 ? y - chipH : y;
      ctx!.fillStyle = WEAPON_COLOR;
      ctx!.fillRect(x, chipY, textW + padX * 2, chipH);
      ctx!.fillStyle = '#0B0D10';
      ctx!.textBaseline = 'middle';
      ctx!.fillText(label, x + padX, chipY + chipH / 2 + 1);
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
      //
      // srcW/srcH come from the worker's own frame, NOT camera.resolution_w/h:
      // that column is a seed-time default (1280x720) that nothing updates
      // when a real camera connects, so trusting it put every box on a
      // non-720p camera through the wrong transform entirely.
      const [srcW, srcH] = srcSizeRef.current;
      if (!srcW || !srcH) {
        raf = requestAnimationFrame(frame);
        return;
      }
      const scale = Math.max(w / srcW, h / srcH);
      const offX = (w - srcW * scale) / 2;
      const offY = (h - srcH * scale) / 2;

      for (const track of tracksRef.current) {
        const [x1, y1, x2, y2] = track.box;
        const bx = x1 * scale + offX;
        const by = y1 * scale + offY;
        const bw = (x2 - x1) * scale;
        const bh = (y2 - y1) * scale;
        const armed = track.weapon !== null;
        // Armed reads in alarm red even for a "person" box -- the colour is
        // saying "this track is a live threat", which outranks the class
        // colour underneath it.
        const color = armed ? WEAPON_COLOR : colorFor(track.cls);

        drawBracketBox(bx, by, bw, bh, color);

        const speed = (track.speed_px_s / 40).toFixed(1); // px/s -> cosmetic m/s-ish scale
        const label = armed
          ? `${track.cls.toUpperCase()}_WITH_${(track.weapon as NonNullable<typeof track.weapon>).weapon_type.toUpperCase()} #${track.track_id}`
          : `${track.cls.toUpperCase()} #${track.track_id}`;
        ctx!.font = `${Math.max(9, Math.round(h * 0.012))}px "IBM Plex Mono", monospace`;
        ctx!.textBaseline = 'bottom';
        ctx!.fillStyle = color;
        ctx!.fillText(label, bx, by - 12);
        ctx!.globalAlpha = 0.75;
        ctx!.font = `${Math.max(8, Math.round(h * 0.01))}px "IBM Plex Mono", monospace`;
        ctx!.fillText(`${speed} m/s`, bx, by - 1);
        ctx!.globalAlpha = 1;

        // The weapon's OWN box, nested inside the person's -- only drawn
        // when the worker actually mapped one back to frame pixels (see
        // pipeline.py's mapped_transform.to_original call).
        if (track.weapon?.box) {
          const [wx1, wy1, wx2, wy2] = track.weapon.box;
          drawSolidBox(
            wx1 * scale + offX,
            wy1 * scale + offY,
            (wx2 - wx1) * scale,
            (wy2 - wy1) * scale,
            track.weapon.weapon_type.toUpperCase(),
          );
        }
      }

      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
    };
  }, [active]);

  if (!active) return null;
  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />;
}
