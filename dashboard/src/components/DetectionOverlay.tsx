import { useEffect, useRef } from 'react';

/**
 * Purely decorative HUD overlay for the live wall (aesthetic only — this
 * component does not read real detections; the pipeline's actual per-frame
 * output is not streamed to the browser today, only finished alerts are
 * (api/src/drishti_api/routers/ws.py). This exists so the camera wall LOOKS
 * like an analytics console is watching it, for demo purposes. It draws
 * fabricated tracks with class-coloured corner-bracket boxes and a fake
 * speed readout, seeded per camera so each tile looks distinct and stable
 * rather than randomly reshuffling every render.
 */

type SimClass = 'person' | 'vehicle';

interface SimTrack {
  id: number;
  cls: SimClass;
  x: number; // center, 0..1 of canvas width
  y: number; // center, 0..1 of canvas height
  w: number; // 0..1 of canvas width
  h: number; // 0..1 of canvas height
  vx: number; // units/sec, 0..1 space
  vy: number;
  bornAt: number;
  lifeMs: number;
  fadeMs: number;
}

const CLASS_COLOR: Record<SimClass, string> = {
  person: '#4FC3F7', // ice
  vehicle: '#FFB020', // signal
};

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function spawnTrack(rand: () => number, now: number, idSeed: number): SimTrack {
  const cls: SimClass = rand() > 0.62 ? 'vehicle' : 'person';
  const w = cls === 'vehicle' ? 0.1 + rand() * 0.08 : 0.045 + rand() * 0.03;
  const h = cls === 'vehicle' ? w * 0.55 : w * 2.4;
  const speed = 0.01 + rand() * 0.035;
  const angle = rand() * Math.PI * 2;
  return {
    id: 100 + Math.floor(idSeed * 900 + rand() * 99),
    cls,
    x: 0.12 + rand() * 0.76,
    y: 0.18 + rand() * 0.64,
    w,
    h,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed * 0.6,
    bornAt: now,
    lifeMs: 6000 + rand() * 9000,
    fadeMs: 500,
  };
}

export function DetectionOverlay({ active, seed }: { active: boolean; seed: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!active) return undefined;
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    let hashSeed = 0;
    for (let i = 0; i < seed.length; i += 1) hashSeed = (hashSeed * 31 + seed.charCodeAt(i)) | 0;
    const rand = mulberry32(hashSeed || 1);

    const now0 = performance.now();
    const tracks: SimTrack[] = Array.from({ length: 2 + Math.floor(rand() * 2) }, (_, i) =>
      spawnTrack(rand, now0 - rand() * 3000, i + 1),
    );

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
      // faint full outline so the box still reads at a glance, not just corners
      ctx!.globalAlpha = 0.22;
      ctx!.strokeRect(x, y, w, h);
      ctx!.globalAlpha = 1;
    }

    function frame(t: number) {
      const w = canvas!.width;
      const h = canvas!.height;
      ctx!.clearRect(0, 0, w, h);

      for (let i = tracks.length - 1; i >= 0; i -= 1) {
        const tr = tracks[i]!;
        const age = t - tr.bornAt;
        if (age > tr.lifeMs) {
          tracks[i] = spawnTrack(rand, t, i + 1);
          continue;
        }
        // bounce softly inside the frame rather than wrapping -- reads as
        // patrolling/loitering, which is more plausible than teleporting
        const dt = 1 / 60;
        tr.x += tr.vx * dt;
        tr.y += tr.vy * dt;
        if (tr.x < 0.05 || tr.x > 0.95) tr.vx *= -1;
        if (tr.y < 0.12 || tr.y > 0.88) tr.vy *= -1;
        tr.x = Math.min(0.95, Math.max(0.05, tr.x));
        tr.y = Math.min(0.88, Math.max(0.12, tr.y));

        const alpha =
          age < tr.fadeMs
            ? age / tr.fadeMs
            : age > tr.lifeMs - tr.fadeMs
              ? (tr.lifeMs - age) / tr.fadeMs
              : 1;
        const color = CLASS_COLOR[tr.cls];
        const bx = (tr.x - tr.w / 2) * w;
        const by = (tr.y - tr.h / 2) * h;
        const bw = tr.w * w;
        const bh = tr.h * h;

        ctx!.save();
        ctx!.globalAlpha = Math.max(0, Math.min(1, alpha)) * 0.95;
        drawBracketBox(bx, by, bw, bh, color);

        const speedPxPerSec = Math.hypot(tr.vx, tr.vy) * w;
        const speed = (speedPxPerSec / 40).toFixed(1); // arbitrary px->m/s-ish scale, cosmetic only
        const label = `${tr.cls === 'vehicle' ? 'VEHICLE' : 'PERSON'} #${tr.id}`;
        const sub = `${speed} m/s  vx${tr.vx >= 0 ? '+' : ''}${(tr.vx * 40).toFixed(1)} vy${tr.vy >= 0 ? '+' : ''}${(tr.vy * 40).toFixed(1)}`;

        ctx!.font = `${Math.max(9, Math.round(h * 0.001) + 9)}px "IBM Plex Mono", monospace`;
        ctx!.textBaseline = 'bottom';
        ctx!.fillStyle = color;
        ctx!.fillText(label, bx, by - 12);
        ctx!.font = `${Math.max(8, Math.round(h * 0.001) + 8)}px "IBM Plex Mono", monospace`;
        ctx!.globalAlpha = (Math.max(0, Math.min(1, alpha)) * 0.95) * 0.75;
        ctx!.fillText(sub, bx, by - 1);
        ctx!.restore();
      }

      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
    };
  }, [active, seed]);

  if (!active) return null;
  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />;
}
