import { useEffect, useState } from 'react';
import type { Camera } from '@/types';
import { api } from '@/lib/api';
import { CameraTile } from '@/components/CameraTile';
import { Empty, Panel } from '@/components/Primitives';

const LAYOUTS = [1, 4, 9, 16] as const;

export function LiveWall() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>(4);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .cameras()
      .then(setCameras)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'load failed'));
  }, []);

  const columns = Math.sqrt(layout);
  const visible = cameras.slice(0, layout);

  return (
    <div className="space-y-3">
      <Panel
        title="Live wall"
        right={
          <div className="flex items-center gap-1">
            {LAYOUTS.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setLayout(n)}
                className={`border px-2 py-0.5 font-mono text-2xs transition-colors ${
                  layout === n
                    ? 'border-signal bg-signal/15 text-signal'
                    : 'border-rule2 text-dim hover:text-text'
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        }
      >
        {error ? (
          <Empty>{error}</Empty>
        ) : visible.length === 0 ? (
          <Empty>no cameras configured — run make seed</Empty>
        ) : (
          <div
            className="grid gap-2 p-2"
            style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
          >
            {visible.map((camera) => (
              <CameraTile key={camera.id} camera={camera} />
            ))}
          </div>
        )}
      </Panel>

      <p className="hazard border border-rule px-3 py-2 font-mono text-2xs leading-relaxed text-dim">
        <span className="text-signal">NO AUTOMATED RESPONSE.</span> This system recommends; a
        human adjudicates every alert (Principle P7). Recording continues independently of
        analytics — if the AI stops, the video does not (P8).
      </p>
    </div>
  );
}
