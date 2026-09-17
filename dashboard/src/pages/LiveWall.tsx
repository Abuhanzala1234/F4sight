import { useEffect, useState } from 'react';
import type { Camera } from '@/types';
import { api } from '@/lib/api';
import { CameraTile } from '@/components/CameraTile';
import { ConnectCameraModal } from '@/components/ConnectCameraModal';
import { Empty, Panel } from '@/components/Primitives';

const LAYOUTS = [1, 4, 9, 16] as const;

export function LiveWall() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>(4);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState<Camera | null>(null);
  const [disconnecting, setDisconnecting] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function disconnect(camera: Camera) {
    if (disconnecting) return;
    setDisconnecting(camera.id);
    setActionError(null);
    try {
      const updated = await api.disconnectCamera(camera.id);
      setCameras((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'disconnect failed');
    } finally {
      setDisconnecting(null);
    }
  }

  useEffect(() => {
    let cancelled = false;

    function load(initial: boolean) {
      api
        .cameras()
        .then((next) => {
          if (cancelled) return;
          setCameras(next);
          setError(null);
        })
        .catch((err: unknown) => {
          // A failed refresh must not blank a wall that is already showing
          // cameras -- only the first load has nothing better to display.
          if (cancelled || !initial) return;
          setError(err instanceof Error ? err.message : 'load failed');
        });
    }

    load(true);
    // The camera table changes underneath this page: the worker hot-starts a
    // newly connected camera, an operator on another console connects or
    // disconnects one, a slot is torn back down. Fetching once on mount left
    // the wall showing whatever was true when the tab was opened, which on a
    // console that stays open for a shift is not true for long.
    const timer = window.setInterval(() => load(false), 10000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
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
              <CameraTile
                key={camera.id}
                camera={camera}
                onConnect={() => setConnecting(camera)}
                onDisconnect={camera.enabled ? () => void disconnect(camera) : undefined}
              />
            ))}
          </div>
        )}
      </Panel>

      {actionError && (
        <p className="border border-alarm/50 bg-alarm/10 px-3 py-2 font-mono text-2xs text-alarm">
          {actionError}
        </p>
      )}

      <p className="hazard border border-rule px-3 py-2 font-mono text-2xs leading-relaxed text-dim">
        <span className="text-signal">NO AUTOMATED RESPONSE.</span> This system recommends; a
        human adjudicates every alert (Principle P7). Recording continues independently of
        analytics — if the AI stops, the video does not (P8).
      </p>

      {connecting && (
        <ConnectCameraModal
          camera={connecting}
          onClose={() => setConnecting(null)}
          onConnected={(updated) => {
            setCameras((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
            setConnecting(null);
          }}
        />
      )}
    </div>
  );
}
