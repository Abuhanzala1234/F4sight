import { useState, type FormEvent } from 'react';
import { api, ApiError } from '@/lib/api';
import type { Camera } from '@/types';

/**
 * Bind an empty camera slot to a live IP camera. No AI/model step here — the
 * detector is already loaded once and shared across every camera (§7.4);
 * this only registers a video source on MediaMTX and flips the camera row to
 * enabled. The worker notices and hot-starts it within a few seconds, no
 * restart.
 */
export function ConnectCameraModal({
  camera,
  onClose,
  onConnected,
}: {
  camera: Camera;
  onClose: () => void;
  onConnected: (camera: Camera) => void;
}) {
  const [ip, setIp] = useState('');
  const [port, setPort] = useState('8080');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const updated = await api.connectCamera(camera.id, ip.trim(), Number(port) || 8080);
      onConnected(updated);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'could not connect');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center bg-void/80 px-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="panel panel-hot w-full max-w-[380px] animate-sweep-in p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <p className="label">Connect camera</p>
            <h3 className="font-display text-lg font-bold text-bright">{camera.code}</h3>
          </div>
          <button type="button" onClick={onClose} className="text-dim hover:text-text">
            ✕
          </button>
        </div>

        <p className="mb-4 font-mono text-2xs leading-relaxed text-dim">
          Point a phone's{' '}
          <span className="text-text">IP Webcam</span> app (or any RTSP-serving camera) at this
          slot — same WiFi, screen unlocked. No model download, no AI setup: detection is already
          running and just needs a video source.
        </p>

        <form onSubmit={submit} className="space-y-3">
          <div className="space-y-1">
            <label htmlFor="ip" className="label">
              Camera IP address
            </label>
            <input
              id="ip"
              className="input"
              placeholder="192.168.1.23"
              value={ip}
              autoFocus
              onChange={(e) => setIp(e.target.value)}
              required
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="port" className="label">
              Port
            </label>
            <input
              id="port"
              className="input"
              value={port}
              onChange={(e) => setPort(e.target.value)}
              inputMode="numeric"
            />
          </div>

          {error && (
            <p className="border border-alarm/50 bg-alarm/10 px-2 py-1.5 font-mono text-2xs text-alarm">
              {error}
            </p>
          )}

          <div className="flex gap-2 pt-1">
            <button type="submit" className="btn btn-primary flex-1 justify-center" disabled={busy}>
              {busy ? 'connecting…' : 'go live'}
            </button>
            <button type="button" className="btn" onClick={onClose}>
              cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
