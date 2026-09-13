import { useState, type FormEvent } from 'react';
import { api, ApiError } from '@/lib/api';
import type { Session } from '@/types';

export function Login({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [username, setUsername] = useState('operator');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.login(username, password));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'sign-in failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-[380px] animate-sweep-in">
        <div className="mb-7">
          <div className="mb-1 flex items-baseline gap-2.5">
            <h1 className="font-display text-3xl font-bold tracking-[0.14em] text-bright">
              DRISHTI
            </h1>
            <span className="font-mono text-xs tracking-[0.2em] text-signal">BOP</span>
          </div>
          <p className="font-mono text-2xs uppercase leading-relaxed tracking-[0.1em] text-dim">
            Intelligent video analytics · existing CCTV
            <br />
            SIH 2026 · PS 26187 · Team SW-73
          </p>
        </div>

        <form onSubmit={submit} className="panel space-y-3 p-4">
          <div className="space-y-1">
            <label htmlFor="u" className="label">
              Operator
            </label>
            <input
              id="u"
              className="input"
              value={username}
              autoComplete="username"
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="p" className="label">
              Passphrase
            </label>
            <input
              id="p"
              type="password"
              className="input"
              value={password}
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {error && (
            <p className="border border-alarm/50 bg-alarm/10 px-2 py-1.5 font-mono text-2xs text-alarm">
              {error}
            </p>
          )}

          <button type="submit" className="btn btn-primary w-full justify-center" disabled={busy}>
            {busy ? 'authenticating…' : 'sign in'}
          </button>
        </form>

        <p className="mt-4 font-mono text-2xs leading-relaxed text-dim">
          Demo credentials are printed by <span className="text-text">make seed</span>. Face
          analytics is <span className="text-phosphor">disabled</span> by default (P6).
        </p>
      </div>
    </div>
  );
}
