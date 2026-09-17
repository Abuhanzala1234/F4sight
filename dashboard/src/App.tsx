import { useCallback, useEffect, useMemo, useState } from 'react';
import type { AlertSummary, Health, ServerMsg, Session } from '@/types';
import { api, clearSession, loadSession } from '@/lib/api';
import { AlertSocket } from '@/lib/ws';
import { Login } from '@/pages/Login';
import { LiveWall } from '@/pages/LiveWall';
import { Alerts } from '@/pages/Alerts';
import { Verify } from '@/pages/Verify';
import { StatusDot } from '@/components/Primitives';
import { Emblem } from '@/components/Emblem';

type Tab = 'wall' | 'alerts' | 'verify';

const TABS: { id: Tab; label: string; key: string }[] = [
  { id: 'wall', label: 'Live wall', key: 'w' },
  { id: 'alerts', label: 'Alerts', key: 'a' },
  { id: 'verify', label: 'Verify', key: 'v' },
];

export default function App() {
  const [session, setSession] = useState<Session | null>(loadSession);
  const [tab, setTab] = useState<Tab>('alerts');
  const [liveAlerts, setLiveAlerts] = useState<AlertSummary[]>([]);
  const [connected, setConnected] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [clock, setClock] = useState(() => new Date());

  useEffect(() => {
    const id = window.setInterval(() => setClock(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!session) return;
    const poll = () => api.health().then(setHealth).catch(() => setHealth(null));
    void poll();
    const id = window.setInterval(poll, 15000);
    return () => window.clearInterval(id);
  }, [session]);

  const onMessage = useCallback((msg: ServerMsg) => {
    if (msg.type === 'alert') setLiveAlerts((prev) => [msg.alert, ...prev].slice(0, 200));
  }, []);

  useEffect(() => {
    if (!session) return;
    const socket = new AlertSocket(session.access_token, onMessage, setConnected);
    socket.connect();
    return () => socket.close();
  }, [session, onMessage]);

  // g-prefixed navigation, the way every keyboard-first tool does it.
  useEffect(() => {
    let armed = false;
    function onKey(event: KeyboardEvent) {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement)
        return;
      if (event.key === 'g') {
        armed = true;
        window.setTimeout(() => (armed = false), 900);
        return;
      }
      if (!armed) return;
      const match = TABS.find((t) => t.key === event.key);
      if (match) {
        setTab(match.id);
        armed = false;
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const degraded = useMemo(
    () => health?.components.filter((c) => !c.ok).map((c) => c.name) ?? [],
    [health],
  );

  if (!session) return <Login onSignedIn={setSession} />;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 border-b border-rule bg-void/95 backdrop-blur">
        <div className="mx-auto flex h-12 max-w-[1800px] items-center gap-5 px-4">
          <div className="flex items-center gap-2">
            <Emblem className="h-6 w-6 text-signal" />
            <span className="font-display text-base font-bold tracking-[0.16em] text-bright">
              IBVAP
            </span>
          </div>

          <nav className="flex items-center gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`group relative px-3 py-1 font-mono text-2xs uppercase tracking-[0.14em] transition-colors ${
                  tab === t.id ? 'text-signal' : 'text-dim hover:text-text'
                }`}
              >
                {t.label}
                <span className="ml-1.5 text-[9px] opacity-50">g{t.key}</span>
                {tab === t.id && (
                  <span className="absolute inset-x-2 -bottom-[1px] h-[2px] bg-signal" />
                )}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-5">
            {degraded.length > 0 && (
              <span
                className="border border-signal/50 bg-signal/10 px-2 py-0.5 font-mono text-2xs text-signal"
                title={`unavailable: ${degraded.join(', ')}`}
              >
                DEGRADED · {degraded.join(' ')}
              </span>
            )}
            <StatusDot
              state={connected ? 'live' : 'down'}
              label={connected ? 'feed live' : 'feed down'}
            />
            <span className="font-mono text-xs tabular-nums text-text">
              {clock.toLocaleTimeString(undefined, { hour12: false })}
            </span>
            <div className="flex items-center gap-2 border-l border-rule pl-4">
              <span className="font-mono text-2xs text-dim">
                {session.display_name}
                <span className="ml-1.5 text-signal">{session.role}</span>
              </span>
              <button
                type="button"
                className="border border-rule2 px-2.5 py-1 font-mono text-2xs uppercase tracking-[0.1em]
                           text-dim transition-colors hover:border-alarm/60 hover:text-alarm"
                onClick={() => {
                  clearSession();
                  setSession(null);
                }}
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1800px] animate-sweep-in px-4 py-3">
        {tab === 'wall' && <LiveWall />}
        {tab === 'alerts' && <Alerts live={liveAlerts} connected={connected} />}
        {tab === 'verify' && <Verify />}
      </main>
    </div>
  );
}
