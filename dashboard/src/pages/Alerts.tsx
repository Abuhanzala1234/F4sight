import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AlertDetail, AlertSummary, Severity, Verification } from '@/types';
import { api } from '@/lib/api';
import { SEVERITY_ORDER, localTime, relativeTime, ruleLabel, shortHash, utcLabel } from '@/lib/format';
import { Empty, Field, KeyHint, Panel, SeverityBadge } from '@/components/Primitives';
import { RiskWaterfall } from '@/components/RiskWaterfall';
import { VerifyPanel } from '@/components/VerifyPanel';

/**
 * Alert triage (BUILD_SPEC §10).
 *
 * Keyboard-first, because an operator who has to reach for a mouse to
 * acknowledge will stop acknowledging, and an un-triaged queue is the same as
 * no queue. j/k move, Enter opens, a acknowledges, t/f adjudicate.
 */
export function Alerts({
  live,
  connected,
}: {
  live: AlertSummary[];
  /** Live-socket state, used to close the gap after a drop — see the replay
   * effect below. */
  connected: boolean;
}) {
  const [alerts, setAlerts] = useState<AlertSummary[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [selected, setSelected] = useState<number>(0);
  const [detail, setDetail] = useState<AlertDetail | null>(null);
  const [verification, setVerification] = useState<Verification | null>(null);
  const [minSeverity, setMinSeverity] = useState<Severity>('info');
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const load = useCallback(async (after?: string) => {
    try {
      const page = await api.alerts(after ? { cursor: after, limit: '50' } : { limit: '50' });
      setAlerts((prev) => (after ? [...prev, ...page.items] : page.items));
      setCursor(page.next_cursor);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'could not load alerts');
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Close the gap after a socket drop. ws.ts reconnects on its own, but an
  // alert raised while it was down was never pushed to anyone -- so without
  // this the list silently omits it until somebody reloads the page, which is
  // precisely the failure the socket was supposed to be immune to. The REST
  // feed is the source of truth; the socket is only an accelerator.
  //
  // Guarded on having been connected before, so the first connect after mount
  // does not immediately refetch what the load above just fetched.
  const wasConnectedRef = useRef(false);
  useEffect(() => {
    if (!connected) return;
    if (wasConnectedRef.current) void load();
    wasConnectedRef.current = true;
  }, [connected, load]);

  // Live arrivals are merged in at the top rather than replacing the list, so
  // an operator mid-triage does not lose their place.
  useEffect(() => {
    if (live.length === 0) return;
    setAlerts((prev) => {
      const known = new Set(prev.map((a) => a.id));
      const fresh = live.filter((a) => !known.has(a.id));
      if (fresh.length === 0) return prev;
      setFlash(fresh[0]?.id ?? null);
      window.setTimeout(() => setFlash(null), 900);
      return [...fresh, ...prev];
    });
  }, [live]);

  const visible = useMemo(() => {
    const floor = SEVERITY_ORDER.indexOf(minSeverity);
    return alerts.filter((a) => SEVERITY_ORDER.indexOf(a.severity) >= floor);
  }, [alerts, minSeverity]);

  const open = useCallback(async (alert: AlertSummary) => {
    setDetail(null);
    setVerification(null);
    try {
      const full = await api.alert(alert.id);
      setDetail(full);
      setVerification(await api.verify(alert.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'could not open alert');
    }
  }, []);

  const acknowledge = useCallback(async (alert: AlertSummary) => {
    await api.acknowledge(alert.id);
    setAlerts((prev) =>
      prev.map((a) => (a.id === alert.id ? { ...a, status: 'acknowledged' as const } : a)),
    );
  }, []);

  const adjudicate = useCallback(async (alert: AlertSummary, verdict: string) => {
    await api.adjudicate(alert.id, verdict);
    setAlerts((prev) =>
      prev.map((a) =>
        a.id === alert.id
          ? { ...a, status: 'adjudicated' as const, adjudication: verdict as AlertSummary['adjudication'] }
          : a,
      ),
    );
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement)
        return;
      const current = visible[selected];
      switch (event.key) {
        case 'j':
          setSelected((i) => Math.min(i + 1, visible.length - 1));
          break;
        case 'k':
          setSelected((i) => Math.max(i - 1, 0));
          break;
        case 'Enter':
          if (current) void open(current);
          break;
        case 'a':
          if (current) void acknowledge(current);
          break;
        case 't':
          if (current) void adjudicate(current, 'true_positive');
          break;
        case 'f':
          if (current) void adjudicate(current, 'false_positive');
          break;
        default:
          return;
      }
      event.preventDefault();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [visible, selected, open, acknowledge, adjudicate]);

  useEffect(() => {
    listRef.current
      ?.querySelectorAll('li')
      [selected]?.scrollIntoView({ block: 'nearest' });
  }, [selected]);

  return (
    <div className="grid gap-3 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
      <Panel
        title={`Alert queue · ${visible.length}`}
        right={
          <select
            value={minSeverity}
            onChange={(e) => setMinSeverity(e.target.value as Severity)}
            className="border border-rule2 bg-void px-1.5 py-0.5 font-mono text-2xs uppercase text-dim focus:border-signal focus:outline-none"
          >
            {SEVERITY_ORDER.map((s) => (
              <option key={s} value={s}>
                ≥ {s}
              </option>
            ))}
          </select>
        }
        className="flex max-h-[calc(100vh-190px)] flex-col"
      >
        {error && (
          <p className="border-b border-alarm/40 bg-alarm/10 px-3 py-1.5 font-mono text-2xs text-alarm">
            {error}
          </p>
        )}

        <ul ref={listRef} className="flex-1 divide-y divide-rule/50 overflow-y-auto">
          {visible.length === 0 && <Empty>queue clear</Empty>}
          {visible.map((alert, index) => (
            <li key={alert.id}>
              <button
                type="button"
                onClick={() => {
                  setSelected(index);
                  void open(alert);
                }}
                className={`flex w-full flex-col gap-1.5 px-3 py-2 text-left transition-colors ${
                  index === selected ? 'bg-signal/[0.07] shadow-[inset_2px_0_0_0_theme(colors.signal)]' : 'hover:bg-raised'
                } ${flash === alert.id ? 'animate-flash' : ''}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <SeverityBadge severity={alert.severity} score={alert.risk_score} />
                  <span
                    className="font-mono text-2xs tabular-nums text-dim"
                    title={utcLabel(alert.ts_utc)}
                  >
                    {localTime(alert.ts_utc)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-display text-sm tracking-wide text-bright">
                    {ruleLabel(alert.kind)}
                  </span>
                  <span className="shrink-0 font-mono text-2xs text-dim">
                    {relativeTime(alert.ts_utc)}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  {alert.status !== 'raised' && (
                    <span className="border border-rule2 px-1 font-mono text-[10px] uppercase text-dim">
                      {alert.adjudication ?? alert.status}
                    </span>
                  )}
                  {alert.ledger_status === 'anchored' && (
                    <span
                      className="font-mono text-[10px] text-phosphor"
                      title="anchored to the ledger"
                    >
                      ⛓ anchored
                    </span>
                  )}
                  <span className="truncate font-mono text-[10px] text-dim">
                    {alert.reason_codes.slice(0, 3).map(ruleLabel).join(' · ')}
                  </span>
                </div>
              </button>
            </li>
          ))}
        </ul>

        <footer className="flex items-center justify-between gap-3 border-t border-rule px-3 py-2">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            <KeyHint keys={['j', 'k']} action="move" />
            <KeyHint keys={['↵']} action="open" />
            <KeyHint keys={['a']} action="ack" />
            <KeyHint keys={['t', 'f']} action="verdict" />
          </div>
          {cursor && (
            <button type="button" className="btn py-1" onClick={() => void load(cursor)}>
              more
            </button>
          )}
        </footer>
      </Panel>

      <div className="space-y-3">
        {detail ? (
          <AlertDetailView detail={detail} verification={verification} />
        ) : (
          <Panel title="Alert detail">
            <Empty>select an alert · j / k to move, ↵ to open</Empty>
          </Panel>
        )}
      </div>
    </div>
  );
}

function AlertDetailView({
  detail,
  verification,
}: {
  detail: AlertDetail;
  verification: Verification | null;
}) {
  return (
    <>
      <Panel
        title="Alert detail"
        hot={detail.severity === 'critical'}
        right={<SeverityBadge severity={detail.severity} score={detail.risk_score} />}
      >
        <div className="grid grid-cols-2 gap-4 px-3 py-3 sm:grid-cols-4">
          <Field label="event" value={ruleLabel(detail.kind)} mono={false} />
          <Field label="track" value={detail.track_id ?? '—'} />
          <Field
            label="time"
            value={localTime(detail.ts_utc)}
            title={utcLabel(detail.ts_utc)}
          />
          <Field label="status" value={detail.adjudication ?? detail.status} />
        </div>

        <div className="hairline">
          <div className="flex items-center justify-between px-3 pt-2">
            <span className="label">Why this fired</span>
            <span className="label">Principle P2 · additive, explainable</span>
          </div>
          <RiskWaterfall breakdown={detail.risk_breakdown} score={detail.risk_score} />
        </div>

        <div className="hairline px-3 py-2">
          <span className="label">Evidence</span>
          <div className="mt-1.5 space-y-1">
            {detail.items.length === 0 && (
              <p className="font-mono text-2xs text-dim">no media stored for this alert</p>
            )}
            {detail.items.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between gap-3 border border-rule bg-void px-2 py-1.5"
              >
                <span className="font-mono text-2xs uppercase tracking-[0.1em] text-text">
                  {item.kind}
                </span>
                <span className="truncate font-mono text-2xs text-dim" title={item.sha256}>
                  {shortHash(item.sha256, 16)}
                </span>
                <span
                  className={`font-mono text-[10px] ${item.enhanced ? 'text-alarm' : 'text-phosphor'}`}
                  title="Evidence must be the original, unenhanced frame (Principle P4)"
                >
                  {item.enhanced ? 'ENHANCED ⚠' : 'ORIGINAL'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Panel>

      {verification && <VerifyPanel result={verification} />}
    </>
  );
}
