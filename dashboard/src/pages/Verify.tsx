import { useState } from 'react';
import { api, ApiError } from '@/lib/api';
import type { Verification } from '@/types';
import { VerifyPanel } from '@/components/VerifyPanel';
import { Panel } from '@/components/Primitives';

/**
 * Paste an evidence document, get every check. Needs no alert in this database
 * and no privileged role: tamper-evidence is a property of the record, so a
 * third party must be able to check it independently (P5).
 *
 * "Load a real alert" and "tamper it" exist so a skeptic never has to go
 * fish a JSON blob out of another tab first — the whole VERIFIED -> TAMPERED
 * story plays out in two clicks, on real evidence, not a canned fixture.
 */
export function Verify() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<Verification | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tampered, setTampered] = useState(false);

  async function verifyObject(doc: Record<string, unknown>) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.verifyDocument(doc));
    } catch (err) {
      if (err instanceof ApiError) setError(err.message);
      else setError('verification failed');
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    try {
      const parsed = JSON.parse(text) as Record<string, unknown>;
      await verifyObject(parsed);
    } catch (err) {
      setError(err instanceof SyntaxError ? `that is not valid JSON: ${err.message}` : 'verification failed');
    }
  }

  async function loadRealAlert() {
    setBusy(true);
    setError(null);
    setResult(null);
    setTampered(false);
    try {
      const page = await api.alerts({ limit: '1' });
      const latest = page.items[0];
      if (!latest) {
        setError('no alerts exist yet — trigger one (walk into a zone on camera) first');
        return;
      }
      const full = await api.alert(latest.id);
      setText(JSON.stringify(full.evidence_doc, null, 2));
      await verifyObject(full.evidence_doc);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'could not load an alert');
    } finally {
      setBusy(false);
    }
  }

  async function tamperIt() {
    setError(null);
    let doc: Record<string, unknown>;
    try {
      doc = JSON.parse(text) as Record<string, unknown>;
    } catch {
      setError('load or paste a document first — nothing here to tamper with yet');
      return;
    }
    // Nudge one real, meaningful number rather than a random byte, so the
    // diff the verdict shows back is legible: "risk score changed", not
    // "byte 412 changed" (P2 -- alerts must be explainable, so the proof
    // that tampering was caught should be too).
    const risk = doc.risk as { score?: number } | undefined;
    if (risk && typeof risk.score === 'number') {
      risk.score = Math.round((risk.score + 1) * 100) / 100;
    } else {
      doc.tampered_by_demo = true;
    }
    setText(JSON.stringify(doc, null, 2));
    setTampered(true);
    await verifyObject(doc);
  }

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <Panel
        title="Verify an evidence document"
        right={
          <div className="flex gap-2">
            <button type="button" className="btn" onClick={() => void loadRealAlert()} disabled={busy}>
              load a real alert
            </button>
            <button
              type="button"
              className="btn border-alarm/50 text-alarm hover:border-alarm hover:text-alarm"
              onClick={() => void tamperIt()}
              disabled={busy || !text}
            >
              tamper it
            </button>
            <button type="button" className="btn btn-primary" onClick={() => void run()} disabled={busy || !text}>
              {busy ? 'checking…' : 'verify'}
            </button>
          </div>
        }
      >
        <div className="p-3">
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setTampered(false);
            }}
            spellCheck={false}
            placeholder='Click "load a real alert" above, or paste an evidence JSON here — the object from an alert&apos;s "evidence_doc" field.'
            className="input h-[420px] resize-none text-xs leading-relaxed"
          />
          {tampered && (
            <p className="mt-2 border border-alarm/50 bg-alarm/10 px-2 py-1.5 font-mono text-2xs text-alarm">
              risk.score was nudged by +1 — this document no longer matches the hash it was issued
              with. Click "verify" again if it didn&apos;t already re-run.
            </p>
          )}
          {error && (
            <p className="mt-2 border border-alarm/50 bg-alarm/10 px-2 py-1.5 font-mono text-2xs text-alarm">
              {error}
            </p>
          )}
          <p className="mt-2 font-mono text-2xs leading-relaxed text-dim">
            The document is canonicalised with RFC 8785 (JCS) and hashed with SHA-256. The
            fields <span className="text-text">evidence_hash</span> and{' '}
            <span className="text-text">ledger</span> are excluded — one is the output and the
            other is filled in after hashing.
          </p>
        </div>
      </Panel>

      {result ? (
        <VerifyPanel result={result} />
      ) : (
        <Panel title="Result">
          <div className="space-y-3 p-4 font-mono text-2xs leading-relaxed text-dim">
            <p>
              Click <span className="text-text">load a real alert</span> to pull the most recent
              alert&apos;s evidence straight from this database and check it — no copy-pasting
              required.
            </p>
            <p>
              Then click <span className="text-text">tamper it</span> to nudge one real number
              and watch the verdict flip to <span className="text-alarm">TAMPERED</span> with a
              field-level diff. That is the point: verification fails loudly, never silently.
            </p>
          </div>
        </Panel>
      )}
    </div>
  );
}
