import { useState } from 'react';
import { api, ApiError } from '@/lib/api';
import type { Verification } from '@/types';
import { VerifyPanel } from '@/components/VerifyPanel';
import { Panel } from '@/components/Primitives';

/**
 * Paste an evidence document, get every check. Needs no alert in this database
 * and no privileged role: tamper-evidence is a property of the record, so a
 * third party must be able to check it independently (P5).
 */
export function Verify() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<Verification | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const parsed = JSON.parse(text) as Record<string, unknown>;
      setResult(await api.verifyDocument(parsed));
    } catch (err) {
      if (err instanceof SyntaxError) setError(`that is not valid JSON: ${err.message}`);
      else if (err instanceof ApiError) setError(err.message);
      else setError('verification failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <Panel
        title="Verify an evidence document"
        right={
          <button type="button" className="btn btn-primary" onClick={run} disabled={busy || !text}>
            {busy ? 'checking…' : 'verify'}
          </button>
        }
      >
        <div className="p-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
            placeholder='Paste an evidence JSON here — the object from an alert&apos;s "evidence_doc" field.'
            className="input h-[420px] resize-none text-xs leading-relaxed"
          />
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
              Every alert carries the exact object that was hashed. Copy it from an alert&apos;s
              detail view, paste it here, and the checks run against the ledger.
            </p>
            <p>
              Change one character of it first, and the verdict becomes{' '}
              <span className="text-alarm">TAMPERED</span> with a field-level diff. That is the
              point: verification fails loudly, never silently.
            </p>
          </div>
        </Panel>
      )}
    </div>
  );
}
