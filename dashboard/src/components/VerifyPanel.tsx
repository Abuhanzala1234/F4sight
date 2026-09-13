import type { Verification, Verdict } from '@/types';
import { shortHash, utcLabel } from '@/lib/format';
import { Panel } from './Primitives';

/**
 * Evidence verification (BUILD_SPEC §8, §10). The panel that wins the
 * blockchain argument.
 *
 * It shows every intermediate value, not a verdict, because the claim is
 * falsifiable: an evaluator should be able to take the canonical byte count and
 * the leaf hash away and recompute them. A green tick that hides its working is
 * worth nothing to a sceptic.
 */

const VERDICT_STYLE: Record<Verdict, { fg: string; border: string; bg: string; text: string }> = {
  VERIFIED: {
    fg: 'text-phosphor',
    border: 'border-phosphor/60',
    bg: 'bg-phosphor/10',
    text: 'Evidence is intact and anchored',
  },
  PENDING_ANCHOR: {
    fg: 'text-signal',
    border: 'border-signal/60',
    bg: 'bg-signal/10',
    text: 'Hash verified · awaiting the next ledger batch',
  },
  TAMPERED: {
    fg: 'text-alarm',
    border: 'border-alarm/70',
    bg: 'bg-alarm/15',
    text: 'This record does not match its hash',
  },
  UNVERIFIABLE: {
    fg: 'text-dim',
    border: 'border-rule2',
    bg: 'bg-raised',
    text: 'Not enough information to verify',
  },
};

function Row({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[132px_minmax(0,1fr)] gap-3 py-1">
      <span className="label pt-[2px]">{label}</span>
      <span className={`${mono ? 'font-mono' : ''} break-all text-xs text-text`}>{value}</span>
    </div>
  );
}

export function VerifyPanel({ result }: { result: Verification }) {
  const style = VERDICT_STYLE[result.verdict];

  return (
    <Panel title="Evidence verification" hot={result.verdict === 'TAMPERED'}>
      <div className={`border-b ${style.border} ${style.bg} px-3 py-2.5`}>
        <div className="flex items-baseline justify-between gap-3">
          <span className={`font-display text-lg font-bold tracking-[0.1em] ${style.fg}`}>
            {result.verdict.replace('_', ' ')}
          </span>
          <span className="text-2xs uppercase tracking-[0.1em] text-dim">{style.text}</span>
        </div>
      </div>

      <ol className="divide-y divide-rule/60">
        {result.checks.map((check) => (
          <li key={check.name} className="flex items-start gap-2.5 px-3 py-2">
            <span
              className={`mt-[1px] font-mono text-sm leading-none ${
                check.passed ? 'text-phosphor' : 'text-alarm'
              }`}
              aria-hidden
            >
              {check.passed ? '✓' : '✗'}
            </span>
            <div className="min-w-0 flex-1">
              <div className="font-mono text-xs text-bright">{check.name}</div>
              <div className="break-all font-mono text-2xs text-dim">{check.detail}</div>
            </div>
            <span
              className={`font-mono text-2xs ${check.passed ? 'text-phosphor' : 'text-alarm'}`}
            >
              {check.passed ? 'PASS' : 'FAIL'}
            </span>
          </li>
        ))}
      </ol>

      <div className="hairline px-3 py-2">
        <Row label="stored hash" value={result.stored_hash} />
        <Row label="recomputed" value={result.recomputed_hash} />
        <Row
          label="canonical form"
          value={`RFC 8785 JCS · ${result.canonical_length} bytes`}
          mono={false}
        />
      </div>

      {result.merkle && (
        <div className="hairline px-3 py-2">
          <div className="label mb-1.5">Merkle inclusion</div>
          <Row
            label="leaf"
            value={`#${result.merkle.leaf_index} · ${shortHash(result.merkle.leaf_hash, 20)}`}
          />
          <Row label="root" value={result.merkle.stored_root} />
          <Row
            label="proof path"
            value={`${result.merkle.proof.length} sibling${result.merkle.proof.length === 1 ? '' : 's'}`}
            mono={false}
          />
          {/* The proof itself, laid out so it can be checked by hand. */}
          <ol className="mt-1 space-y-[2px]">
            {result.merkle.proof.map(([side, hash], i) => (
              <li key={i} className="flex items-center gap-2 font-mono text-2xs text-dim">
                <span className="w-8 text-right tabular-nums">{i}</span>
                <span
                  className={`inline-block w-4 text-center ${
                    side === 'L' ? 'text-ice' : 'text-signal'
                  }`}
                >
                  {side}
                </span>
                <span className="break-all">{hash}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {result.ledger && (
        <div className="hairline px-3 py-2">
          <div className="label mb-1.5">Ledger anchor</div>
          <Row label="backend" value={result.ledger.backend} />
          <Row label="transaction" value={result.ledger.tx_id ?? '—'} />
          {result.ledger.block_number !== null && (
            <Row label="block" value={String(result.ledger.block_number)} />
          )}
          {result.ledger.anchored_at && (
            <Row label="anchored" value={utcLabel(result.ledger.anchored_at)} />
          )}
        </div>
      )}

      {result.verdict === 'PENDING_ANCHOR' && (
        <p className="hazard hairline px-3 py-2 font-mono text-2xs leading-relaxed text-dim">
          The evidence hash checks out. This alert has not yet been written into a Merkle
          batch. Anchoring is deliberately off the alerting path — ledger unavailability
          never delays an alert (BUILD_SPEC §7.12).
        </p>
      )}

      {result.diff.length > 0 && (
        <div className="hairline px-3 py-2">
          <div className="label mb-1.5 text-alarm">What changed</div>
          <ul className="space-y-[2px]">
            {result.diff.map((line) => (
              <li key={line} className="break-all font-mono text-2xs text-alarm/90">
                {line}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}
