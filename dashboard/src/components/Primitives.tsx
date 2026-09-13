import type { ReactNode } from 'react';
import type { Severity } from '@/types';
import { SEVERITY } from '@/lib/format';

export function Panel({
  title,
  right,
  hot = false,
  className = '',
  children,
}: {
  title?: string;
  right?: ReactNode;
  hot?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`panel ${hot ? 'panel-hot' : ''} ${className}`}>
      {title && (
        <header className="flex items-center justify-between border-b border-rule px-3 py-2">
          <h2 className="label text-[10px]">{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

/**
 * Severity chip. Colour + glyph + notch count — three channels, so the badge
 * still reads correctly in greyscale or with a colour vision deficiency (§10).
 */
export function SeverityBadge({ severity, score }: { severity: Severity; score?: number }) {
  const s = SEVERITY[severity];
  return (
    <span
      className={`inline-flex items-center gap-1.5 border ${s.border} ${s.bg} px-1.5 py-0.5 font-mono text-2xs ${s.fg}`}
      title={`${s.label}${score !== undefined ? ` · risk ${score.toFixed(1)}/100` : ''}`}
    >
      <span aria-hidden className="text-[11px] leading-none">
        {s.glyph}
      </span>
      <span className="tracking-[0.12em]">{s.label}</span>
      <span aria-hidden className="flex gap-[2px]">
        {Array.from({ length: 5 }, (_, i) => (
          <span
            key={i}
            className={`block h-[9px] w-[2px] ${i < s.notches ? 'bg-current' : 'bg-current opacity-20'}`}
          />
        ))}
      </span>
      {score !== undefined && <span className="tabular-nums opacity-80">{score.toFixed(0)}</span>}
    </span>
  );
}

export function StatusDot({
  state,
  label,
}: {
  state: 'live' | 'warn' | 'down' | 'idle';
  label?: string;
}) {
  const colour = {
    live: 'bg-phosphor',
    warn: 'bg-signal',
    down: 'bg-alarm',
    idle: 'bg-dim',
  }[state];
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={`block h-1.5 w-1.5 ${colour} ${state === 'live' ? 'animate-pulse-dot' : ''}`}
      />
      {label && <span className="label">{label}</span>}
    </span>
  );
}

export function KeyHint({ keys, action }: { keys: string[]; action: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {keys.map((k) => (
        <kbd key={k} className="kbd">
          {k}
        </kbd>
      ))}
      <span className="text-2xs uppercase tracking-[0.1em] text-dim">{action}</span>
    </span>
  );
}

export function Field({ label, value, mono = true, title }: {
  label: string;
  value: ReactNode;
  mono?: boolean;
  title?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <span className={`${mono ? 'font-mono' : ''} text-sm text-bright`} title={title}>
        {value}
      </span>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
      <span className="font-mono text-2xs uppercase tracking-[0.2em] text-dim">{children}</span>
    </div>
  );
}
