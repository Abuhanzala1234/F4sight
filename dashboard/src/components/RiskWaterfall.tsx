import type { RiskContribution } from '@/types';
import { ruleLabel } from '@/lib/format';

/**
 * The additive risk model, made visible (BUILD_SPEC §10, Principle P2).
 *
 * This component is the argument. The score is not a number a model emitted —
 * it is a sum, and every term is on screen. Positive contributions run right in
 * amber; the negative ones that P3 relies on (poor light, brief sighting, low
 * detector confidence) run LEFT in ice blue, so an operator can see at a glance
 * that the system is hedging and why.
 *
 * The footer restates the arithmetic. If the terms ever failed to add up, the
 * API would already have refused to serve the alert — but showing the sum is
 * what makes the claim checkable rather than asserted.
 */
export function RiskWaterfall({
  breakdown,
  score,
}: {
  breakdown: RiskContribution[];
  score: number;
}) {
  const magnitude = Math.max(...breakdown.map((c) => Math.abs(c.weight)), 10);
  const sum = breakdown.reduce((acc, c) => acc + c.weight, 0);
  const balances = Math.abs(sum - score) < 0.011;

  return (
    <div className="px-3 py-3">
      <div className="space-y-[3px]">
        {breakdown.map((c, i) => {
          const positive = c.weight >= 0;
          const pct = (Math.abs(c.weight) / magnitude) * 50;
          return (
            <div
              key={`${c.code}-${i}`}
              className="group grid grid-cols-[minmax(0,1fr)_minmax(0,190px)_46px] items-center gap-3"
              style={{ animationDelay: `${i * 45}ms` }}
            >
              <span
                className={`truncate font-mono text-xs ${positive ? 'text-text' : 'text-ice/80'}`}
                title={JSON.stringify(c.detail, null, 2)}
              >
                {ruleLabel(c.code)}
              </span>

              {/* The bar track. A centre rule marks zero: bars to the right add
                  risk, bars to the left subtract it. */}
              <div className="relative h-4 border border-rule/60 bg-void">
                <span className="absolute left-1/2 top-0 h-full w-px bg-rule2" aria-hidden />
                <span
                  className={`absolute top-[2px] h-[10px] animate-bar-grow ${
                    positive ? 'origin-left bg-signal/80' : 'origin-right bg-ice/70'
                  }`}
                  style={{
                    left: positive ? '50%' : `${50 - pct}%`,
                    width: `${pct}%`,
                    animationDelay: `${i * 45}ms`,
                  }}
                  aria-hidden
                />
              </div>

              <span
                className={`text-right font-mono text-xs tabular-nums ${
                  positive ? 'text-signal' : 'text-ice'
                }`}
              >
                {positive ? '+' : '−'}
                {Math.abs(c.weight).toFixed(1)}
              </span>
            </div>
          );
        })}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-rule pt-2">
        <span className="label">
          {breakdown.length} contribution{breakdown.length === 1 ? '' : 's'}
          {breakdown.some((c) => c.weight < 0) && (
            <span className="ml-2 text-ice/70">· blue terms reduce the score</span>
          )}
        </span>
        <span className="flex items-baseline gap-2 font-mono text-xs">
          <span className="text-dim">Σ</span>
          <span className={balances ? 'text-phosphor' : 'text-alarm'}>{sum.toFixed(2)}</span>
          <span className="text-dim">=</span>
          <span className="text-lg font-semibold tabular-nums text-bright">
            {score.toFixed(1)}
          </span>
          <span className="text-dim">/100</span>
        </span>
      </div>

      {!balances && (
        <p className="mt-2 border border-alarm/50 bg-alarm/10 px-2 py-1 font-mono text-2xs text-alarm">
          CONTRIBUTIONS DO NOT SUM TO THE SCORE — this alert's explanation cannot be trusted
          (Principle P2). Report this.
        </p>
      )}
    </div>
  );
}
