/** Presentation helpers. */

import type { Severity } from '@/types';

/**
 * Severity is carried by THREE channels, not one: colour, a glyph, and a notch
 * count. BUILD_SPEC §10 requires that colour is never the sole carrier — about
 * 1 in 12 men has a colour vision deficiency, and a control room is exactly
 * where that must not matter.
 */
export const SEVERITY: Record<
  Severity,
  { label: string; glyph: string; notches: number; fg: string; bg: string; border: string }
> = {
  info: { label: 'INFO', glyph: '·', notches: 1, fg: 'text-dim', bg: 'bg-dim/10', border: 'border-dim/40' },
  low: { label: 'LOW', glyph: '▪', notches: 2, fg: 'text-ice', bg: 'bg-ice/10', border: 'border-ice/40' },
  medium: { label: 'MED', glyph: '▲', notches: 3, fg: 'text-signal', bg: 'bg-signal/10', border: 'border-signal/40' },
  high: { label: 'HIGH', glyph: '◆', notches: 4, fg: 'text-ember', bg: 'bg-ember/10', border: 'border-ember/50' },
  critical: { label: 'CRIT', glyph: '✶', notches: 5, fg: 'text-alarm', bg: 'bg-alarm/15', border: 'border-alarm/60' },
};

export const SEVERITY_ORDER: Severity[] = ['info', 'low', 'medium', 'high', 'critical'];

/** Human-readable rule codes. The dashboard should never show SCREAMING_SNAKE. */
const RULE_LABELS: Record<string, string> = {
  ZONE_INTRUSION: 'Zone intrusion',
  TRIPWIRE_CROSS: 'Tripwire crossed',
  LOITER: 'Loitering',
  NIGHT_MOVEMENT: 'Movement at night',
  PERIMETER_APPROACH: 'Approaching perimeter',
  UNAUTHORISED_VEHICLE: 'Unauthorised vehicle',
  WATCHLIST_FACE: 'Watchlist face match',
  WATCHLIST_PLATE: 'Watchlist plate match',
  CROWD_FORMING: 'Crowd forming',
  ABANDONED_OBJECT: 'Abandoned object',
  CAMERA_TAMPER: 'Camera tamper',
  LOW_CONFIDENCE: 'Low detector confidence',
  DEGRADED_INPUT: 'Degraded image conditions',
  SHORT_TRACK: 'Object seen only briefly',
  KNOWN_PATROL_WINDOW: 'Scheduled patrol window',
  CLAMP: 'Score clamped to range',
};

export const ruleLabel = (code: string): string =>
  RULE_LABELS[code] ?? code.toLowerCase().replace(/_/g, ' ');

/** Every timestamp shows local time; UTC is on the title attribute (§10). */
export function localTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour12: false });
}

export function localDateTime(iso: string): string {
  const d = new Date(iso);
  return `${d.toLocaleDateString(undefined, { day: '2-digit', month: 'short' })} ${d.toLocaleTimeString(undefined, { hour12: false })}`;
}

export function utcLabel(iso: string): string {
  return `${new Date(iso).toISOString().replace('T', ' ').slice(0, 19)} UTC`;
}

export function relativeTime(iso: string): string {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 5) return 'now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export const shortHash = (hash: string, n = 10): string =>
  hash.length > n ? `${hash.slice(0, n)}…` : hash;
