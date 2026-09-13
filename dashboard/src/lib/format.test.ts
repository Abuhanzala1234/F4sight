import { describe, expect, it } from 'vitest';
import { SEVERITY, SEVERITY_ORDER, ruleLabel, shortHash } from './format';

describe('severity encoding', () => {
  it('never relies on colour alone', () => {
    // §10: colour is never the sole severity carrier. Each level must be
    // distinguishable by glyph and notch count too.
    const glyphs = SEVERITY_ORDER.map((s) => SEVERITY[s].glyph);
    const notches = SEVERITY_ORDER.map((s) => SEVERITY[s].notches);
    expect(new Set(glyphs).size).toBe(SEVERITY_ORDER.length);
    expect(new Set(notches).size).toBe(SEVERITY_ORDER.length);
  });

  it('notches increase with severity', () => {
    const notches = SEVERITY_ORDER.map((s) => SEVERITY[s].notches);
    expect([...notches].sort((a, b) => a - b)).toEqual(notches);
  });
});

describe('ruleLabel', () => {
  it('humanises known codes', () => {
    expect(ruleLabel('ZONE_INTRUSION')).toBe('Zone intrusion');
    expect(ruleLabel('WATCHLIST_PLATE')).toBe('Watchlist plate match');
  });

  it('never shows raw SCREAMING_SNAKE for unknown codes', () => {
    expect(ruleLabel('SOME_NEW_RULE')).toBe('some new rule');
  });
});

describe('shortHash', () => {
  it('truncates with an ellipsis', () => {
    expect(shortHash('a'.repeat(64), 8)).toBe('aaaaaaaa…');
  });

  it('leaves short strings alone', () => {
    expect(shortHash('abc', 8)).toBe('abc');
  });
});
