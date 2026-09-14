/** F4SIGHT's badge: a 24-spoke chakra (the wheel on the Indian flag, not the
 * state emblem) with an aperture/iris at the hub. Vigilance + vision, not a
 * reproduction of an official seal. */
export function Emblem({ className = '', ring = true }: { className?: string; ring?: boolean }) {
  const spokes = Array.from({ length: 24 }, (_, i) => i * 15);
  return (
    <svg viewBox="0 0 100 100" className={className} fill="none" aria-hidden="true">
      {ring && <circle cx="50" cy="50" r="47" stroke="currentColor" strokeWidth="1.5" opacity="0.35" />}
      <circle cx="50" cy="50" r="38" stroke="currentColor" strokeWidth="2" />
      {spokes.map((deg) => (
        <line
          key={deg}
          x1="50"
          y1="50"
          x2="50"
          y2="14"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          transform={`rotate(${deg} 50 50)`}
          opacity="0.9"
        />
      ))}
      <circle cx="50" cy="50" r="12.5" fill="currentColor" opacity="0.08" />
      <circle cx="50" cy="50" r="12.5" stroke="currentColor" strokeWidth="2" />
      <circle cx="50" cy="50" r="5" fill="currentColor" />
    </svg>
  );
}
