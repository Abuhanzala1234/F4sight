type IconProps = { className?: string };
const base = 'stroke-current';
const wrap = (children: React.ReactNode, className?: string) => (
  <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className={`${base} ${className ?? ''}`}>
    {children}
  </svg>
);

export const IconCamera = ({ className }: IconProps) =>
  wrap(
    <>
      <path d="M3 8.5A1.5 1.5 0 0 1 4.5 7h2.6l1-1.6A1.5 1.5 0 0 1 9.4 4.7h5.2a1.5 1.5 0 0 1 1.3.7l1 1.6h2.6A1.5 1.5 0 0 1 21 8.5v9A1.5 1.5 0 0 1 19.5 19h-15A1.5 1.5 0 0 1 3 17.5v-9Z" />
      <circle cx="12" cy="13" r="3.6" />
    </>,
    className,
  );

export const IconAlert = ({ className }: IconProps) =>
  wrap(
    <>
      <path d="M12 3.5 2.5 20h19L12 3.5Z" />
      <path d="M12 10v4.2" />
      <circle cx="12" cy="17" r="0.4" fill="currentColor" stroke="none" />
    </>,
    className,
  );

export const IconShieldCheck = ({ className }: IconProps) =>
  wrap(
    <>
      <path d="M12 3.5 5 6.2V11c0 5 3 8.3 7 9.5 4-1.2 7-4.5 7-9.5V6.2L12 3.5Z" />
      <path d="M8.7 12.1l2.1 2.1 4.3-4.5" />
    </>,
    className,
  );

export const IconFolder = ({ className }: IconProps) =>
  wrap(
    <path d="M3.5 7A1.5 1.5 0 0 1 5 5.5h4l1.6 2H19A1.5 1.5 0 0 1 20.5 9v8A1.5 1.5 0 0 1 19 18.5H5A1.5 1.5 0 0 1 3.5 17V7Z" />,
    className,
  );

export const IconMapPin = ({ className }: IconProps) =>
  wrap(
    <>
      <path d="M12 21s7-6.6 7-11.5A7 7 0 0 0 5 9.5C5 14.4 12 21 12 21Z" />
      <circle cx="12" cy="9.5" r="2.4" />
    </>,
    className,
  );

export const IconLock = ({ className }: IconProps) =>
  wrap(
    <>
      <rect x="4.5" y="10.5" width="15" height="9.5" rx="1.5" />
      <path d="M7.5 10.5V7.8a4.5 4.5 0 1 1 9 0v2.7" />
    </>,
    className,
  );

export const IconChart = ({ className }: IconProps) =>
  wrap(
    <>
      <path d="M4 20V4" />
      <path d="M4 20h16" />
      <path d="M7.5 20v-6" />
      <path d="M12 20V9" />
      <path d="M16.5 20v-3.5" />
    </>,
    className,
  );

export const IconInfo = ({ className }: IconProps) =>
  wrap(
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5.2" />
      <circle cx="12" cy="8" r="0.4" fill="currentColor" stroke="none" />
    </>,
    className,
  );

export const IconPhone = ({ className }: IconProps) =>
  wrap(
    <path d="M5 4.5h3.2l1.3 4-2 1.3a12 12 0 0 0 5.7 5.7l1.3-2 4 1.3V18a1.5 1.5 0 0 1-1.6 1.5A15 15 0 0 1 3.5 6.1 1.5 1.5 0 0 1 5 4.5Z" />,
    className,
  );
