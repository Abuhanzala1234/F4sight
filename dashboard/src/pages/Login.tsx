import { useRef, useState, type FormEvent, type ReactNode } from 'react';
import { api, ApiError } from '@/lib/api';
import type { Session } from '@/types';
import { Emblem } from '@/components/Emblem';
import {
  IconAlert,
  IconCamera,
  IconChart,
  IconFolder,
  IconInfo,
  IconLock,
  IconMapPin,
  IconPhone,
  IconShieldCheck,
} from '@/components/TileIcons';

const NAV_ITEMS = ['About', 'Live Wall', 'Alerts', 'Verify Evidence', 'Watchlist'];

type Tile = {
  label: string;
  sub: string;
  icon: (props: { className?: string }) => ReactNode;
  tone: 'navy' | 'navy2' | 'navy3' | 'gold';
};

const TILES: Tile[] = [
  { label: 'Live Camera Wall', sub: 'Real-time feeds', icon: IconCamera, tone: 'navy' },
  { label: 'Alerts & Intrusions', sub: 'Person / vehicle', icon: IconAlert, tone: 'gold' },
  { label: 'Verify Evidence', sub: 'Hash-checked clips', icon: IconShieldCheck, tone: 'navy2' },
  { label: 'Watchlist', sub: 'Plates & vehicles', icon: IconFolder, tone: 'navy' },
  { label: 'Zones & Geofence', sub: 'Tripwires, areas', icon: IconMapPin, tone: 'navy3' },
  { label: 'Evidence Vault', sub: 'RFC 8785 hashed', icon: IconLock, tone: 'navy' },
  { label: 'Reports & Analytics', sub: 'Site throughput', icon: IconChart, tone: 'gold' },
  { label: 'About F4SIGHT', sub: 'SIH 2026 · PS 26187', icon: IconInfo, tone: 'navy2' },
];

const TONE_STYLE: Record<Tile['tone'], string> = {
  navy: 'bg-gradient-to-br from-[#101C33] to-[#0B1424] text-white/90 hover:shadow-[0_0_0_1px_rgba(227,177,85,0.35),0_18px_40px_-14px_rgba(227,177,85,0.35)]',
  navy2: 'bg-gradient-to-br from-[#132542] to-[#0D1830] text-white/90 hover:shadow-[0_0_0_1px_rgba(227,177,85,0.35),0_18px_40px_-14px_rgba(227,177,85,0.35)]',
  navy3: 'bg-gradient-to-br from-[#0F2038] to-[#0A1526] text-white/90 hover:shadow-[0_0_0_1px_rgba(227,177,85,0.35),0_18px_40px_-14px_rgba(227,177,85,0.35)]',
  gold: 'bg-gradient-to-br from-[#E8BD6C] to-[#C8922E] text-[#241705] hover:shadow-[0_18px_40px_-14px_rgba(227,177,85,0.55)]',
};

function TricolourRule({ className = '' }: { className?: string }) {
  return (
    <div className={`flex h-1 w-full ${className}`} aria-hidden="true">
      <div className="flex-1 bg-saffron" />
      <div className="flex-1 bg-white/90" />
      <div className="flex-1 bg-indiagreen" />
    </div>
  );
}

export function Login({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [username, setUsername] = useState('operator');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const userRef = useRef<HTMLInputElement>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.login(username, password));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'sign-in failed');
    } finally {
      setBusy(false);
    }
  }

  function goToSignIn() {
    document.getElementById('signin')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    window.setTimeout(() => userRef.current?.focus(), 380);
  }

  return (
    <div className="fixed inset-0 z-[110] overflow-y-auto bg-[#070B14] font-sans text-[#F4F7FC] antialiased">
      <TricolourRule />

      {/* ---- header ---- */}
      <header className="sticky top-0 z-20 border-b border-white/[0.07] bg-[#070B14]/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1180px] items-center gap-4 px-5 py-3.5 sm:px-8">
          <Emblem className="h-9 w-9 shrink-0 animate-spin-slow text-gold" />
          <h1 className="font-display text-[1.45rem] font-bold leading-none tracking-[0.03em] text-white">
            F4SIGHT
          </h1>
          <div className="ml-auto flex items-center gap-4">
            <p className="hidden text-right text-[10.5px] font-semibold uppercase leading-tight tracking-[0.14em] text-white/35 sm:block">
              Smart India Hackathon 2026
              <br />
              PS 26187 · Team SW-73
            </p>
            <button
              type="button"
              onClick={goToSignIn}
              className="border border-gold/50 bg-gold/10 px-4 py-1.5 text-[12px] font-bold uppercase tracking-[0.12em] text-gold transition-all hover:bg-gold hover:text-[#241705]"
            >
              Login
            </button>
          </div>
        </div>
      </header>

      {/* ---- nav bar ---- */}
      <nav className="border-b border-white/[0.06] bg-white/[0.02]">
        <div className="mx-auto flex max-w-[1180px] items-center overflow-x-auto px-2 sm:px-6">
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              type="button"
              onClick={goToSignIn}
              className="whitespace-nowrap px-4 py-2.5 text-[12px] font-semibold uppercase tracking-[0.1em] text-white/50 transition-colors hover:text-gold"
            >
              {item}
            </button>
          ))}
        </div>
      </nav>

      {/* ---- hero ---- */}
      <section className="relative overflow-hidden pb-28 pt-16 sm:pb-36 sm:pt-24">
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 900px 500px at 82% 8%, rgba(227,177,85,0.16), transparent 60%), radial-gradient(ellipse 700px 500px at 8% 90%, rgba(52,211,153,0.08), transparent 60%), #070B14',
          }}
        />
        <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.05]" aria-hidden="true">
          <defs>
            <pattern id="grid" width="46" height="46" patternUnits="userSpaceOnUse">
              <path d="M46 0H0V46" fill="none" stroke="white" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
        <Emblem
          className="pointer-events-none absolute -right-24 -top-24 h-[30rem] w-[30rem] animate-spin-slow text-gold opacity-[0.07] sm:h-[38rem] sm:w-[38rem]"
        />
        <div className="pointer-events-none absolute inset-x-0 top-0 h-full overflow-hidden opacity-[0.5]">
          <div className="absolute inset-x-0 h-px animate-scan bg-gradient-to-r from-transparent via-gold/70 to-transparent" />
        </div>

        <div className="relative mx-auto max-w-[1180px] px-5 sm:px-8">
          <p className="mb-5 inline-flex animate-sweep-in items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.3em] text-gold">
            <span className="h-px w-7 bg-gold" /> Border Security Force · Prototype
          </p>
          <h2
            className="max-w-2xl animate-sweep-in font-display text-[2.5rem] font-bold leading-[1.06] tracking-tight sm:text-[3.4rem]"
            style={{ animationDelay: '60ms' }}
          >
            Vigilance at the border,
            <br />
            <span className="bg-gradient-to-r from-[#F3CE8E] via-gold to-[#B87A22] bg-clip-text text-transparent">
              powered by sight.
            </span>
          </h2>
          <p
            className="mt-6 max-w-lg animate-sweep-in text-[15.5px] leading-relaxed text-white/55"
            style={{ animationDelay: '120ms' }}
          >
            F4SIGHT turns cameras already on the fence line into a live intrusion-detection
            network — person and vehicle tracking, cryptographically verifiable evidence, and
            alerts an operator can act on. Built end to end, free, for Smart India Hackathon 2026.
          </p>
          <div className="mt-8 flex animate-sweep-in flex-wrap gap-3" style={{ animationDelay: '180ms' }}>
            <button
              type="button"
              onClick={goToSignIn}
              className="group relative overflow-hidden bg-gradient-to-r from-[#EFC784] to-gold px-6 py-3 text-[13px] font-bold uppercase tracking-[0.1em] text-[#241705] shadow-[0_12px_32px_-10px_rgba(227,177,85,0.6)] transition-transform hover:-translate-y-0.5"
            >
              Operator sign-in
            </button>
            <a
              href="#about"
              className="border border-white/15 px-6 py-3 text-[13px] font-semibold uppercase tracking-[0.1em] text-white/70 transition-colors hover:border-white/40 hover:text-white"
            >
              About the project
            </a>
          </div>
        </div>
      </section>

      {/* ---- tile grid, overlapping the hero ---- */}
      <div className="relative mx-auto -mt-14 max-w-[1180px] px-5 sm:-mt-20 sm:px-8">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {TILES.map((tile, i) => (
            <button
              key={tile.label}
              type="button"
              onClick={goToSignIn}
              style={{ animationDelay: `${220 + i * 45}ms` }}
              className={`group flex aspect-square animate-sweep-in flex-col items-center justify-center gap-2.5 border border-white/[0.06] p-3 text-center shadow-[0_14px_32px_-16px_rgba(0,0,0,0.6)] transition-all duration-300 hover:-translate-y-1.5 ${TONE_STYLE[tile.tone]}`}
            >
              <tile.icon className="h-8 w-8 shrink-0 transition-transform duration-300 group-hover:scale-110 sm:h-9 sm:w-9" />
              <span className="text-[11.5px] font-bold uppercase leading-tight tracking-[0.02em] sm:text-[12.5px]">
                {tile.label}
              </span>
              <span className="text-[9.5px] font-medium uppercase tracking-[0.08em] opacity-60">
                {tile.sub}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* ---- about + sign-in ---- */}
      <section id="about" className="relative mx-auto max-w-[1180px] px-5 py-20 sm:px-8 sm:py-28">
        <div className="grid gap-14 md:grid-cols-[1.15fr_0.85fr] md:gap-16">
          <div className="animate-sweep-in">
            <p className="mb-2 text-[11px] font-bold uppercase tracking-[0.22em] text-gold">
              About the project
            </p>
            <h3 className="font-display text-2xl font-bold text-white sm:text-[1.85rem]">
              Every border outpost already has cameras.
              <br className="hidden sm:block" /> Most of them are never watched.
            </h3>
            <p className="mt-5 max-w-xl text-[14.5px] leading-[1.8] text-white/50">
              F4SIGHT is a free, self-hosted analytics layer for existing CCTV: detection and
              tracking run on-device, zone and tripwire rules turn a crossing into a scored
              alert, and every alert carries evidence — snapshot, clip, and a canonicalised
              hash — an investigator can independently verify later.
            </p>
            <dl className="mt-9 grid grid-cols-2 gap-x-6 gap-y-6 border-t border-white/10 pt-7 sm:grid-cols-3">
              {[
                ['Detection', 'YOLO11n · ONNX'],
                ['Tracking', 'ByteTrack'],
                ['Evidence', 'RFC 8785 hashed'],
                ['Ledger', 'Merkle-anchored'],
                ['Cost', '$0 · free stack'],
                ['Mode', 'Offline-first'],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="text-[10.5px] font-bold uppercase tracking-[0.14em] text-white/30">
                    {k}
                  </dt>
                  <dd className="mt-1 text-[13.5px] font-semibold text-white/85">{v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div id="signin" className="scroll-mt-24">
            <div className="relative animate-sweep-in overflow-hidden border border-white/10 bg-white/[0.03] p-6 shadow-[0_30px_60px_-24px_rgba(0,0,0,0.7)] backdrop-blur-sm sm:p-7">
              <div
                className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full opacity-30 blur-3xl"
                style={{ background: 'radial-gradient(circle, rgba(227,177,85,0.5), transparent 70%)' }}
              />
              <p className="text-[11px] font-bold uppercase tracking-[0.2em] text-gold">
                Restricted access
              </p>
              <h4 className="mt-1 font-display text-lg font-bold text-white">Operator sign-in</h4>
              <p className="mt-1 text-[12.5px] text-white/40">
                Credentials are issued per site. Demo logins are printed by{' '}
                <code className="text-white/60">make seed</code>.
              </p>

              <form onSubmit={submit} className="mt-6 space-y-4">
                <div className="space-y-1.5">
                  <label htmlFor="u" className="text-[11px] font-bold uppercase tracking-[0.1em] text-white/45">
                    Username
                  </label>
                  <input
                    id="u"
                    ref={userRef}
                    className="w-full border border-white/10 bg-black/25 px-3 py-2.5 text-[14px] text-white outline-none transition-colors focus:border-gold focus:bg-black/40"
                    value={username}
                    autoComplete="username"
                    onChange={(e) => setUsername(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="p" className="text-[11px] font-bold uppercase tracking-[0.1em] text-white/45">
                    Passphrase
                  </label>
                  <input
                    id="p"
                    type="password"
                    className="w-full border border-white/10 bg-black/25 px-3 py-2.5 text-[14px] text-white outline-none transition-colors focus:border-gold focus:bg-black/40"
                    value={password}
                    autoComplete="current-password"
                    onChange={(e) => setPassword(e.target.value)}
                  />
                </div>

                {error && (
                  <p className="border border-red-500/30 bg-red-500/10 px-3 py-2 text-[12.5px] text-red-300">
                    {error}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={busy}
                  className="w-full bg-gradient-to-r from-[#EFC784] to-gold py-2.5 text-[13px] font-bold uppercase tracking-[0.1em] text-[#241705] shadow-[0_12px_28px_-10px_rgba(227,177,85,0.55)] transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
                >
                  {busy ? 'Authenticating…' : 'Sign in'}
                </button>
              </form>

              <p className="mt-6 flex items-start gap-2 border-t border-white/10 pt-4 text-[11.5px] leading-relaxed text-white/35">
                <IconLock className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                Face analytics is disabled by default. No automated response — F4SIGHT
                recommends, a human decides.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ---- footer ---- */}
      <footer className="border-t border-white/[0.06] bg-black/20 pb-8 pt-10">
        <div className="mx-auto flex max-w-[1180px] flex-col items-center gap-3 px-5 text-center sm:flex-row sm:justify-between sm:px-8 sm:text-left">
          <div className="flex items-center gap-3">
            <Emblem className="h-7 w-7 shrink-0 text-gold" />
            <div>
              <p className="text-[13px] font-bold uppercase tracking-[0.1em] text-white">
                F4SIGHT
              </p>
              <p className="text-[11.5px] text-white/35">
                Smart India Hackathon 2026 · PS 26187 · Team SW-73 (ByteForge)
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-[11.5px] text-white/35">
            <IconPhone className="h-3.5 w-3.5" />
            Emergency: 112 · Non-emergency: site duty desk
          </div>
        </div>
      </footer>
      <TricolourRule />
    </div>
  );
}
