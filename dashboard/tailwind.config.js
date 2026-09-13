/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Instrument-panel palette. Near-black base, hairline rules, and
        // phosphor accents — a control room at 02:00, not a SaaS dashboard.
        void: '#07080A',
        panel: '#0D0F13',
        raised: '#12151A',
        rule: '#1D222A',
        rule2: '#2A313C',
        dim: '#6B7684',
        text: '#C9D1D9',
        bright: '#F0F4F8',
        signal: '#FFB020', // primary accent — amber, the colour of caution
        phosphor: '#35E07F', // verified / healthy
        ice: '#4FC3F7', // negative risk contributions, informational
        alarm: '#FF4D4D',
        ember: '#FF8A3D',
      },
      fontFamily: {
        display: ['"Chakra Petch"', 'ui-sans-serif', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
        sans: ['"IBM Plex Sans"', 'ui-sans-serif', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.08em' }],
      },
      borderRadius: { none: '0', sm: '2px', DEFAULT: '3px' },
      animation: {
        'pulse-dot': 'pulseDot 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'sweep-in': 'sweepIn 320ms cubic-bezier(0.16, 1, 0.3, 1) both',
        'bar-grow': 'barGrow 520ms cubic-bezier(0.16, 1, 0.3, 1) both',
        'flash': 'flash 900ms ease-out',
      },
      keyframes: {
        pulseDot: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.25' } },
        sweepIn: {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'none' },
        },
        barGrow: { from: { transform: 'scaleX(0)' }, to: { transform: 'scaleX(1)' } },
        flash: {
          '0%': { backgroundColor: 'rgba(255,176,32,0.22)' },
          '100%': { backgroundColor: 'transparent' },
        },
      },
    },
  },
  plugins: [],
};
