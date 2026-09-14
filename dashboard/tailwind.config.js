/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // F4SIGHT palette — "night watch": deep navy-black with warm gold
        // accents, not a flat instrument-panel grey. Every component uses
        // these semantic names rather than raw colours, so retheming here
        // reskins the whole app in one place.
        void: '#070B14', // page background
        panel: '#0D1526', // card / panel surface
        raised: '#152034', // default button / raised surface
        rule: '#1E2C45', // hairline
        rule2: '#33465F', // stronger rule, HUD corner brackets
        dim: '#7E8CA6', // secondary text
        text: '#C7D3E6', // body text
        bright: '#F4F7FC', // headings, high emphasis
        signal: '#E3B155', // primary accent — gold, the colour of caution
        phosphor: '#34D399', // verified / healthy
        ice: '#4FC3F7', // informational
        alarm: '#F16565',
        ember: '#FB923C',

        // Public-facing "F4SIGHT" landing palette — same night-watch family,
        // used directly (not via the semantic names) on the landing page.
        paper: '#070B14',
        ink: '#F4F7FC',
        navy: '#0B1424',
        navy2: '#101C33',
        navy3: '#0D1830',
        gold: '#E3B155',
        saffron: '#FF9933',
        indiagreen: '#34D399',
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
        'spin-slow': 'spin 60s linear infinite',
        'scan': 'scan 6s ease-in-out infinite',
        'drift': 'drift 22s ease-in-out infinite',
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
        scan: {
          '0%,100%': { transform: 'translateY(-10%)', opacity: '0' },
          '10%,90%': { opacity: '1' },
          '50%': { transform: 'translateY(110%)', opacity: '1' },
        },
        drift: {
          '0%,100%': { transform: 'translate(0,0)' },
          '50%': { transform: 'translate(-2%,2%)' },
        },
      },
    },
  },
  plugins: [],
};
