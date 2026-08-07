import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#080b13',
          900: '#0d1220',
          850: '#121a2b',
          800: '#18223a',
          700: '#233150',
          600: '#33456d',
        },
        signal: {
          approve: '#34d399',
          decline: '#f87171',
          escalate: '#fbbf24',
          neutral: '#7c93b8',
          accent: '#60a5fa',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
