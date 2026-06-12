import typography from '@tailwindcss/typography'

/**
 * Lity — Codex-aligned design tokens.
 *
 * Cool/true-neutral palette, near-monochrome. Primary actions use a near-black
 * (light) / near-white (dark) SOLID; the cyan accent is reserved for focus,
 * links, and active/selected states. Driven by CSS variables (index.css) so
 * light/dark switch via the `.dark` class; RGB-channel vars keep `<alpha-value>`.
 */

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surfaces
        canvas: 'rgb(var(--canvas) / <alpha-value>)',
        panel: 'rgb(var(--panel) / <alpha-value>)',
        surface: 'rgb(var(--surface) / <alpha-value>)',
        'surface-2': 'rgb(var(--surface-2) / <alpha-value>)',
        'surface-3': 'rgb(var(--surface-3) / <alpha-value>)',
        elevated: 'rgb(var(--elevated) / <alpha-value>)',
        // Text
        primary: 'rgb(var(--text) / <alpha-value>)',
        secondary: 'rgb(var(--text-2) / <alpha-value>)',
        tertiary: 'rgb(var(--text-3) / <alpha-value>)',
        // Hairlines
        hairline: 'var(--hairline)',
        'hairline-strong': 'var(--hairline-strong)',
        // Accent — the single restrained pop (focus, links, active)
        accent: {
          DEFAULT: 'rgb(var(--accent) / <alpha-value>)',
          hover: 'rgb(var(--accent-hover) / <alpha-value>)',
          press: 'rgb(var(--accent-press) / <alpha-value>)',
          contrast: 'rgb(var(--accent-contrast) / <alpha-value>)',
          soft: 'rgb(var(--accent) / 0.55)',
        },
        // Solid — near-black/near-white primary fill (OpenAI/Codex signature)
        solid: {
          DEFAULT: 'rgb(var(--solid) / <alpha-value>)',
          hover: 'rgb(var(--solid-hover) / <alpha-value>)',
          contrast: 'rgb(var(--solid-contrast) / <alpha-value>)',
        },
        // Semantic
        success: 'rgb(var(--success) / <alpha-value>)',
        warn: 'rgb(var(--warn) / <alpha-value>)',
        danger: 'rgb(var(--danger) / <alpha-value>)',
        // Diff
        'diff-add': 'rgb(var(--diff-add) / <alpha-value>)',
        'diff-del': 'rgb(var(--diff-del) / <alpha-value>)',
      },
      backgroundColor: {
        'diff-add': 'var(--diff-add-bg)',
        'diff-del': 'var(--diff-del-bg)',
      },
      borderColor: {
        DEFAULT: 'var(--hairline)',
      },
      ringColor: {
        DEFAULT: 'rgb(var(--accent) / 0.45)',
      },
      fontFamily: {
        sans: [
          '"OpenAI Sans"',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          '"Inter"',
          'Roboto',
          'Helvetica',
          'Arial',
          'sans-serif',
        ],
        mono: [
          '"OpenAI Sans Mono"',
          'ui-monospace',
          '"SF Mono"',
          'SFMono-Regular',
          '"JetBrains Mono"',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
      // Codex scale: dense, weight-driven hierarchy; tracking tightens with size.
      fontSize: {
        display: ['1.75rem', { lineHeight: '2.125rem', letterSpacing: '-0.02em', fontWeight: '600' }],
        'title-1': ['1.375rem', { lineHeight: '1.75rem', letterSpacing: '-0.018em', fontWeight: '600' }],
        'title-2': ['1.125rem', { lineHeight: '1.5rem', letterSpacing: '-0.014em', fontWeight: '600' }],
        'title-3': ['1rem', { lineHeight: '1.5rem', letterSpacing: '-0.011em', fontWeight: '600' }],
        'body-lg': ['1rem', { lineHeight: '1.625rem', letterSpacing: '-0.011em' }],
        body: ['0.875rem', { lineHeight: '1.375rem', letterSpacing: '-0.006em' }],
        callout: ['0.8125rem', { lineHeight: '1.25rem', letterSpacing: '-0.002em' }],
        footnote: ['0.8125rem', { lineHeight: '1.125rem' }],
        caption: ['0.75rem', { lineHeight: '1rem', letterSpacing: '0.01em' }],
        'caption-2': ['0.6875rem', { lineHeight: '0.875rem', letterSpacing: '0.02em' }],
      },
      borderRadius: {
        sm: '6px',
        DEFAULT: '8px',
        md: '10px',
        lg: '12px',
        xl: '16px',
        '2xl': '20px',
        '3xl': '24px',
      },
      boxShadow: {
        xs: '0 1px 2px rgb(var(--shadow) / 0.06)',
        sm: 'var(--elev-1)',
        md: 'var(--elev-2)',
        lg: 'var(--elev-2)',
        xl: 'var(--elev-3)',
        edge: 'inset 0 1px 0 0 rgb(255 255 255 / 0.05)',
      },
      transitionTimingFunction: {
        out: 'cubic-bezier(0.32, 0.72, 0, 1)',
        'in-out': 'cubic-bezier(0.65, 0, 0.35, 1)',
        standard: 'cubic-bezier(0.32, 0.72, 0, 1)',
      },
      transitionDuration: {
        fast: '130ms',
      },
      keyframes: {
        blink: { '0%, 100%': { opacity: '1' }, '50%': { opacity: '0' } },
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'scale-in': {
          from: { opacity: '0', transform: 'scale(0.97) translateY(6px)' },
          to: { opacity: '1', transform: 'scale(1) translateY(0)' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          from: { transform: 'translateX(100%)' },
          to: { transform: 'translateX(0)' },
        },
        'slide-in-left': {
          from: { transform: 'translateX(-100%)' },
          to: { transform: 'translateX(0)' },
        },
      },
      animation: {
        blink: 'blink 1s step-end infinite',
        'fade-in': 'fade-in 150ms cubic-bezier(0.32, 0.72, 0, 1)',
        'scale-in': 'scale-in 190ms cubic-bezier(0.32, 0.72, 0, 1)',
        'slide-up': 'slide-up 220ms cubic-bezier(0.32, 0.72, 0, 1)',
        'slide-in-right': 'slide-in-right 260ms cubic-bezier(0.32, 0.72, 0, 1)',
        'slide-in-left': 'slide-in-left 260ms cubic-bezier(0.32, 0.72, 0, 1)',
      },
    },
  },
  plugins: [typography],
}
