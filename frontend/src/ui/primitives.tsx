import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes, TextareaHTMLAttributes } from 'react'

import { cx } from '../lib/cx'

/* ============================================================================
   Button
   ========================================================================== */
type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
type ButtonSize = 'sm' | 'md' | 'lg'

const BTN_BASE =
  'inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium transition-[background-color,color,box-shadow,transform] duration-fast ease-out outline-none focus-visible:ring-2 focus-visible:ring-accent/40 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40'

const BTN_VARIANT: Record<ButtonVariant, string> = {
  primary: 'bg-solid text-solid-contrast hover:bg-solid-hover active:bg-solid',
  secondary:
    'bg-surface text-primary border border-hairline-strong hover:bg-surface-2 active:bg-surface-3',
  ghost: 'text-secondary hover:bg-surface-2 hover:text-primary active:bg-surface-3',
  danger: 'bg-danger text-white hover:opacity-90 active:scale-[0.98]',
}

const BTN_SIZE: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-footnote',
  md: 'h-9 px-3.5 text-callout',
  lg: 'h-10 px-4 text-body',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  icon?: ReactNode
  block?: boolean
}

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  block,
  className,
  children,
  type = 'button',
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(BTN_BASE, BTN_VARIANT[variant], BTN_SIZE[size], block && 'w-full', className)}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
}

/* ============================================================================
   IconButton — square, icon-only
   ========================================================================== */
interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  size?: 'sm' | 'md'
  active?: boolean
  label: string
}

export function IconButton({
  size = 'md',
  active = false,
  label,
  className,
  children,
  type = 'button',
  ...rest
}: IconButtonProps) {
  return (
    <button
      type={type}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cx(
        'inline-flex flex-none items-center justify-center rounded-md transition-colors duration-fast ease-out outline-none focus-visible:ring-2 focus-visible:ring-accent/40 active:scale-[0.96] disabled:pointer-events-none disabled:opacity-40',
        size === 'sm' ? 'h-8 w-8' : 'h-9 w-9',
        active
          ? 'bg-accent/12 text-accent'
          : 'text-secondary hover:bg-surface-2 hover:text-primary',
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

/* ============================================================================
   Badge / Pill
   ========================================================================== */
type BadgeTone = 'neutral' | 'accent' | 'success' | 'warn' | 'danger'

const BADGE_TONE: Record<BadgeTone, string> = {
  neutral: 'bg-surface-2 text-secondary',
  accent: 'bg-accent/12 text-accent',
  success: 'bg-success/12 text-success',
  warn: 'bg-warn/14 text-warn',
  danger: 'bg-danger/12 text-danger',
}

export function Badge({
  tone = 'neutral',
  className,
  title,
  children,
}: {
  tone?: BadgeTone
  className?: string
  title?: string
  children: ReactNode
}) {
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-caption font-medium',
        BADGE_TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/** A small status dot in a semantic tone. */
export function Dot({ tone = 'neutral', className }: { tone?: BadgeTone | 'off'; className?: string }) {
  const color =
    tone === 'success'
      ? 'bg-success'
      : tone === 'warn'
        ? 'bg-warn'
        : tone === 'danger'
          ? 'bg-danger'
          : tone === 'accent'
            ? 'bg-accent'
            : 'bg-tertiary'
  return <span className={cx('h-1.5 w-1.5 flex-none rounded-full', color, className)} />
}

/* ============================================================================
   Toggle — Apple-style switch
   ========================================================================== */
export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label?: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx(
        'relative inline-flex h-[22px] w-[38px] flex-none items-center rounded-full transition-colors duration-150 ease-out outline-none focus-visible:ring-2 focus-visible:ring-accent/40 disabled:opacity-40',
        checked ? 'bg-accent' : 'bg-hairline-strong',
      )}
    >
      <span
        className={cx(
          'inline-block h-[18px] w-[18px] transform rounded-full bg-white shadow-sm transition-transform duration-150 ease-out',
          checked ? 'translate-x-[18px]' : 'translate-x-[2px]',
        )}
      />
    </button>
  )
}

/* ============================================================================
   Form controls
   ========================================================================== */
const CONTROL_BASE =
  'w-full rounded-md border border-hairline bg-surface text-body text-primary placeholder:text-tertiary transition-colors duration-fast outline-none focus:border-accent/70 disabled:opacity-50'

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cx(CONTROL_BASE, 'h-9 px-3', className)} {...rest} />
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cx(CONTROL_BASE, 'resize-none px-3 py-2 leading-relaxed', className)} {...rest} />
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cx(CONTROL_BASE, 'h-9 cursor-pointer px-3 pr-8', className)} {...rest}>
      {children}
    </select>
  )
}

/** Labelled form row: label + control + optional hint. */
export function Field({
  label,
  hint,
  htmlFor,
  className,
  children,
}: {
  label?: string
  hint?: ReactNode
  htmlFor?: string
  className?: string
  children: ReactNode
}) {
  return (
    <div className={cx('flex flex-col gap-1.5', className)}>
      {label && (
        <label htmlFor={htmlFor} className="text-footnote font-medium text-secondary">
          {label}
        </label>
      )}
      {children}
      {hint && <p className="text-caption leading-relaxed text-tertiary">{hint}</p>}
    </div>
  )
}

/** A settings/list row with a title, optional description, and a trailing control. */
export function SettingRow({
  title,
  description,
  control,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  control: ReactNode
  className?: string
}) {
  return (
    <div className={cx('flex items-center justify-between gap-4 py-0.5', className)}>
      <div className="min-w-0">
        <div className="text-body text-primary">{title}</div>
        {description && <div className="mt-0.5 text-caption leading-relaxed text-tertiary">{description}</div>}
      </div>
      <div className="flex-none">{control}</div>
    </div>
  )
}

/* ============================================================================
   Segmented control — for tabs and small mode pickers
   ========================================================================== */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
  className,
}: {
  options: { value: T; label: ReactNode }[]
  value: T
  onChange: (value: T) => void
  size?: 'sm' | 'md'
  className?: string
}) {
  return (
    <div className={cx('inline-flex gap-0.5 rounded-lg bg-surface-2 p-0.5', className)}>
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={cx(
              'inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors duration-fast outline-none focus-visible:ring-2 focus-visible:ring-accent/40',
              size === 'sm' ? 'h-7 px-2.5 text-footnote' : 'h-8 px-3 text-callout',
              active ? 'bg-surface text-primary shadow-xs' : 'text-secondary hover:text-primary',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

/* ============================================================================
   Spinner + Kbd
   ========================================================================== */
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        'inline-block animate-spin rounded-full border-2 border-current border-r-transparent align-[-0.125em]',
        className || 'h-4 w-4',
      )}
      role="status"
      aria-label="Chargement"
    />
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-5 min-w-[20px] items-center justify-center rounded border border-hairline bg-surface px-1 font-sans text-caption-2 font-medium text-secondary">
      {children}
    </kbd>
  )
}
