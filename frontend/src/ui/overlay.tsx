import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'

import { cx } from '../lib/cx'
import { XIcon } from '../components/Icons'
import { IconButton } from './primitives'

/* ============================================================================
   Hooks
   ========================================================================== */
export function useClickOutside<T extends HTMLElement>(onOutside: () => void) {
  const ref = useRef<T>(null)
  useEffect(() => {
    function handle(event: MouseEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) onOutside()
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [onOutside])
  return ref
}

export function useEscape(onEscape: () => void) {
  useEffect(() => {
    function handle(event: KeyboardEvent) {
      if (event.key === 'Escape') onEscape()
    }
    document.addEventListener('keydown', handle)
    return () => document.removeEventListener('keydown', handle)
  }, [onEscape])
}

function useLockBodyScroll() {
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previous
    }
  }, [])
}

/* ============================================================================
   Modal
   ========================================================================== */
const MODAL_WIDTH = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
} as const

export function Modal({
  title,
  description,
  icon,
  onClose,
  size = 'md',
  footer,
  children,
  closeOnOverlay = true,
  bodyClassName,
}: {
  title?: ReactNode
  description?: ReactNode
  icon?: ReactNode
  onClose: () => void
  size?: keyof typeof MODAL_WIDTH
  footer?: ReactNode
  children: ReactNode
  closeOnOverlay?: boolean
  bodyClassName?: string
}) {
  useEscape(onClose)
  useLockBodyScroll()

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto p-4 sm:p-6">
      <div
        className="fixed inset-0 bg-black/40 backdrop-blur-[2px] animate-fade-in"
        onClick={closeOnOverlay ? onClose : undefined}
      />
      <div
        role="dialog"
        aria-modal="true"
        className={cx(
          'relative my-auto flex w-full flex-col rounded-2xl border border-hairline bg-elevated shadow-xl animate-scale-in',
          // height tied to the fixed inset-0 parent (resizes reliably in macOS
          // WKWebView) — vh units there go stale on window resize/maximize.
          'max-h-full',
          MODAL_WIDTH[size],
        )}
      >
        {(title || icon) && (
          <header className="flex items-start gap-3 border-b border-hairline px-5 py-4">
            {icon && <span className="mt-0.5 flex-none text-secondary">{icon}</span>}
            <div className="min-w-0 flex-1">
              {title && <h2 className="text-title-2 text-primary">{title}</h2>}
              {description && <p className="mt-1 text-callout leading-relaxed text-secondary">{description}</p>}
            </div>
            <IconButton size="sm" label="Fermer" onClick={onClose} className="-mr-1.5 -mt-1">
              <XIcon className="h-4 w-4" />
            </IconButton>
          </header>
        )}
        <div className={cx('min-h-0 flex-1 overflow-y-auto px-5 py-4', bodyClassName)}>{children}</div>
        {footer && (
          <footer className="flex items-center justify-end gap-2 border-t border-hairline px-5 py-3.5">
            {footer}
          </footer>
        )}
      </div>
    </div>,
    document.body,
  )
}

/* ============================================================================
   Menu / Popover — anchored dropdown
   ========================================================================== */
const MenuContext = createContext<{ close: () => void }>({ close: () => {} })

export function Menu({
  trigger,
  children,
  align = 'left',
  up = false,
  className,
}: {
  trigger: (state: { open: boolean; onClick: () => void }) => ReactNode
  children: ReactNode
  align?: 'left' | 'right'
  up?: boolean
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useClickOutside<HTMLDivElement>(() => setOpen(false))
  useEscape(() => setOpen(false))

  return (
    <div ref={ref} className="relative inline-flex">
      {trigger({ open, onClick: () => setOpen((value) => !value) })}
      {open && (
        <div
          className={cx(
            'absolute z-50 min-w-[13rem] rounded-xl border border-hairline bg-elevated p-1.5 shadow-lg animate-scale-in',
            up ? 'bottom-full mb-2' : 'top-full mt-2',
            align === 'right' ? 'right-0' : 'left-0',
            className,
          )}
        >
          <MenuContext.Provider value={{ close: () => setOpen(false) }}>{children}</MenuContext.Provider>
        </div>
      )}
    </div>
  )
}

export function MenuItem({
  icon,
  children,
  onClick,
  danger,
  active,
  trailing,
}: {
  icon?: ReactNode
  children: ReactNode
  onClick?: () => void
  danger?: boolean
  active?: boolean
  trailing?: ReactNode
}) {
  const { close } = useContext(MenuContext)
  return (
    <button
      type="button"
      onClick={() => {
        onClick?.()
        close()
      }}
      className={cx(
        'flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left text-callout transition-colors duration-fast outline-none focus-visible:bg-surface-2',
        danger
          ? 'text-danger hover:bg-danger/10'
          : active
            ? 'bg-accent/12 text-accent'
            : 'text-primary hover:bg-surface-2',
      )}
    >
      {icon && <span className="flex-none text-secondary">{icon}</span>}
      <span className="flex-1 truncate">{children}</span>
      {trailing}
    </button>
  )
}

export function MenuLabel({ children }: { children: ReactNode }) {
  return <div className="px-2.5 pb-1 pt-1.5 text-caption font-medium uppercase tracking-wide text-tertiary">{children}</div>
}

export function MenuSeparator() {
  return <div className="my-1 h-px bg-hairline" />
}
