import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import type { HealthItem } from '../types'
import { cx } from '../lib/cx'
import { useClickOutside } from '../ui'
import { ChevronDownIcon } from './Icons'

/** Service-health status pill for the sidebar footer, with an upward popover. */
export function HealthMenu() {
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState<HealthItem[]>([])
  const ref = useClickOutside<HTMLDivElement>(() => setOpen(false))

  async function refresh() {
    try {
      setItems(await bridge.getHealth())
    } catch {
      setItems([])
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const down = items.filter((item) => !item.ok).length
  const known = items.length > 0
  const tone = !known ? 'bg-tertiary' : down === 0 ? 'bg-success' : 'bg-warn'
  const label = !known ? 'Services' : down === 0 ? 'Tout fonctionne' : `${down} hors ligne`

  return (
    <div ref={ref} className="relative flex-1">
      <button
        type="button"
        onClick={() => {
          const next = !open
          setOpen(next)
          if (next) void refresh()
        }}
        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-footnote text-secondary transition-colors hover:bg-surface-2 hover:text-primary"
      >
        <span className={cx('h-2 w-2 flex-none rounded-full', tone)} />
        <span className="flex-1 truncate text-left">{label}</span>
        <ChevronDownIcon className="h-3.5 w-3.5 flex-none text-tertiary" />
      </button>
      {open && (
        <div className="absolute bottom-full left-0 z-50 mb-2 w-72 rounded-xl border border-hairline bg-elevated p-1.5 shadow-lg animate-scale-in">
          {items.length === 0 && (
            <p className="px-2 py-3 text-center text-footnote text-tertiary">Chargement…</p>
          )}
          {items.map((item) => (
            <div key={item.name} className="flex items-start gap-2.5 rounded-md px-2 py-1.5">
              <span
                className={cx('mt-1.5 h-2 w-2 flex-none rounded-full', item.ok ? 'bg-success' : 'bg-tertiary')}
              />
              <div className="min-w-0">
                <div className="text-callout text-primary">{item.name}</div>
                <div className="text-caption text-tertiary">{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
