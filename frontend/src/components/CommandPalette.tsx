import { useEffect, useMemo, useState } from 'react'

import { cx } from '../lib/cx'
import type { ConversationMeta, SavedPrompt } from '../types'
import { Kbd } from '../ui'
import { SearchIcon } from './Icons'

interface PaletteAction {
  id: string
  label: string
  run: () => void
}

interface CommandPaletteProps {
  onClose: () => void
  conversations: ConversationMeta[]
  prompts: SavedPrompt[]
  actions: PaletteAction[]
  onSelectConversation: (id: string) => void
  onUsePrompt: (text: string) => void
}

interface Item {
  id: string
  label: string
  hint: string
  run: () => void
}

export function CommandPalette(props: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)

  const items = useMemo<Item[]>(() => {
    const base: Item[] = [
      ...props.actions.map((action) => ({
        id: `a:${action.id}`,
        label: action.label,
        hint: 'Action',
        run: () => {
          action.run()
          props.onClose()
        },
      })),
      ...props.prompts.map((prompt, i) => ({
        id: `p:${i}`,
        label: prompt.title,
        hint: 'Prompt',
        run: () => {
          props.onUsePrompt(prompt.text)
          props.onClose()
        },
      })),
      ...props.conversations.map((conversation) => ({
        id: `c:${conversation.id}`,
        label: conversation.title,
        hint: 'Conversation',
        run: () => {
          props.onSelectConversation(conversation.id)
          props.onClose()
        },
      })),
    ]
    const needle = query.trim().toLowerCase()
    return needle ? base.filter((item) => item.label.toLowerCase().includes(needle)) : base
  }, [query, props])

  useEffect(() => {
    setIndex(0)
  }, [query])

  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setIndex((i) => Math.min(i + 1, items.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setIndex((i) => Math.max(i - 1, 0))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      items[index]?.run()
    } else if (event.key === 'Escape') {
      props.onClose()
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 p-4 pt-32 animate-fade-in"
      onClick={props.onClose}
    >
      <div
        className="material w-full max-w-xl overflow-hidden rounded-2xl border border-hairline shadow-xl animate-scale-in"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center gap-2.5 border-b border-hairline px-3.5 py-3">
          <SearchIcon className="h-4 w-4 flex-none text-tertiary" />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Rechercher une action, un prompt, une conversation…"
            className="w-full bg-transparent text-body text-primary outline-none placeholder:text-tertiary"
          />
        </div>
        <div className="max-h-[22rem] overflow-y-auto p-1.5">
          {items.length === 0 && (
            <p className="px-3 py-8 text-center text-callout text-tertiary">Aucun résultat</p>
          )}
          {items.map((item, i) => (
            <button
              key={item.id}
              type="button"
              onClick={item.run}
              onMouseEnter={() => setIndex(i)}
              className={cx(
                'flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-callout transition-colors',
                i === index ? 'bg-accent/12 text-primary' : 'text-secondary',
              )}
            >
              <span className="truncate">{item.label}</span>
              <span
                className={cx(
                  'flex-none rounded-full px-2 py-0.5 text-caption font-medium',
                  i === index ? 'bg-accent/15 text-accent' : 'bg-surface-2 text-tertiary',
                )}
              >
                {item.hint}
              </span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 border-t border-hairline px-3.5 py-2 text-caption text-tertiary">
          <span className="flex items-center gap-1">
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd>
            naviguer
          </span>
          <span className="flex items-center gap-1">
            <Kbd>↵</Kbd>
            ouvrir
          </span>
          <span className="flex items-center gap-1">
            <Kbd>esc</Kbd>
            fermer
          </span>
        </div>
      </div>
    </div>
  )
}
