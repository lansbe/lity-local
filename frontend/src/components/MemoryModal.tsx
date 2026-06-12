import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import type { MemoryData } from '../types'
import { Button, Input, Modal, Select } from '../ui'
import { BrainIcon, TrashIcon } from './Icons'

const EMPTY: MemoryData = { user_profile: {}, assistant_profile: {}, facts: {} }

const SECTIONS: { key: keyof MemoryData; label: string }[] = [
  { key: 'user_profile', label: 'Sur toi' },
  { key: 'assistant_profile', label: "Sur l'assistant local" },
  { key: 'facts', label: 'Faits' },
]

export function MemoryModal({ onClose }: { onClose: () => void }) {
  const [memory, setMemory] = useState<MemoryData>(EMPTY)
  const [draft, setDraft] = useState<{ category: keyof MemoryData; key: string; value: string }>({
    category: 'user_profile',
    key: '',
    value: '',
  })

  useEffect(() => {
    bridge
      .getMemory()
      .then((value) => setMemory({ ...EMPTY, ...value }))
      .catch(() => {})
  }, [])

  async function addEntry() {
    const key = draft.key.trim()
    if (!key) return
    const updated = await bridge.updateMemoryEntry(draft.category, key, draft.value)
    setMemory({ ...EMPTY, ...updated })
    setDraft({ ...draft, key: '', value: '' })
  }

  async function remove(category: keyof MemoryData, key: string) {
    const updated = await bridge.deleteMemoryEntry(category, key)
    setMemory({ ...EMPTY, ...updated })
  }

  async function clearAll() {
    if (!window.confirm('Tout effacer de la mémoire long terme ?')) return
    const updated = await bridge.clearMemory()
    setMemory({ ...EMPTY, ...updated })
  }

  const isEmpty = SECTIONS.every((section) => Object.keys(memory[section.key]).length === 0)

  return (
    <Modal
      title="Mémoire"
      icon={<BrainIcon className="h-5 w-5" />}
      onClose={onClose}
      footer={
        <>
          <button
            type="button"
            onClick={clearAll}
            className="mr-auto text-footnote text-tertiary transition-colors hover:text-danger"
          >
            Tout effacer
          </button>
          <Button variant="primary" onClick={onClose}>
            Fermer
          </Button>
        </>
      }
    >
      <div className="space-y-5">
        {isEmpty && (
          <p className="py-6 text-center text-callout leading-relaxed text-tertiary">
            Aucune mémoire pour l'instant. La mémoire retient automatiquement ce que tu dis (prénom,
            préférences…), et tu peux ajouter des entrées ici.
          </p>
        )}
        {SECTIONS.map((section) => {
          const entries = Object.entries(memory[section.key])
          if (entries.length === 0) return null
          return (
            <div key={section.key}>
              <p className="mb-1.5 text-caption font-medium uppercase tracking-wide text-tertiary">
                {section.label}
              </p>
              <div className="space-y-1">
                {entries.map(([key, value]) => (
                  <div
                    key={key}
                    className="group flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2 text-callout"
                  >
                    <span className="font-medium text-primary">{key}</span>
                    <span className="min-w-0 flex-1 truncate text-secondary">{value}</span>
                    <button
                      type="button"
                      onClick={() => remove(section.key, key)}
                      aria-label="Supprimer"
                      className="flex-none rounded p-1 text-tertiary opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                    >
                      <TrashIcon className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )
        })}

        <div className="border-t border-hairline pt-4">
          <p className="mb-2 text-footnote font-medium text-secondary">Ajouter / mettre à jour</p>
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={draft.category}
              onChange={(event) => setDraft({ ...draft, category: event.target.value as keyof MemoryData })}
              className="w-auto"
            >
              {SECTIONS.map((section) => (
                <option key={section.key} value={section.key}>
                  {section.label}
                </option>
              ))}
            </Select>
            <Input
              value={draft.key}
              onChange={(event) => setDraft({ ...draft, key: event.target.value })}
              placeholder="clé (ex. prénom)"
              className="w-32"
            />
            <Input
              value={draft.value}
              onChange={(event) => setDraft({ ...draft, value: event.target.value })}
              placeholder="valeur"
              className="min-w-0 flex-1"
            />
            <Button variant="primary" onClick={addEntry}>
              Ajouter
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  )
}
