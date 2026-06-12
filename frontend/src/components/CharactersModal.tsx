import { useEffect, useMemo, useState } from 'react'

import { bridge } from '../bridge'
import type { CharacterProfile, CharactersResult } from '../types'
import { Badge, Button, Field, Input, Modal, Select, Spinner, Textarea } from '../ui'
import { CheckIcon, ImageIcon, PlusIcon, SparklesIcon, TrashIcon } from './Icons'

type CharacterDraft = {
  name: string
  description: string
  gender: string
  style: string
  instructions: string
  voice: string
  image_model: string
  seed: string
}

const EMPTY_RESULT: CharactersResult = {
  characters: [],
  active_character_id: '',
  active_character: null,
}

const NEW_ID = '__new__'

const EMPTY_DRAFT: CharacterDraft = {
  name: '',
  description: '',
  gender: '',
  style: 'portrait réaliste, lumière douce',
  instructions: '',
  voice: '',
  image_model: '',
  seed: '-1',
}

function draftFromCharacter(character: CharacterProfile | null): CharacterDraft {
  if (!character) return EMPTY_DRAFT
  return {
    name: character.name || '',
    description: character.description || '',
    gender: character.gender || '',
    style: character.style || '',
    instructions: character.instructions || '',
    voice: character.voice || '',
    image_model: character.image_model || '',
    seed: String(character.seed ?? -1),
  }
}

function draftPayload(draft: CharacterDraft): Partial<CharacterProfile> {
  const seed = Number.parseInt(draft.seed, 10)
  return {
    name: draft.name.trim(),
    description: draft.description.trim(),
    gender: draft.gender.trim(),
    style: draft.style.trim(),
    instructions: draft.instructions.trim(),
    voice: draft.voice.trim(),
    image_model: draft.image_model.trim(),
    seed: Number.isFinite(seed) ? seed : -1,
  }
}

function firstImage(character: CharacterProfile | null): string {
  if (!character) return ''
  return (
    character.thumbnail ||
    character.emotions.neutral?.image ||
    Object.values(character.emotions).find((emotion) => emotion.image)?.image ||
    ''
  )
}

export function CharactersModal({
  activeCharacter,
  onActiveChange,
  onClose,
}: {
  activeCharacter: CharacterProfile | null
  onActiveChange: (character: CharacterProfile | null) => void
  onClose: () => void
}) {
  const [data, setData] = useState<CharactersResult>(EMPTY_RESULT)
  const [selectedId, setSelectedId] = useState(NEW_ID)
  const [draft, setDraft] = useState<CharacterDraft>(EMPTY_DRAFT)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [generating, setGenerating] = useState(false)

  const activeId = data.active_character_id || activeCharacter?.id || ''
  const selected = useMemo(
    () => data.characters.find((character) => character.id === selectedId) || null,
    [data.characters, selectedId],
  )
  const isNew = selectedId === NEW_ID

  async function refresh(preferredId?: string) {
    try {
      const result = await bridge.listCharacters()
      setData({ ...EMPTY_RESULT, ...result })
      if (result.active_character !== undefined) onActiveChange(result.active_character ?? null)
      const nextId =
        preferredId ||
        selectedId ||
        result.active_character_id ||
        result.characters[0]?.id ||
        NEW_ID
      const next =
        nextId !== NEW_ID && result.characters.some((character) => character.id === nextId)
          ? nextId
          : result.characters[0]?.id || NEW_ID
      setSelectedId(next)
      setDraft(draftFromCharacter(result.characters.find((character) => character.id === next) || null))
    } catch (error) {
      setNotice(`Chargement impossible : ${String(error)}`)
    }
  }

  useEffect(() => {
    void refresh(activeCharacter?.id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function startNew() {
    setSelectedId(NEW_ID)
    setDraft(EMPTY_DRAFT)
    setNotice('')
  }

  function choose(character: CharacterProfile) {
    setSelectedId(character.id)
    setDraft(draftFromCharacter(character))
    setNotice('')
  }

  function applyResult(result: CharactersResult, preferredId?: string) {
    const next = { ...EMPTY_RESULT, ...result }
    setData(next)
    if (result.active_character !== undefined) onActiveChange(result.active_character ?? null)
    const nextId =
      preferredId ||
      result.character?.id ||
      (selectedId !== NEW_ID ? selectedId : '') ||
      next.characters[0]?.id ||
      NEW_ID
    const found = next.characters.find((character) => character.id === nextId) || null
    setSelectedId(found?.id || NEW_ID)
    setDraft(draftFromCharacter(found))
    setNotice(result.message || (result.ok === false ? 'Action impossible.' : ''))
  }

  async function save() {
    const payload = draftPayload(draft)
    if (!payload.name) {
      setNotice('Nom requis.')
      return
    }
    setBusy(true)
    try {
      const result = isNew
        ? await bridge.createCharacter(payload)
        : await bridge.updateCharacter(selectedId, payload)
      applyResult(result, result.character?.id || selectedId)
      if (result.ok) setNotice(isNew ? 'Personnage créé.' : 'Personnage enregistré.')
    } catch (error) {
      setNotice(`Enregistrement impossible : ${String(error)}`)
    } finally {
      setBusy(false)
    }
  }

  async function activate() {
    if (!selected) return
    setBusy(true)
    try {
      applyResult(await bridge.setConversationCharacter(selected.id), selected.id)
    } finally {
      setBusy(false)
    }
  }

  async function clearActive() {
    setBusy(true)
    try {
      applyResult(await bridge.setConversationCharacter(''), selectedId)
      onActiveChange(null)
    } finally {
      setBusy(false)
    }
  }

  async function remove() {
    if (!selected) return
    if (!window.confirm(`Supprimer « ${selected.name} » ?`)) return
    setBusy(true)
    try {
      const result = await bridge.deleteCharacter(selected.id)
      applyResult(result, result.characters[0]?.id || NEW_ID)
    } finally {
      setBusy(false)
    }
  }

  async function generateEmotions() {
    if (!selected) return
    setGenerating(true)
    try {
      const result = await bridge.generateCharacterEmotions(selected.id)
      applyResult(result, selected.id)
      setNotice(result.message || '')
    } catch (error) {
      setNotice(`Génération impossible : ${String(error)}`)
    } finally {
      setGenerating(false)
    }
  }

  const preview = firstImage(selected)
  const emotions = selected ? Object.entries(selected.emotions) : []

  return (
    <Modal
      title="Personnages"
      icon={<SparklesIcon className="h-5 w-5" />}
      onClose={onClose}
      size="xl"
      bodyClassName="p-0"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Fermer
          </Button>
          <Button variant="primary" onClick={() => void save()} disabled={busy}>
            {busy ? <Spinner className="h-4 w-4" /> : null}
            Enregistrer
          </Button>
        </>
      }
    >
      <div className="grid min-h-[32rem] grid-cols-1 md:grid-cols-[16rem_minmax(0,1fr)]">
        <aside className="border-b border-hairline bg-surface-2/50 p-3 md:border-b-0 md:border-r">
          <Button
            block
            variant={isNew ? 'primary' : 'secondary'}
            icon={<PlusIcon className="h-4 w-4" />}
            onClick={startNew}
          >
            Nouveau
          </Button>

          <div className="mt-3 space-y-1.5">
            {data.characters.map((character) => {
              const image = firstImage(character)
              const active = character.id === activeId
              const selectedRow = character.id === selectedId
              return (
                <button
                  key={character.id}
                  type="button"
                  onClick={() => choose(character)}
                  className={`flex w-full items-center gap-2 rounded-md px-2 py-2 text-left transition-colors ${
                    selectedRow ? 'bg-accent/12 text-primary' : 'text-secondary hover:bg-surface'
                  }`}
                >
                  <span className="flex h-9 w-9 flex-none items-center justify-center overflow-hidden rounded-md bg-surface">
                    {image ? (
                      <img src={image} alt="" className="h-full w-full object-cover" />
                    ) : (
                      <SparklesIcon className="h-4 w-4 text-tertiary" />
                    )}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-callout font-medium">
                      {character.name}
                    </span>
                    <span className="block truncate text-caption text-tertiary">
                      {character.description || character.gender || 'Profil local'}
                    </span>
                  </span>
                  {active && <CheckIcon className="h-3.5 w-3.5 flex-none text-accent" />}
                </button>
              )
            })}
          </div>
        </aside>

        <section className="min-w-0 space-y-5 p-4">
          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="flex h-28 w-28 flex-none items-center justify-center overflow-hidden rounded-md border border-hairline bg-surface-2">
              {preview ? (
                <img src={preview} alt="" className="h-full w-full object-cover" />
              ) : (
                <ImageIcon className="h-8 w-8 text-tertiary" />
              )}
            </div>
            <div className="min-w-0 flex-1 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="truncate text-title-3 text-primary">
                  {isNew ? 'Nouveau personnage' : selected?.name || 'Personnage'}
                </h3>
                {!isNew && selected?.id === activeId && <Badge tone="accent">Actif</Badge>}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="secondary"
                  icon={<CheckIcon className="h-4 w-4" />}
                  onClick={() => void activate()}
                  disabled={!selected || selected.id === activeId || busy}
                >
                  Activer
                </Button>
                <Button
                  variant="secondary"
                  icon={<SparklesIcon className="h-4 w-4" />}
                  onClick={() => void generateEmotions()}
                  disabled={!selected || generating}
                >
                  {generating ? <Spinner className="h-4 w-4" /> : null}
                  Émotions
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => void clearActive()}
                  disabled={!activeId || busy}
                >
                  Aucun
                </Button>
                <Button
                  variant="ghost"
                  icon={<TrashIcon className="h-4 w-4" />}
                  onClick={() => void remove()}
                  disabled={!selected || busy}
                >
                  Supprimer
                </Button>
              </div>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Nom">
              <Input
                value={draft.name}
                onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                placeholder="Mira"
              />
            </Field>
            <Field label="Genre">
              <Select
                value={draft.gender}
                onChange={(event) => setDraft({ ...draft, gender: event.target.value })}
              >
                <option value="">Non précisé</option>
                <option value="femme">Femme</option>
                <option value="homme">Homme</option>
                <option value="androgyne">Androgyne</option>
              </Select>
            </Field>
          </div>

          <Field label="Description">
            <Textarea
              rows={3}
              value={draft.description}
              onChange={(event) => setDraft({ ...draft, description: event.target.value })}
              placeholder="Âge, style, visage, énergie, détails distinctifs…"
            />
          </Field>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Style visuel">
              <Input
                value={draft.style}
                onChange={(event) => setDraft({ ...draft, style: event.target.value })}
                placeholder="portrait réaliste, lumière douce"
              />
            </Field>
            <Field label="Modèle image">
              <Input
                value={draft.image_model}
                onChange={(event) => setDraft({ ...draft, image_model: event.target.value })}
                placeholder="checkpoint local"
              />
            </Field>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Voix">
              <Input
                value={draft.voice}
                onChange={(event) => setDraft({ ...draft, voice: event.target.value })}
                placeholder="voix locale optionnelle"
              />
            </Field>
            <Field label="Seed">
              <Input
                value={draft.seed}
                onChange={(event) => setDraft({ ...draft, seed: event.target.value })}
                inputMode="numeric"
              />
            </Field>
          </div>

          <Field label="Instructions">
            <Textarea
              rows={4}
              value={draft.instructions}
              onChange={(event) => setDraft({ ...draft, instructions: event.target.value })}
              placeholder="Ton, rôle, limites et façon de répondre quand ce personnage est actif."
            />
          </Field>

          {emotions.length > 0 && (
            <div className="space-y-2">
              <div className="text-caption font-medium uppercase tracking-wide text-tertiary">
                Pack d'émotions
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {emotions.map(([key, emotion]) => (
                  <div
                    key={key}
                    className="overflow-hidden rounded-md border border-hairline bg-surface"
                  >
                    <div className="aspect-square bg-surface-2">
                      {emotion.image ? (
                        <img
                          src={emotion.image}
                          alt=""
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="flex h-full items-center justify-center">
                          <ImageIcon className="h-5 w-5 text-tertiary" />
                        </div>
                      )}
                    </div>
                    <div className="truncate px-2 py-1.5 text-caption text-secondary">
                      {emotion.label || key}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {notice && <p className="text-caption text-tertiary">{notice}</p>}
        </section>
      </div>
    </Modal>
  )
}
