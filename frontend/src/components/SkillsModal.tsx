import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import type { Skill, SkillsResult } from '../types'
import { Badge, Button, Field, Input, Modal, Textarea, Toggle } from '../ui'
import { PlusIcon, PuzzleIcon, TrashIcon } from './Icons'

const EMPTY: SkillsResult = { enabled: false, semantic: false, dir: '', skills: [] }
const EMPTY_DRAFT = { name: '', description: '', whenToUse: '', body: '' }

export function SkillsModal({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<SkillsResult>(EMPTY)
  const [notice, setNotice] = useState('')
  const [creating, setCreating] = useState(false)
  const [draft, setDraft] = useState(EMPTY_DRAFT)
  const [busy, setBusy] = useState(false)

  async function refresh() {
    try {
      setData(await bridge.listSkills())
    } catch {
      setData(EMPTY)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  const skills = data.skills
  const custom = skills.filter((skill) => !skill.builtin)
  const builtin = skills.filter((skill) => skill.builtin)

  async function setMaster(enabled: boolean) {
    setData((prev) => ({ ...prev, enabled }))
    try {
      await bridge.updateSettings({ skills_enabled: enabled })
    } catch {
      void refresh()
    }
  }

  async function setSemantic(enabled: boolean) {
    setData((prev) => ({ ...prev, semantic: enabled }))
    try {
      await bridge.updateSettings({ skills_semantic: enabled })
    } catch {
      void refresh()
    }
  }

  async function toggle(skill: Skill) {
    const next = !skill.enabled
    setData((prev) => ({
      ...prev,
      skills: prev.skills.map((item) => (item.name === skill.name ? { ...item, enabled: next } : item)),
    }))
    try {
      await bridge.toggleSkill(skill.name, next)
    } catch {
      void refresh()
    }
  }

  async function remove(skill: Skill) {
    if (!window.confirm(`Supprimer la compétence « ${skill.name} » ?`)) return
    try {
      const result = await bridge.deleteSkill(skill.name)
      setNotice(result.message ?? '')
    } finally {
      void refresh()
    }
  }

  async function create() {
    if (!draft.name.trim() || (!draft.description.trim() && !draft.body.trim())) {
      setNotice('Donne au moins un nom et une description (ou un contenu).')
      return
    }
    setBusy(true)
    try {
      const result = await bridge.createSkill(draft.name, draft.description, draft.body, draft.whenToUse)
      setNotice(result.message ?? '')
      if (result.ok) {
        setDraft(EMPTY_DRAFT)
        setCreating(false)
        void refresh()
      }
    } finally {
      setBusy(false)
    }
  }

  function SkillRow({ skill }: { skill: Skill }) {
    return (
      <div className="group flex items-start gap-2 rounded-lg bg-surface-2 px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="truncate font-mono text-callout text-primary">{skill.name}</span>
            {skill.builtin && <Badge tone="neutral">Intégrée</Badge>}
          </div>
          <p className="mt-0.5 line-clamp-2 text-caption text-tertiary">{skill.description}</p>
        </div>
        <div className="flex flex-none items-center gap-1.5 pt-0.5">
          <Toggle
            checked={skill.enabled}
            onChange={() => void toggle(skill)}
            label={`Activer ${skill.name}`}
            disabled={!data.enabled}
          />
          {!skill.builtin && (
            <button
              type="button"
              onClick={() => void remove(skill)}
              title="Supprimer"
              aria-label={`Supprimer ${skill.name}`}
              className="rounded p-1 text-tertiary opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
            >
              <TrashIcon className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <Modal
      title="Compétences"
      size="lg"
      icon={<PuzzleIcon className="h-5 w-5" />}
      onClose={onClose}
      bodyClassName="space-y-4"
      footer={
        <Button variant="primary" onClick={onClose}>
          Fermer
        </Button>
      }
    >
      <p className="text-callout text-secondary">
        Des savoir-faire locaux appliqués quand ta demande s'y prête — à la manière des Skills de
        Claude Code et Codex. 100 % local, aucune donnée envoyée.
      </p>

      <div className="flex items-center justify-between gap-3 rounded-lg bg-surface-2 px-3 py-2">
        <div className="min-w-0">
          <div className="text-callout text-primary">Activer les compétences</div>
          <div className="text-caption text-tertiary">
            Injecte le savoir-faire pertinent dans le contexte, en chat comme en mode agent.
          </div>
        </div>
        <Toggle checked={data.enabled} onChange={(value) => void setMaster(value)} label="Activer les compétences" />
      </div>

      <div className="flex items-center justify-between gap-3 rounded-lg bg-surface-2 px-3 py-2">
        <div className="min-w-0">
          <div className="text-callout text-primary">Correspondance sémantique</div>
          <div className="text-caption text-tertiary">
            Utilise le modèle d'embedding en plus des mots-clés — plus précis, un peu plus lent.
          </div>
        </div>
        <Toggle
          checked={data.semantic}
          onChange={(value) => void setSemantic(value)}
          label="Correspondance sémantique"
          disabled={!data.enabled}
        />
      </div>

      {custom.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-caption font-medium uppercase tracking-wide text-tertiary">Mes compétences</p>
          {custom.map((skill) => (
            <SkillRow key={skill.name} skill={skill} />
          ))}
        </div>
      )}

      <div className="space-y-1.5">
        <p className="text-caption font-medium uppercase tracking-wide text-tertiary">Intégrées</p>
        {builtin.map((skill) => (
          <SkillRow key={skill.name} skill={skill} />
        ))}
        {skills.length === 0 && (
          <p className="text-callout text-tertiary">Aucune compétence pour l'instant.</p>
        )}
      </div>

      {creating ? (
        <div className="space-y-2.5 rounded-lg border border-hairline bg-surface-2/50 p-3">
          <Field label="Nom">
            <Input
              value={draft.name}
              onChange={(event) => setDraft({ ...draft, name: event.target.value })}
              placeholder="ex. : traduction-fr-en"
            />
          </Field>
          <Field label="Description — ce que ça fait ET quand l'utiliser (c'est ce qui déclenche la compétence)">
            <Textarea
              rows={2}
              value={draft.description}
              onChange={(event) => setDraft({ ...draft, description: event.target.value })}
              placeholder="Traduit un texte entre le français et l'anglais. À utiliser quand l'utilisateur demande une traduction…"
            />
          </Field>
          <Field label="Méthode / instructions">
            <Textarea
              rows={5}
              value={draft.body}
              onChange={(event) => setDraft({ ...draft, body: event.target.value })}
              placeholder={'# Traduction\n1. Garde le sens exact.\n2. …'}
            />
          </Field>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setCreating(false)}>
              Annuler
            </Button>
            <Button variant="primary" onClick={() => void create()} disabled={busy}>
              Créer
            </Button>
          </div>
        </div>
      ) : (
        <Button
          variant="secondary"
          icon={<PlusIcon className="h-4 w-4" />}
          onClick={() => {
            setNotice('')
            setCreating(true)
          }}
        >
          Nouvelle compétence
        </Button>
      )}

      {data.dir && (
        <p className="text-caption text-tertiary">
          Dossier : <span className="font-mono">{data.dir}</span>
        </p>
      )}
      {notice && <p className="text-caption text-tertiary">{notice}</p>}
    </Modal>
  )
}
