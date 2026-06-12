import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import { Button, Field, Modal, SettingRow, Textarea, Toggle } from '../ui'
import { SparklesIcon } from './Icons'

/** Per-conversation instructions: extra system prompt + optional temperature. */
export function ConversationInstructionsModal({ onClose }: { onClose: () => void }) {
  const [instructions, setInstructions] = useState('')
  const [useTemp, setUseTemp] = useState(false)
  const [temp, setTemp] = useState(0.7)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    bridge
      .getConversationInstructions()
      .then((data) => {
        setInstructions(data.instructions || '')
        if (typeof data.temperature === 'number') {
          setUseTemp(true)
          setTemp(data.temperature)
        }
      })
      .catch(() => {})
  }, [])

  async function save() {
    setSaving(true)
    try {
      await bridge.setConversationInstructions(instructions, useTemp ? temp : null)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      title="Instructions de conversation"
      icon={<SparklesIcon className="h-5 w-5" />}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Annuler
          </Button>
          <Button variant="primary" onClick={save} disabled={saving}>
            Enregistrer
          </Button>
        </>
      }
    >
      <div className="space-y-5">
        <Field
          label="Instructions (prompt système, pour CETTE conversation)"
          hint="S'ajoute aux instructions globales, seulement pour cette conversation."
        >
          <Textarea
            value={instructions}
            onChange={(event) => setInstructions(event.target.value)}
            rows={5}
            placeholder="Ex. : Tu es une codeuse senior, concise, qui répond en français avec du code commenté."
          />
        </Field>

        <div>
          <SettingRow
            title="Créativité personnalisée"
            description="Remplace la température par défaut du modèle pour cette conversation."
            control={<Toggle checked={useTemp} onChange={setUseTemp} label="Activer la température" />}
          />
          {useTemp && (
            <div className="mt-3 flex items-center gap-3">
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.1}
                value={temp}
                onChange={(event) => setTemp(Number(event.target.value))}
                className="flex-1 accent-accent"
              />
              <span className="w-24 text-right text-callout tabular-nums text-secondary">
                {temp.toFixed(1)} · {temp <= 0.3 ? 'précis' : temp >= 1 ? 'créatif' : 'équilibré'}
              </span>
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}
