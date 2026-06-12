import { useEffect, useState, type ReactNode } from 'react'

import { bridge } from '../bridge'
import type { AppSettings, ClaudeStatus, CodexStatus, GrokStatus, LmStudioModel } from '../types'
import { Badge, Button, Field, Input, Modal, Select, SettingRow, Textarea, Toggle } from '../ui'
import { GlobeIcon, TrashIcon } from './Icons'

const EMPTY: AppSettings = {
  custom_instructions: '',
  embedding_model: 'nomic-embed-text',
  selected_model: '',
  chat_provider: 'ollama',
  lmstudio_base_url: 'http://127.0.0.1:1234/v1',
  lmstudio_model: 'qwen2.5-coder-14b-instruct-mlx-4bit',
  codex_model: '',
  codex_reasoning_effort: 'medium',
  claude_model: '',
  claude_effort: 'medium',
  grok_model: '',
  default_agent: false,
  default_yolo: false,
  saved_prompts: [],
  web_search_enabled: false,
  searxng_url: 'http://localhost:8080',
  cross_session_memory: true,
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-3 text-caption font-semibold uppercase tracking-wide text-tertiary">{title}</h3>
      <div className="space-y-4">{children}</div>
    </section>
  )
}

export function SettingsModal({
  onClose,
  onSaved,
  onInstallWeb,
}: {
  onClose: () => void
  onSaved?: (settings: AppSettings) => void
  onInstallWeb?: () => void
}) {
  const [settings, setSettings] = useState<AppSettings>(EMPTY)
  const [saving, setSaving] = useState(false)
  const [promptDraft, setPromptDraft] = useState({ title: '', text: '' })
  const [installedModels, setInstalledModels] = useState<string[]>([])
  const [lmStudioModels, setLmStudioModels] = useState<LmStudioModel[]>([])
  const [lmStudioMessage, setLmStudioMessage] = useState('')
  const [codexStatus, setCodexStatus] = useState<CodexStatus | null>(null)
  const [codexLoginRunning, setCodexLoginRunning] = useState(false)
  const [codexLoginMessage, setCodexLoginMessage] = useState('')
  const [claudeStatus, setClaudeStatus] = useState<ClaudeStatus | null>(null)
  const [claudeLoginRunning, setClaudeLoginRunning] = useState(false)
  const [claudeLoginMessage, setClaudeLoginMessage] = useState('')
  const [grokStatus, setGrokStatus] = useState<GrokStatus | null>(null)
  const [grokLoginRunning, setGrokLoginRunning] = useState(false)
  const [grokLoginMessage, setGrokLoginMessage] = useState('')

  function addPrompt() {
    const title = promptDraft.title.trim()
    const text = promptDraft.text.trim()
    if (!title || !text) return
    setSettings((previous) => ({
      ...previous,
      saved_prompts: [...previous.saved_prompts, { title, text }],
    }))
    setPromptDraft({ title: '', text: '' })
  }

  function removePrompt(index: number) {
    setSettings((previous) => ({
      ...previous,
      saved_prompts: previous.saved_prompts.filter((_, i) => i !== index),
    }))
  }

  useEffect(() => {
    const unsubscribeCodex = bridge.on('codex_login', (payload) => {
      setCodexLoginRunning(false)
      setCodexLoginMessage(payload?.message || '')
      if (payload?.status) setCodexStatus(payload.status)
    })
    const unsubscribeClaude = bridge.on('claude_login', (payload) => {
      setClaudeLoginRunning(false)
      setClaudeLoginMessage(payload?.message || '')
      if (payload?.status) setClaudeStatus(payload.status)
    })
    const unsubscribeGrok = bridge.on('grok_login', (payload) => {
      setGrokLoginRunning(false)
      setGrokLoginMessage(payload?.message || '')
      if (payload?.status) setGrokStatus(payload.status)
    })
    bridge
      .getSettings()
      .then((value) => setSettings({ ...EMPTY, ...value }))
      .catch(() => {})
    bridge
      .listModelsDetailed()
      .then((list) => setInstalledModels(list.map((model) => model.name)))
      .catch(() => {})
    bridge
      .lmstudioModels()
      .then((catalog) => {
        setLmStudioModels(catalog.models || [])
        setLmStudioMessage(catalog.message || '')
      })
      .catch(() => {
        setLmStudioModels([])
        setLmStudioMessage('LM Studio indisponible.')
      })
    bridge
      .codexStatus()
      .then(setCodexStatus)
      .catch(() =>
        setCodexStatus({ available: false, authenticated: false, message: 'Statut Codex indisponible.' }),
      )
    bridge
      .claudeStatus()
      .then(setClaudeStatus)
      .catch(() =>
        setClaudeStatus({ available: false, authenticated: false, message: 'Statut Claude indisponible.' }),
      )
    bridge
      .grokStatus()
      .then(setGrokStatus)
      .catch(() =>
        setGrokStatus({ available: false, authenticated: false, message: 'Statut Grok indisponible.' }),
      )
    return () => {
      unsubscribeCodex()
      unsubscribeClaude()
      unsubscribeGrok()
    }
  }, [])

  function update<K extends keyof AppSettings>(key: K, value: AppSettings[K]) {
    setSettings((previous) => ({ ...previous, [key]: value }))
  }

  async function save() {
    setSaving(true)
    try {
      const updated = await bridge.updateSettings({
        custom_instructions: settings.custom_instructions,
        embedding_model: settings.embedding_model,
        lmstudio_base_url: settings.lmstudio_base_url,
        lmstudio_model: settings.lmstudio_model,
        default_agent: settings.default_agent,
        default_yolo: settings.default_yolo,
        saved_prompts: settings.saved_prompts,
        // web_search_enabled is NOT persisted here: web is a per-session toggle.
        searxng_url: settings.searxng_url,
        cross_session_memory: settings.cross_session_memory,
      })
      onSaved?.(updated)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  async function connectCodex() {
    setCodexLoginRunning(true)
    setCodexLoginMessage('Ouverture de la connexion Codex…')
    try {
      const result = await bridge.codexLogin()
      setCodexLoginRunning(result.running)
      setCodexLoginMessage(result.message)
      if (result.status) setCodexStatus(result.status)
      if (!result.running) setCodexLoginRunning(false)
    } catch (error) {
      setCodexLoginRunning(false)
      setCodexLoginMessage(String(error))
    }
  }

  async function connectClaude() {
    setClaudeLoginRunning(true)
    setClaudeLoginMessage('Ouverture de la connexion Claude…')
    try {
      const result = await bridge.claudeLogin()
      setClaudeLoginRunning(result.running)
      setClaudeLoginMessage(result.message)
      if (result.status) setClaudeStatus(result.status)
      if (!result.running) setClaudeLoginRunning(false)
    } catch (error) {
      setClaudeLoginRunning(false)
      setClaudeLoginMessage(String(error))
    }
  }

  async function connectGrok(deviceAuth = false) {
    setGrokLoginRunning(true)
    setGrokLoginMessage(deviceAuth ? 'Ouverture de la connexion Grok par code…' : 'Ouverture de la connexion Grok…')
    try {
      const result = await bridge.grokLogin(deviceAuth)
      setGrokLoginRunning(result.running)
      setGrokLoginMessage(result.message)
      if (result.status) setGrokStatus(result.status)
      if (!result.running) setGrokLoginRunning(false)
    } catch (error) {
      setGrokLoginRunning(false)
      setGrokLoginMessage(String(error))
    }
  }

  return (
    <Modal
      title="Réglages"
      size="lg"
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Annuler
          </Button>
          <Button variant="primary" onClick={save} disabled={saving}>
            {saving ? 'Enregistrement…' : 'Enregistrer'}
          </Button>
        </>
      }
    >
      <div className="space-y-7">
        <Section title="Moteur local">
          <div className="space-y-3 rounded-xl border border-hairline bg-surface-2/50 p-3.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-body font-medium text-primary">LM Studio · MLX</span>
              <Badge tone={lmStudioModels.length ? 'success' : 'warn'}>
                {lmStudioModels.length ? 'Prêt' : 'À lancer'}
              </Badge>
            </div>
            <p className="text-footnote text-tertiary">
              {lmStudioMessage || 'Serveur OpenAI local : Developer → Start Server.'}
            </p>
            <Field label="URL locale" hint="Par défaut : serveur LM Studio compatible OpenAI.">
              <Input
                value={settings.lmstudio_base_url}
                onChange={(event) => update('lmstudio_base_url', event.target.value)}
                placeholder="http://127.0.0.1:1234/v1"
              />
            </Field>
            <Field label="Modèle LM Studio" hint="Modèle chargé dans LM Studio pour le chat local.">
              {lmStudioModels.length ? (
                <Select
                  value={settings.lmstudio_model}
                  onChange={(event) => update('lmstudio_model', event.target.value)}
                >
                  {settings.lmstudio_model &&
                    !lmStudioModels.some((model) => model.slug === settings.lmstudio_model) && (
                      <option value={settings.lmstudio_model}>{settings.lmstudio_model}</option>
                    )}
                  {lmStudioModels.map((model) => (
                    <option key={model.slug} value={model.slug}>
                      {model.display_name || model.slug}
                    </option>
                  ))}
                </Select>
              ) : (
                <Input
                  value={settings.lmstudio_model}
                  onChange={(event) => update('lmstudio_model', event.target.value)}
                  placeholder="qwen2.5-coder-14b-instruct-mlx-4bit"
                />
              )}
            </Field>
          </div>
        </Section>

        <Section title="Compte Codex">
          <div className="space-y-2.5 rounded-xl border border-hairline bg-surface-2/50 p-3.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-body font-medium text-primary">OpenAI · Codex</span>
              <Badge tone={codexStatus?.authenticated ? 'success' : 'warn'}>
                {codexStatus?.authenticated ? 'Connecté' : 'À connecter'}
              </Badge>
            </div>
            <p className="text-footnote text-tertiary">{codexStatus?.message || 'Vérification de Codex…'}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="secondary" onClick={() => void connectCodex()} disabled={codexLoginRunning}>
                {codexLoginRunning
                  ? 'Connexion en cours…'
                  : codexStatus?.authenticated
                    ? 'Reconnecter Codex'
                    : 'Connecter Codex'}
              </Button>
              {codexLoginMessage && <span className="text-footnote text-tertiary">{codexLoginMessage}</span>}
            </div>
            <p className="text-caption text-tertiary">
              La connexion se fait avec `codex login` ; aucune clé API n'est stockée dans l'app.
            </p>
          </div>
        </Section>

        <Section title="Compte Claude">
          <div className="space-y-2.5 rounded-xl border border-hairline bg-surface-2/50 p-3.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-body font-medium text-primary">Anthropic · Claude</span>
              <Badge tone={claudeStatus?.authenticated ? 'success' : 'warn'}>
                {claudeStatus?.authenticated ? 'Connecté' : 'À connecter'}
              </Badge>
            </div>
            <p className="text-footnote text-tertiary">{claudeStatus?.message || 'Vérification de Claude…'}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="secondary" onClick={() => void connectClaude()} disabled={claudeLoginRunning}>
                {claudeLoginRunning
                  ? 'Connexion en cours…'
                  : claudeStatus?.authenticated
                    ? 'Reconnecter Claude'
                    : 'Connecter Claude'}
              </Button>
              {claudeLoginMessage && <span className="text-footnote text-tertiary">{claudeLoginMessage}</span>}
            </div>
            <p className="text-caption text-tertiary">
              La connexion utilise `claude setup-token` (OAuth navigateur) ; aucune clé API n'est
              stockée dans l'app. Le CLI Claude gère son authentification. Une clé
              `ANTHROPIC_API_KEY` ou un `claude` déjà connecté fonctionnent aussi.
            </p>
          </div>
        </Section>

        <Section title="Compte Grok">
          <div className="space-y-2.5 rounded-xl border border-hairline bg-surface-2/50 p-3.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-body font-medium text-primary">xAI · Grok</span>
              <Badge tone={grokStatus?.authenticated ? 'success' : 'warn'}>
                {grokStatus?.authenticated ? 'Connecté' : 'À connecter'}
              </Badge>
            </div>
            <p className="text-footnote text-tertiary">{grokStatus?.message || 'Vérification de Grok…'}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="secondary" onClick={() => void connectGrok()} disabled={grokLoginRunning}>
                {grokLoginRunning
                  ? 'Connexion en cours…'
                  : grokStatus?.authenticated
                    ? 'Reconnecter Grok'
                    : 'Connecter Grok'}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => void connectGrok(true)} disabled={grokLoginRunning}>
                Code appareil
              </Button>
              {grokLoginMessage && <span className="text-footnote text-tertiary">{grokLoginMessage}</span>}
            </div>
            <p className="text-caption text-tertiary">
              Nécessite le CLI Grok Build (`curl -fsSL https://x.ai/cli/install.sh | bash`). La
              connexion se fait avec `grok login` ou `grok login --device-auth` ; une clé `XAI_API_KEY`
              dans l'environnement fonctionne aussi. Aucune clé n'est stockée dans l'app.
            </p>
          </div>
        </Section>

        <Section title="Général">
          <Field
            label="Instructions personnalisées"
            hint="Ajoutées au prompt système pour toutes les conversations."
          >
            <Textarea
              value={settings.custom_instructions}
              onChange={(event) => update('custom_instructions', event.target.value)}
              rows={4}
              placeholder="Ex. : Réponds toujours en français, de façon concise, avec des exemples de code."
            />
          </Field>

          <Field
            label="Modèle d'embedding (RAG)"
            hint="Modèles Ollama installés. Choisis un modèle dédié comme nomic-embed-text (Modèles → Embeddings)."
          >
            <Select
              value={settings.embedding_model}
              onChange={(event) => update('embedding_model', event.target.value)}
            >
              {settings.embedding_model && !installedModels.includes(settings.embedding_model) && (
                <option value={settings.embedding_model}>{settings.embedding_model} (non installé)</option>
              )}
              {installedModels.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
              {installedModels.length === 0 && !settings.embedding_model && (
                <option value="">Aucun modèle installé</option>
              )}
            </Select>
          </Field>
        </Section>

        <Section title="Agent">
          <SettingRow
            title="Mode agent par défaut"
            description="L'IA inspecte le projet avec des outils avant de répondre."
            control={<Toggle checked={settings.default_agent} onChange={(v) => update('default_agent', v)} />}
          />
          <SettingRow
            title="Mode autonome par défaut"
            description="L'agent écrit et exécute sans valider chaque changement. À réserver à un workspace de confiance."
            control={<Toggle checked={settings.default_yolo} onChange={(v) => update('default_yolo', v)} />}
          />
        </Section>

        <Section title="Recherche web">
          <Field
            label="URL SearXNG"
            hint="SearXNG (auto-hébergé, sans clé) est prioritaire. Sans URL, repli sur DuckDuckGo puis Wikipédia. La recherche s'active via la pastille « Web » sous le chat (par session)."
          >
            <Input
              value={settings.searxng_url}
              onChange={(event) => update('searxng_url', event.target.value)}
              placeholder="http://localhost:8080"
            />
          </Field>
          {onInstallWeb && (
            <Button size="sm" variant="secondary" icon={<GlobeIcon className="h-4 w-4" />} onClick={onInstallWeb}>
              Installer / configurer SearXNG (Docker)
            </Button>
          )}
        </Section>

        <Section title="Mémoire">
          <SettingRow
            title="Mémoire inter-sessions"
            description="Indexe tes conversations et rappelle le contexte pertinent. 100 % local, recherche hybride."
            control={
              <Toggle checked={settings.cross_session_memory} onChange={(v) => update('cross_session_memory', v)} />
            }
          />
        </Section>

        <Section title="Prompts enregistrés · ⌘K">
          {settings.saved_prompts.length > 0 && (
            <div className="space-y-1">
              {settings.saved_prompts.map((prompt, index) => (
                <div
                  key={index}
                  className="group flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2 text-callout"
                >
                  <span className="min-w-0 flex-1 truncate text-primary">{prompt.title}</span>
                  <button
                    type="button"
                    onClick={() => removePrompt(index)}
                    aria-label="Supprimer"
                    className="flex-none rounded p-1 text-tertiary opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                  >
                    <TrashIcon className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={promptDraft.title}
              onChange={(event) => setPromptDraft({ ...promptDraft, title: event.target.value })}
              placeholder="Titre"
              className="w-32"
            />
            <Input
              value={promptDraft.text}
              onChange={(event) => setPromptDraft({ ...promptDraft, text: event.target.value })}
              placeholder="Texte du prompt"
              className="min-w-0 flex-1"
            />
            <Button variant="secondary" onClick={addPrompt}>
              Ajouter
            </Button>
          </div>
        </Section>
      </div>
    </Modal>
  )
}
