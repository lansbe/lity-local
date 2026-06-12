import { useCallback, useEffect, useRef, useState } from 'react'

import { bridge } from './bridge'
import { ChatHeader } from './components/ChatHeader'
import {
  CommandApprovalBanner,
  type PendingApproval,
} from './components/CommandApprovalBanner'
import { Composer } from './components/Composer'
import { EmptyState } from './components/EmptyState'
import { MessageList } from './components/MessageList'
import { ModeBar } from './components/ModeBar'
import { ModelPicker, type CliProviderMenu } from './components/ModelPicker'
import { CommandPalette } from './components/CommandPalette'
import { MemoryModal } from './components/MemoryModal'
import { ModelsModal } from './components/ModelsModal'
import { SkillsModal } from './components/SkillsModal'
import { WebSetupModal } from './components/WebSetupModal'
import { ConversationInstructionsModal } from './components/ConversationInstructionsModal'
import { CharactersModal } from './components/CharactersModal'
import { SettingsModal } from './components/SettingsModal'
import { Sidebar } from './components/Sidebar'
import { UsageModal } from './components/UsageModal'
import { WorkspacePanel } from './components/WorkspacePanel'
import { toggleImageSession } from './lib/imageSession'
import { toggleVideoSession } from './lib/videoSession'
import { splitReasoning } from './lib/reasoning'
import { useInitialBackendLoad } from './hooks/useInitialBackendLoad'
import { useIsCompact, useIsMobile } from './hooks/useMediaQuery'
import { useTheme } from './hooks/useTheme'
import { cx } from './lib/cx'
import type {
  Change,
  CharacterProfile,
  ChatMessage,
  ChatProvider,
  ClaudeModel,
  ClaudeReasoningEffort,
  ClaudeReasoningLevel,
  ClaudeStatus,
  CodexModel,
  CodexReasoningEffort,
  CodexReasoningLevel,
  CodexStatus,
  ConversationMeta,
  GrokModel,
  GrokStatus,
  LmStudioModel,
  LmStudioModelsResult,
  LoadedFile,
  SavedPrompt,
  SendResult,
  StepEvent,
  WebSetupOutcome,
  WebStatus,
} from './types'

const FALLBACK_CODEX_REASONING: CodexReasoningLevel[] = [
  { effort: 'medium', description: '' },
]

const FALLBACK_CLAUDE_REASONING: ClaudeReasoningLevel[] = [
  { effort: 'medium', description: '' },
]

export default function App() {
  const { theme, toggle } = useTheme()

  const [, setAssistantName] = useState('Assistant')
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [chatProvider, setChatProvider] = useState<ChatProvider>('ollama')
  const [codexStatus, setCodexStatus] = useState<CodexStatus | null>(null)
  const [codexModels, setCodexModels] = useState<CodexModel[]>([])
  const [codexDefaultModel, setCodexDefaultModel] = useState('')
  const [codexModel, setCodexModel] = useState('')
  const [codexReasoningEffort, setCodexReasoningEffort] =
    useState<CodexReasoningEffort>('medium')
  const [claudeStatus, setClaudeStatus] = useState<ClaudeStatus | null>(null)
  const [claudeModels, setClaudeModels] = useState<ClaudeModel[]>([])
  const [claudeDefaultModel, setClaudeDefaultModel] = useState('')
  const [claudeModel, setClaudeModel] = useState('')
  const [claudeEffort, setClaudeEffort] = useState<ClaudeReasoningEffort>('medium')
  const [grokStatus, setGrokStatus] = useState<GrokStatus | null>(null)
  const [grokModels, setGrokModels] = useState<GrokModel[]>([])
  const [grokDefaultModel, setGrokDefaultModel] = useState('')
  const [grokModel, setGrokModel] = useState('')
  const [lmStudioModels, setLmStudioModels] = useState<LmStudioModel[]>([])
  const [lmStudioRecommended, setLmStudioRecommended] = useState<LmStudioModel[]>([])
  const [lmStudioDefaultModel, setLmStudioDefaultModel] = useState('')
  const [lmStudioModel, setLmStudioModel] = useState('')
  const [workdir, setWorkdir] = useState('')

  const [conversations, setConversations] = useState<ConversationMeta[]>([])
  const [activeId, setActiveId] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeCharacter, setActiveCharacter] = useState<CharacterProfile | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [warning, setWarning] = useState('')
  const [genStats, setGenStats] = useState<{
    tokens_per_sec: number
    context_used: number
    context_length: number
    usage_pct: number
  } | null>(null)

  const [prefill, setPrefill] = useState('')
  const [prefillNonce, setPrefillNonce] = useState(0)

  // Shell — responsive layout
  const isMobile = useIsMobile() // < 768px
  const isCompact = useIsCompact() // < 1024px
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Workspace (Codex-style code panel)
  const [workspaceOpen, setWorkspaceOpen] = useState(false)
  const [changes, setChanges] = useState<Change[]>([])
  const [files, setFiles] = useState<string[]>([])
  const [loadedFiles, setLoadedFiles] = useState<LoadedFile[]>([])
  const [changeCount, setChangeCount] = useState(0)
  const changeSeq = useRef(0)
  // Set when the user hits Stop, so the late result of the in-flight turn is
  // ignored (we already froze the message and freed the UI).
  const cancelledRef = useRef(false)

  // Agent mode (Codex-style tool loop)
  const [agentMode, setAgentMode] = useState(false)
  const [allowCommands, setAllowCommands] = useState(false)
  const [yolo, setYolo] = useState(false)
  const [webSearch, setWebSearch] = useState(false)
  const [webSetup, setWebSetup] = useState<WebStatus | null>(null)
  // Hands-free voice conversation loop.
  const [voiceMode, setVoiceMode] = useState(false)
  const [relisten, setRelisten] = useState(0)
  const voiceModeRef = useRef(false)

  // RAG + conversation search
  const [ragEnabled, setRagEnabled] = useState(false)
  const [indexedChunks, setIndexedChunks] = useState(0)
  const [indexing, setIndexing] = useState(false)
  const [search, setSearch] = useState('')
  const [searchMatchIds, setSearchMatchIds] = useState<string[] | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [instructionsOpen, setInstructionsOpen] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [usageOpen, setUsageOpen] = useState(false)
  const [modelsOpen, setModelsOpen] = useState(false)
  const [skillsOpen, setSkillsOpen] = useState(false)
  const [charactersOpen, setCharactersOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [imageActive, setImageActive] = useState(false)
  const [videoActive, setVideoActive] = useState(false)
  const [commandApproval, setCommandApproval] = useState(false)
  const [savedPrompts, setSavedPrompts] = useState<SavedPrompt[]>([])
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null)

  const pendingChanges = changes.filter((change) => change.status === 'pending').length

  const refreshCodexCatalog = useCallback(async () => {
    try {
      const [status, catalog] = await Promise.all([bridge.codexStatus(), bridge.codexModels()])
      setCodexStatus(status)
      setCodexModels(catalog.models || [])
      setCodexDefaultModel(catalog.default_model || '')
      return { status, catalog }
    } catch {
      setCodexStatus(null)
      setCodexModels([])
      setCodexDefaultModel('')
      return null
    }
  }, [])

  const refreshClaudeCatalog = useCallback(async () => {
    try {
      const [status, catalog] = await Promise.all([bridge.claudeStatus(), bridge.claudeModels()])
      setClaudeStatus(status)
      setClaudeModels(catalog.models || [])
      setClaudeDefaultModel(catalog.default_model || '')
      return { status, catalog }
    } catch {
      setClaudeStatus(null)
      setClaudeModels([])
      setClaudeDefaultModel('')
      return null
    }
  }, [])

  const refreshGrokCatalog = useCallback(async () => {
    try {
      const [status, catalog] = await Promise.all([bridge.grokStatus(), bridge.grokModels()])
      setGrokStatus(status)
      setGrokModels(catalog.models || [])
      setGrokDefaultModel(catalog.default_model || '')
      return { status, catalog }
    } catch {
      setGrokStatus(null)
      setGrokModels([])
      setGrokDefaultModel('')
      return null
    }
  }, [])

  const refreshLmStudioCatalog = useCallback(async () => {
    try {
      const catalog: LmStudioModelsResult = await bridge.lmstudioModels()
      setLmStudioModels(catalog.models || [])
      setLmStudioRecommended(catalog.recommended || [])
      setLmStudioDefaultModel(catalog.default_model || '')
      return { catalog }
    } catch {
      setLmStudioModels([])
      setLmStudioRecommended([])
      setLmStudioDefaultModel('')
      return null
    }
  }, [])

  const loadInitial = useInitialBackendLoad({
    refreshCodexCatalog,
    refreshClaudeCatalog,
    refreshGrokCatalog,
    refreshLmStudioCatalog,
    setActiveId,
    setAgentMode,
    setAllowCommands,
    setAssistantName,
    setChangeCount,
    setChatProvider,
    setCodexModel,
    setCodexReasoningEffort,
    setClaudeModel,
    setClaudeEffort,
    setGrokModel,
    setLmStudioModel,
    setCommandApproval,
    setConversations,
    setError,
    setFiles,
    setImageActive,
    setVideoActive,
    setIndexedChunks,
    setLoadedFiles,
    setMessages,
    setModels,
    setRagEnabled,
    setSavedPrompts,
    setActiveCharacter,
    setSelectedModel,
    setWebSearch,
    setWorkdir,
    setYolo,
  })

  useEffect(() => {
    void loadInitial()
  }, [loadInitial])

  // Responsive: the sidebar is open by default on ≥md, closed (drawer) on mobile.
  useEffect(() => {
    setSidebarOpen(!isMobile)
  }, [isMobile])

  // Give-way cascade: when the workspace opens on a narrow window, the sidebar
  // yields the room (the chat column is the protected pane).
  useEffect(() => {
    if (workspaceOpen && isCompact) setSidebarOpen(false)
  }, [workspaceOpen, isCompact])

  // Append streamed tokens to the trailing (pending) assistant message.
  useEffect(() => {
    return bridge.on('chunk', (payload: { content: string }) => {
      setMessages((previous) => {
        const lastIndex = previous.length - 1
        if (lastIndex < 0) return previous
        const last = previous[lastIndex]
        if (last.role !== 'assistant') return previous
        const next = [...previous]
        next[lastIndex] = { ...last, content: last.content + (payload?.content ?? ''), pending: true }
        return next
      })
    })
  }, [])

  // Append agent tool steps to the trailing assistant message (agent mode).
  useEffect(() => {
    return bridge.on('step', (payload: StepEvent) => {
      setMessages((previous) => {
        const lastIndex = previous.length - 1
        if (lastIndex < 0) return previous
        const last = previous[lastIndex]
        if (last.role !== 'assistant') return previous
        const next = [...previous]
        next[lastIndex] = { ...last, steps: [...(last.steps ?? []), payload], pending: true }
        return next
      })
    })
  }, [])

  // Debounced full-text conversation search (title + message contents).
  useEffect(() => {
    const query = search.trim()
    if (!query) {
      setSearchMatchIds(null)
      return
    }
    const handle = setTimeout(async () => {
      try {
        const matches = await bridge.searchConversations(query)
        setSearchMatchIds(matches.map((match) => match.id))
      } catch {
        setSearchMatchIds(null)
      }
    }, 200)
    return () => clearTimeout(handle)
  }, [search])

  // Per-command approval requests emitted by the agent loop.
  useEffect(() => {
    return bridge.on('approval_request', (payload: PendingApproval) => setPendingApproval(payload))
  }, [])

  useEffect(() => {
    return bridge.on('codex_login', () => {
      void refreshCodexCatalog()
    })
  }, [])

  useEffect(() => {
    return bridge.on('claude_login', () => {
      void refreshClaudeCatalog()
    })
  }, [])

  useEffect(() => {
    return bridge.on('grok_login', () => {
      void refreshGrokCatalog()
    })
  }, [])

  // Background AI-generated title (first message of a conversation).
  useEffect(() => {
    return bridge.on('title_update', (payload: { conversations?: ConversationMeta[] }) => {
      if (payload?.conversations) setConversations(payload.conversations)
    })
  }, [])

  // Hands-free voice loop: when speech playback finishes, re-arm the mic.
  useEffect(() => {
    voiceModeRef.current = voiceMode
  }, [voiceMode])
  useEffect(() => {
    return bridge.on('tts_done', () => {
      if (voiceModeRef.current) setRelisten((value) => value + 1)
    })
  }, [])

  // Global keyboard shortcuts: Cmd/Ctrl+K (palette), Cmd/Ctrl+N (new chat).
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const mod = event.metaKey || event.ctrlKey
      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPaletteOpen((open) => !open)
      } else if (mod && event.key.toLowerCase() === 'n') {
        event.preventDefault()
        void handleNew()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Merge so streamed steps/content are preserved when finalizing the turn.
  function finalizeLast(update: ChatMessage) {
    setMessages((previous) => {
      const lastIndex = previous.length - 1
      if (lastIndex < 0) return previous
      const next = [...previous]
      next[lastIndex] = { ...previous[lastIndex], ...update, pending: false }
      return next
    })
  }

  function applyResult(result: SendResult, elapsedMs?: number) {
    const type = result.type
    if (type === 'ai_response' || type === 'text') {
      finalizeLast({
        role: 'assistant',
        content: result.content ?? '',
        image: result.image,
        elapsedMs,
      })
    } else if (type === 'image_generation_result') {
      // The image is the whole message — no "générée avec …" caption noise.
      finalizeLast({ role: 'assistant', content: '', image: result.image, elapsedMs })
      setImageActive(false) // the session ends after a generation
    } else if (type === 'image_parameters_proposal') {
      finalizeLast({ role: 'assistant', content: result.message ?? '' })
    } else if (type === 'image_cancelled' || type === 'image_mode_ready' || type === 'image_dependency' || type === 'image_normal_chat') {
      finalizeLast({ role: 'system', content: result.message ?? '' })
      if (type === 'image_cancelled') setImageActive(false)
    } else if (type === 'video_generation_result') {
      // The clip is the whole message — no caption noise.
      finalizeLast({ role: 'assistant', content: '', video: result.video, elapsedMs })
      setVideoActive(false) // the session ends after a generation
    } else if (type === 'video_parameters_proposal') {
      finalizeLast({ role: 'assistant', content: result.message ?? '' })
    } else if (type === 'video_cancelled' || type === 'video_mode_ready' || type === 'video_dependency' || type === 'video_normal_chat') {
      finalizeLast({ role: 'system', content: result.message ?? '' })
      if (type === 'video_cancelled') setVideoActive(false)
    } else if (type === 'slash' || type === 'intent_handled') {
      finalizeLast({ role: 'system', content: result.message ?? '' })
    } else if (type === 'error') {
      finalizeLast({ role: 'system', content: `Erreur : ${result.message ?? ''}` })
    } else {
      finalizeLast({ role: 'assistant', content: result.content ?? result.message ?? '', elapsedMs })
    }
    // Out-of-band notice for the turn (e.g. the active model can't read the
    // attached image). Currently the only channel — the backend otherwise drops
    // it — so surface it as a system message right after the answer.
    if (result.system_notification && (type === 'ai_response' || type === 'text')) {
      pushSystem(result.system_notification)
    }
    if (result.conversations) setConversations(result.conversations)
    if (result.active_conversation_id) setActiveId(result.active_conversation_id)
    if (result.active_character !== undefined) setActiveCharacter(result.active_character ?? null)
    if (typeof result.change_count === 'number') setChangeCount(result.change_count)
    registerChanges(result)
    // Reflect any files the agent created/edited in autonomous mode in the tree.
    if (workdir) {
      void bridge
        .listWorkspaceFiles()
        .then((workspace) => setFiles(workspace.files || []))
        .catch(() => {})
    }
    // Refresh the speed / context-usage HUD from the latest turn's metrics.
    if (type === 'ai_response' || type === 'text') {
      void bridge
        .generationStats()
        .then(setGenStats)
        .catch(() => {})
      // Hands-free: read the answer aloud (tts_done then re-arms the mic).
      if (voiceModeRef.current && result.content) {
        const answer = splitReasoning(result.content).answer || result.content
        void bridge.speak(answer).catch(() => {})
      }
    }
  }

  async function handleSend(text: string, images?: string[]) {
    if (busy) return
    cancelledRef.current = false
    setBusy(true)
    setMessages((previous) => [
      ...previous,
      { role: 'user', content: text, images },
      { role: 'assistant', content: '', pending: true },
    ])
    try {
      const started = performance.now()
      const result = await bridge.sendMessage(text, activeId || undefined, images)
      if (!cancelledRef.current) applyResult(result, Math.round(performance.now() - started))
    } catch (error) {
      if (!cancelledRef.current) finalizeLast({ role: 'system', content: `Erreur : ${String(error)}` })
    } finally {
      if (!cancelledRef.current) setBusy(false)
    }
  }

  async function handleRegenerate() {
    if (busy) return
    cancelledRef.current = false
    setBusy(true)
    setMessages((previous) => {
      const lastIndex = previous.length - 1
      if (lastIndex < 0 || previous[lastIndex].role !== 'assistant') return previous
      const next = [...previous]
      next[lastIndex] = { role: 'assistant', content: '', pending: true }
      return next
    })
    try {
      const started = performance.now()
      const result = await bridge.regenerate()
      if (!cancelledRef.current) applyResult(result, Math.round(performance.now() - started))
    } catch (error) {
      if (!cancelledRef.current) finalizeLast({ role: 'system', content: `Erreur : ${String(error)}` })
    } finally {
      if (!cancelledRef.current) setBusy(false)
    }
  }

  async function handleEditResend(newText: string) {
    if (busy || !newText) return
    cancelledRef.current = false
    setBusy(true)
    setMessages((previous) => {
      const next = [...previous]
      if (next.length && next[next.length - 1].role === 'assistant') next.pop()
      for (let index = next.length - 1; index >= 0; index--) {
        if (next[index].role === 'user') {
          next[index] = { role: 'user', content: newText }
          break
        }
      }
      next.push({ role: 'assistant', content: '', pending: true })
      return next
    })
    try {
      const started = performance.now()
      const result = await bridge.editAndResend(newText)
      if (!cancelledRef.current) applyResult(result, Math.round(performance.now() - started))
    } catch (error) {
      if (!cancelledRef.current) finalizeLast({ role: 'system', content: `Erreur : ${String(error)}` })
    } finally {
      if (!cancelledRef.current) setBusy(false)
    }
  }

  function registerChanges(result: SendResult) {
    const fresh: Change[] = []
    for (const block of result.create_blocks ?? []) {
      fresh.push({ id: `chg-${changeSeq.current++}`, kind: 'create', block, status: 'pending' })
    }
    for (const block of result.edit_blocks ?? []) {
      fresh.push({ id: `chg-${changeSeq.current++}`, kind: 'edit', block, status: 'pending' })
    }
    if (fresh.length > 0) {
      setChanges((previous) => [...previous, ...fresh])
      setWorkspaceOpen(true)
    }
  }

  async function applyChange(change: Change) {
    const result =
      change.kind === 'create'
        ? await bridge.applyCreate(change.block)
        : await bridge.applyEdit(change.block)
    setChanges((previous) =>
      previous.map((item) =>
        item.id === change.id
          ? { ...item, status: result.success ? 'applied' : 'error', message: result.message }
          : item,
      ),
    )
    if (result.success && result.files) setFiles(result.files)
    if (typeof result.change_count === 'number') setChangeCount(result.change_count)
  }

  async function handleUndo() {
    const result = await bridge.undoChange()
    setChangeCount(result.change_count ?? 0)
    if (result.files) setFiles(result.files)
  }

  function rejectChange(change: Change) {
    setChanges((previous) =>
      previous.map((item) => (item.id === change.id ? { ...item, status: 'rejected' } : item)),
    )
  }

  async function handleLoadFile(path: string) {
    const result = await bridge.loadContextFile(path)
    setLoadedFiles(result.loaded || [])
  }

  async function handleCloseFile(path: string) {
    const result = await bridge.closeContextFile(path)
    setLoadedFiles(result.loaded || [])
  }

  async function handleRefreshFiles() {
    const result = await bridge.listWorkspaceFiles()
    setFiles(result.files || [])
    setWorkdir(result.workdir || '')
  }

  function handleStop() {
    cancelledRef.current = true
    void bridge.stop()
    // Optimistic: freeze the in-flight assistant message (stop the typing
    // indicator) and free the composer right away, instead of waiting for the
    // backend turn to wind down. The late result is ignored (cancelledRef).
    setMessages((previous) => {
      const lastIndex = previous.length - 1
      if (lastIndex < 0 || previous[lastIndex].role !== 'assistant') return previous
      const next = [...previous]
      const last = previous[lastIndex]
      next[lastIndex] = {
        ...last,
        pending: false,
        content: last.content || '_Génération arrêtée._',
      }
      return next
    })
    setBusy(false)
  }

  async function handleNew() {
    // A new conversation is a draft: it only appears in the sidebar once the
    // first message is sent (the send result returns the refreshed list).
    const meta = await bridge.newConversation()
    setActiveId(meta.id)
    setActiveCharacter(meta.active_character ?? null)
    setMessages([])
    if (isMobile) setSidebarOpen(false)
  }

  async function handleSelect(id: string) {
    if (id === activeId || busy) return
    const result = await bridge.switchConversation(id)
    setActiveId(result.active_conversation_id || id)
    setMessages(result.messages || [])
    setWorkdir(result.workdir || '')
    setFiles(result.files || [])
    if (result.model) setSelectedModel(result.model)
    setActiveCharacter(result.active_character ?? null)
    if (isMobile) setSidebarOpen(false)
  }

  async function handlePin(id: string, pinned: boolean) {
    const result = await bridge.setPinned(id, pinned)
    setConversations(result.conversations || [])
  }

  async function handleExport(id: string) {
    const result = await bridge.exportConversation(id, 'markdown')
    if (!result.ok) return
    const blob = new Blob([result.content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = result.filename || 'conversation.md'
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  async function toggleCommandApproval() {
    const result = await bridge.setCommandApproval(!commandApproval)
    setCommandApproval(result.command_approval)
  }

  function approve(allow: boolean) {
    if (!pendingApproval) return
    void bridge.approveCommand(pendingApproval.id, allow)
    setPendingApproval(null)
  }

  async function refreshModels() {
    const list = await bridge.listModels()
    setModels(list.models || [])
    setSelectedModel(list.selected || '')
  }

  async function handleRename(id: string, title: string) {
    const result = await bridge.renameConversation(id, title)
    if (result.conversations) setConversations(result.conversations)
  }

  async function handleDelete(id: string) {
    const result = await bridge.deleteConversation(id)
    setConversations(result.conversations || [])
    setActiveId(result.active_conversation_id || '')
    setMessages(result.messages || [])
    setActiveCharacter(result.active_character ?? null)
  }

  async function handleSelectChatModel(value: string) {
    if (value.startsWith('lmstudio:')) {
      const slug = value.slice('lmstudio:'.length)
      setChatProvider('lmstudio')
      setLmStudioModel(slug)
      await bridge.updateSettings({ chat_provider: 'lmstudio', lmstudio_model: slug })
      return
    }

    if (value.startsWith('codex:')) {
      const slug = value.slice('codex:'.length)
      const model = codexModels.find((candidate) => candidate.slug === slug)
      const supported = model?.supported_reasoning_levels.map((level) => level.effort) || []
      const reasoning = supported.includes(codexReasoningEffort)
        ? codexReasoningEffort
        : model?.default_reasoning_level || 'medium'
      setChatProvider('codex')
      setCodexModel(slug)
      setCodexReasoningEffort(reasoning)
      await bridge.updateSettings({
        chat_provider: 'codex',
        codex_model: slug,
        codex_reasoning_effort: reasoning,
      })
      return
    }

    if (value.startsWith('claude:')) {
      const slug = value.slice('claude:'.length)
      const model = claudeModels.find((candidate) => candidate.slug === slug)
      const supported = model?.supported_reasoning_levels.map((level) => level.effort) || []
      const effort = supported.includes(claudeEffort)
        ? claudeEffort
        : model?.default_reasoning_level || 'medium'
      setChatProvider('claude')
      setClaudeModel(slug)
      setClaudeEffort(effort)
      await bridge.updateSettings({
        chat_provider: 'claude',
        claude_model: slug,
        claude_effort: effort,
      })
      return
    }

    if (value.startsWith('grok:')) {
      const slug = value.slice('grok:'.length)
      setChatProvider('grok')
      setGrokModel(slug)
      await bridge.updateSettings({ chat_provider: 'grok', grok_model: slug })
      return
    }

    const localModel = value.startsWith('ollama:') ? value.slice('ollama:'.length) : value
    if (!localModel) return
    await bridge.updateSettings({ chat_provider: 'ollama' })
    const result = await bridge.setModel(localModel)
    setChatProvider('ollama')
    setSelectedModel(result.selected || localModel)
  }

  async function handleSelectCodexReasoning(effort: CodexReasoningEffort) {
    setCodexReasoningEffort(effort)
    await bridge.updateSettings({ codex_reasoning_effort: effort })
  }

  async function handleSelectClaudeReasoning(effort: ClaudeReasoningEffort) {
    setClaudeEffort(effort)
    await bridge.updateSettings({ claude_effort: effort })
  }

  async function handleChooseWorkdir() {
    const result = await bridge.chooseWorkdir()
    if (result.workdir) {
      setWorkdir(result.workdir)
      const workspace = await bridge.listWorkspaceFiles()
      setFiles(workspace.files || [])
      setLoadedFiles((await bridge.getLoadedFiles()) || [])
    }
  }

  // Warn when an agent/web feature is turned on with a model that can't tool-call.
  async function warnIfNoTools(enabled: boolean) {
    if (!enabled) return
    try {
      const { model, supports } = await bridge.modelSupportsTools()
      if (supports === false) {
        setWarning(
          `Le modèle « ${model} » ne gère pas les outils : la recherche web et le mode agent ne ` +
            `fonctionneront pas. Choisis un modèle compatible (llama3.1, qwen2.5, mistral…).`,
        )
      }
    } catch {
      /* unknown — stay silent */
    }
  }

  async function toggleAgent() {
    const result = await bridge.setAgentMode(!agentMode)
    setAgentMode(result.agent_mode)
    void warnIfNoTools(result.agent_mode)
  }

  async function toggleCommands() {
    const result = await bridge.setAllowCommands(!allowCommands)
    setAllowCommands(result.allow_commands)
  }

  async function toggleYolo() {
    const result = await bridge.setYolo(!yolo)
    setYolo(result.yolo)
    setAgentMode(result.agent_mode)
    void warnIfNoTools(result.yolo)
  }

  async function toggleWebSearch() {
    if (!webSearch) {
      // First activation: if no search engine actually answers AND the user
      // hasn't already decided, offer the one-click local SearXNG install
      // instead of pretending web works. The modal is NOT marked as resolved
      // here — only an explicit choice inside it counts, so dismissing it keeps
      // the offer available next time.
      try {
        const status = await bridge.webStatus()
        if (!status.reachable && !status.setup_resolved) {
          setWebSetup(status)
          return
        }
      } catch {
        // status check unavailable → fall through to the plain toggle
      }
    }
    const result = await bridge.setWebSearch(!webSearch)
    setWebSearch(result.web_search)
    setAgentMode(result.agent_mode)
    void warnIfNoTools(result.web_search)
  }

  /** Open the web-setup modal deliberately (e.g. from Settings), bypassing the
   *  "already resolved" guard so the user can (re)install SearXNG any time. */
  async function openWebSetup() {
    try {
      setWebSetup(await bridge.webStatus())
    } catch {
      // ignore — nothing to show if the status call fails
    }
  }

  async function finishWebSetup(outcome: WebSetupOutcome) {
    setWebSetup(null)
    // Dismissing (clicked away / closed) is NOT a decision: leave web off and
    // keep offering the setup on the next Web click.
    if (outcome === 'dismissed') return
    // 'fallback' or 'installed' are real choices → remember them so the modal
    // stops auto-appearing, and turn web search on.
    void bridge.markWebSetupResolved().catch(() => {})
    const result = await bridge.setWebSearch(true)
    setWebSearch(result.web_search)
    setAgentMode(result.agent_mode)
    void warnIfNoTools(result.web_search)
  }

  function toggleVoice() {
    setVoiceMode((on) => {
      if (on) void bridge.stopSpeaking().catch(() => {})
      return !on
    })
  }

  async function handleIndexProject() {
    setIndexing(true)
    try {
      const result = await bridge.indexProject()
      setIndexedChunks(result.chunks ?? 0)
      if (result.ok) {
        setRagEnabled(true)
        setError('')
      } else if (result.message) {
        setError(result.message)
      }
    } finally {
      setIndexing(false)
    }
  }

  async function toggleRag() {
    const result = await bridge.setRag(!ragEnabled)
    setRagEnabled(result.rag_enabled)
  }

  function pushSystem(content: string) {
    setMessages((previous) => [...previous, { role: 'system', content }])
  }

  async function toggleImage() {
    await toggleImageSession(imageActive, { setImageActive, pushSystem })
  }

  async function toggleVideo() {
    await toggleVideoSession(videoActive, { setVideoActive, pushSystem })
  }

  function pickSuggestion(text: string) {
    setPrefill(text)
    setPrefillNonce((nonce) => nonce + 1)
  }

  const activeTitle =
    conversations.find((conversation) => conversation.id === activeId)?.title ||
    'Lity'

  const visibleConversations = searchMatchIds
    ? conversations.filter((conversation) => searchMatchIds.includes(conversation.id))
    : conversations

  const activeCodexModelSlug = codexModel || codexDefaultModel
  const activeCodexModel = codexModels.find((model) => model.slug === activeCodexModelSlug)
  const codexReasoningLevels = activeCodexModel?.supported_reasoning_levels.length
    ? activeCodexModel.supported_reasoning_levels
    : FALLBACK_CODEX_REASONING

  const activeClaudeModelSlug = claudeModel || claudeDefaultModel
  const activeClaudeModel = claudeModels.find((model) => model.slug === activeClaudeModelSlug)
  // When the model is known, use its real effort levels as-is — an empty list
  // (e.g. Haiku, which has no effort) correctly hides the effort menu. Only fall
  // back when the catalogue hasn't loaded yet (model not found).
  const claudeReasoningLevels = activeClaudeModel
    ? activeClaudeModel.supported_reasoning_levels
    : FALLBACK_CLAUDE_REASONING

  const activeGrokModelSlug = grokModel || grokDefaultModel
  const activeLmStudioModelSlug = lmStudioModel || lmStudioDefaultModel
  const lmStudioPickerModels = lmStudioModels.length ? lmStudioModels : lmStudioRecommended

  const cliProviders: CliProviderMenu[] = [
    {
      id: 'lmstudio',
      label: 'LM Studio · local MLX',
      prefix: 'lmstudio:',
      connected: lmStudioModels.length > 0,
      models: lmStudioPickerModels,
      reasoning: { effort: '', levels: [], onSelect: () => {} },
    },
    {
      id: 'codex',
      label: 'OpenAI · Codex',
      prefix: 'codex:',
      connected: Boolean(codexStatus?.authenticated),
      models: codexModels,
      reasoning: {
        effort: codexReasoningEffort,
        levels: codexReasoningLevels,
        defaultEffort: activeCodexModel?.default_reasoning_level,
        onSelect: (effort) => void handleSelectCodexReasoning(effort as CodexReasoningEffort),
      },
    },
    {
      id: 'claude',
      label: 'Anthropic · Claude',
      prefix: 'claude:',
      connected: Boolean(claudeStatus?.authenticated),
      models: claudeModels,
      reasoning: {
        effort: claudeEffort,
        levels: claudeReasoningLevels,
        defaultEffort: activeClaudeModel?.default_reasoning_level,
        onSelect: (effort) => void handleSelectClaudeReasoning(effort as ClaudeReasoningEffort),
      },
    },
    {
      id: 'grok',
      label: 'xAI · Grok',
      prefix: 'grok:',
      connected: Boolean(grokStatus?.authenticated),
      models: grokModels,
      // Grok has no CLI reasoning-effort knob, so no effort menu is shown.
      reasoning: { effort: '', levels: [], onSelect: () => {} },
    },
  ]

  const selectedChatModelValue =
    chatProvider === 'lmstudio' && activeLmStudioModelSlug
      ? `lmstudio:${activeLmStudioModelSlug}`
      : chatProvider === 'codex' && activeCodexModelSlug
      ? `codex:${activeCodexModelSlug}`
      : chatProvider === 'claude' && activeClaudeModelSlug
        ? `claude:${activeClaudeModelSlug}`
        : chatProvider === 'grok' && activeGrokModelSlug
          ? `grok:${activeGrokModelSlug}`
          : selectedModel
            ? `ollama:${selectedModel}`
            : ''
  const noAvailableModels =
    models.length === 0 &&
    lmStudioModels.length === 0 &&
    !(codexStatus?.authenticated && codexModels.length > 0) &&
    !(claudeStatus?.authenticated && claudeModels.length > 0) &&
    !(grokStatus?.authenticated && grokModels.length > 0)

  return (
    <div className="flex h-full w-full overflow-hidden bg-canvas text-primary">
      {sidebarOpen && (
        <>
          {isMobile && (
            <div
              className="fixed inset-0 z-40 bg-black/50 animate-fade-in md:hidden"
              onClick={() => setSidebarOpen(false)}
              aria-hidden
            />
          )}
          <div
            className={
              isMobile
                ? 'fixed inset-y-0 left-0 z-50 animate-slide-in-left md:hidden'
                : 'hidden md:flex'
            }
          >
            <Sidebar
              conversations={visibleConversations}
              activeId={activeId}
              query={search}
              theme={theme}
              onQueryChange={setSearch}
              onNew={handleNew}
              onSelect={handleSelect}
              onRename={handleRename}
              onDelete={handleDelete}
              onPin={handlePin}
              onExport={handleExport}
              onCollapse={() => setSidebarOpen(false)}
              onToggleTheme={toggle}
              onOpenSettings={() => setSettingsOpen(true)}
            />
          </div>
        </>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <ChatHeader
          title={activeTitle}
          workdir={workdir}
          theme={theme}
          workspaceOpen={workspaceOpen}
          pendingChanges={pendingChanges}
          onChooseWorkdir={handleChooseWorkdir}
          onToggleTheme={toggle}
          onToggleWorkspace={() => setWorkspaceOpen((open) => !open)}
          onToggleSidebar={() => setSidebarOpen((open) => !open)}
          onOpenMemory={() => setMemoryOpen(true)}
          onOpenUsage={() => setUsageOpen(true)}
          onOpenModels={() => setModelsOpen(true)}
          onOpenSkills={() => setSkillsOpen(true)}
          onOpenCharacters={() => setCharactersOpen(true)}
          onOpenInstructions={() => setInstructionsOpen(true)}
          activeCharacter={activeCharacter}
        />
        {error && (
          <div className="flex items-center justify-between gap-3 border-b border-warn/25 bg-warn/10 px-4 py-2 text-callout text-warn">
            <span>{error}</span>
            <button
              type="button"
              onClick={() => void loadInitial()}
              className="flex-none rounded-md px-2 py-1 font-medium underline-offset-2 hover:underline"
            >
              Réessayer
            </button>
          </div>
        )}
        {warning && (
          <div className="flex items-center justify-between gap-3 border-b border-warn/25 bg-warn/10 px-4 py-2 text-callout text-warn">
            <span>{warning}</span>
            <button
              type="button"
              onClick={() => setWarning('')}
              className="flex-none rounded-md px-2 py-1 font-medium underline-offset-2 hover:underline"
            >
              Compris
            </button>
          </div>
        )}
        {messages.length === 0 ? (
          <EmptyState
            onPick={pickSuggestion}
            noModels={noAvailableModels}
            onManageModels={() => setModelsOpen(true)}
          />
        ) : (
          <MessageList
            messages={messages}
            busy={busy}
            characterAvatar={activeCharacter?.thumbnail || activeCharacter?.emotions.neutral?.image || ''}
            onRegenerate={handleRegenerate}
            onEdit={handleEditResend}
          />
        )}
        <CommandApprovalBanner pending={pendingApproval} onDecision={approve} />
        <div className="px-4 pt-2">
          <div className="mx-auto max-w-[45rem]">
            <ModeBar
              agentMode={agentMode}
              webSearch={webSearch}
              yolo={yolo}
              imageActive={imageActive}
              videoActive={videoActive}
              voiceMode={voiceMode}
              allowCommands={allowCommands}
              commandApproval={commandApproval}
              onToggleAgent={toggleAgent}
              onToggleWeb={toggleWebSearch}
              onToggleYolo={toggleYolo}
              onToggleImage={toggleImage}
              onToggleVideo={toggleVideo}
              onToggleVoice={toggleVoice}
              onToggleCommands={toggleCommands}
              onToggleApproval={toggleCommandApproval}
            />
          </div>
        </div>
        {genStats && (genStats.tokens_per_sec > 0 || genStats.context_used > 0) && (
          <div className="px-4">
            <div className="mx-auto max-w-[45rem] pt-1.5 text-right text-caption tabular-nums text-tertiary">
              {genStats.tokens_per_sec > 0 && <span>{genStats.tokens_per_sec} tok/s</span>}
              {genStats.context_used > 0 && (
                <span>
                  {genStats.tokens_per_sec > 0 ? ' · ' : ''}
                  contexte {genStats.context_used.toLocaleString('fr')}
                  {genStats.context_length > 0
                    ? ` / ${genStats.context_length.toLocaleString('fr')} (${genStats.usage_pct}%)`
                    : ' tokens'}
                </span>
              )}
            </div>
          </div>
        )}
        <Composer
          busy={busy}
          onSend={handleSend}
          onStop={handleStop}
          prefill={prefill}
          prefillNonce={prefillNonce}
          voiceMode={voiceMode}
          relistenNonce={relisten}
          modelPicker={
            <ModelPicker
              up
              align="right"
              localModels={models}
              cliProviders={cliProviders}
              selectedModelValue={selectedChatModelValue}
              onSelectModel={handleSelectChatModel}
              onOpenModels={() => setModelsOpen(true)}
            />
          }
        />
      </div>
      {workspaceOpen &&
        (() => {
          const panel = (
            <WorkspacePanel
              changes={changes}
              onApply={applyChange}
              onReject={rejectChange}
              files={files}
              workdir={workdir}
              loaded={loadedFiles}
              onLoadFile={handleLoadFile}
              onCloseFile={handleCloseFile}
              onChooseWorkdir={handleChooseWorkdir}
              onRefreshFiles={handleRefreshFiles}
              onClose={() => setWorkspaceOpen(false)}
              ragEnabled={ragEnabled}
              indexedChunks={indexedChunks}
              indexing={indexing}
              onIndexProject={handleIndexProject}
              onToggleRag={toggleRag}
              changeCount={changeCount}
              onUndo={handleUndo}
            />
          )
          // Push panel on ≥lg; slide-over drawer (with backdrop) on narrower.
          return isCompact ? (
            <>
              <div
                className="fixed inset-0 z-40 bg-black/40 animate-fade-in"
                onClick={() => setWorkspaceOpen(false)}
                aria-hidden
              />
              <div className="fixed inset-y-0 right-0 z-50 w-full max-w-[30rem] animate-slide-in-right">
                {panel}
              </div>
            </>
          ) : (
            <div className="flex w-[360px] flex-none xl:w-[400px]">{panel}</div>
          )
        })()}
      {settingsOpen && (
        <SettingsModal
          onClose={() => setSettingsOpen(false)}
          onSaved={(settings) => {
            setSavedPrompts(settings.saved_prompts || [])
            setLmStudioModel(settings.lmstudio_model || '')
            void refreshLmStudioCatalog()
          }}
          onInstallWeb={() => {
            setSettingsOpen(false)
            void openWebSetup()
          }}
        />
      )}
      {instructionsOpen && <ConversationInstructionsModal onClose={() => setInstructionsOpen(false)} />}
      {memoryOpen && <MemoryModal onClose={() => setMemoryOpen(false)} />}
      {usageOpen && <UsageModal onClose={() => setUsageOpen(false)} />}
      {modelsOpen && (
        <ModelsModal onClose={() => setModelsOpen(false)} onChanged={() => void refreshModels()} />
      )}
      {skillsOpen && <SkillsModal onClose={() => setSkillsOpen(false)} />}
      {charactersOpen && (
        <CharactersModal
          activeCharacter={activeCharacter}
          onActiveChange={setActiveCharacter}
          onClose={() => setCharactersOpen(false)}
        />
      )}
      {webSetup && (
        <WebSetupModal status={webSetup} onResolve={(outcome) => void finishWebSetup(outcome)} />
      )}
      {paletteOpen && (
        <CommandPalette
          onClose={() => setPaletteOpen(false)}
          conversations={conversations}
          prompts={savedPrompts}
          actions={[
            { id: 'new', label: 'Nouvelle conversation', run: () => void handleNew() },
            { id: 'settings', label: 'Réglages', run: () => setSettingsOpen(true) },
            { id: 'memory', label: 'Mémoire', run: () => setMemoryOpen(true) },
            { id: 'usage', label: 'Usage Claude & Codex', run: () => setUsageOpen(true) },
            { id: 'models', label: 'Gérer les modèles', run: () => setModelsOpen(true) },
            { id: 'skills', label: 'Compétences', run: () => setSkillsOpen(true) },
            { id: 'characters', label: 'Personnages', run: () => setCharactersOpen(true) },
            { id: 'theme', label: 'Basculer le thème', run: toggle },
            { id: 'workspace', label: "Atelier de code", run: () => setWorkspaceOpen((o) => !o) },
          ]}
          onSelectConversation={(id) => void handleSelect(id)}
          onUsePrompt={pickSuggestion}
        />
      )}
    </div>
  )
}
