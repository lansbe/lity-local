import { useCallback } from 'react'
import type { Dispatch, SetStateAction } from 'react'

import { bridge } from '../bridge'
import type {
  ChatMessage,
  ChatProvider,
  CharacterProfile,
  ClaudeModelsResult,
  ClaudeReasoningEffort,
  ClaudeStatus,
  CodexModel,
  CodexModelsResult,
  CodexReasoningEffort,
  CodexStatus,
  ConversationMeta,
  GrokModelsResult,
  GrokStatus,
  LmStudioModelsResult,
  LoadedFile,
  SavedPrompt,
} from '../types'

type SetState<T> = Dispatch<SetStateAction<T>>

interface CodexCatalogState {
  status: CodexStatus
  catalog: CodexModelsResult
}

interface ClaudeCatalogState {
  status: ClaudeStatus
  catalog: ClaudeModelsResult
}

interface GrokCatalogState {
  status: GrokStatus
  catalog: GrokModelsResult
}

interface LmStudioCatalogState {
  catalog: LmStudioModelsResult
}

interface InitialBackendLoadOptions {
  refreshCodexCatalog: () => Promise<CodexCatalogState | null>
  refreshClaudeCatalog: () => Promise<ClaudeCatalogState | null>
  refreshGrokCatalog: () => Promise<GrokCatalogState | null>
  refreshLmStudioCatalog: () => Promise<LmStudioCatalogState | null>
  setError: SetState<string>
  setAssistantName: SetState<string>
  setConversations: SetState<ConversationMeta[]>
  setActiveId: SetState<string>
  setWorkdir: SetState<string>
  setModels: SetState<string[]>
  setSelectedModel: SetState<string>
  setChatProvider: SetState<ChatProvider>
  setCodexModel: SetState<string>
  setCodexReasoningEffort: SetState<CodexReasoningEffort>
  setClaudeModel: SetState<string>
  setClaudeEffort: SetState<ClaudeReasoningEffort>
  setGrokModel: SetState<string>
  setLmStudioModel: SetState<string>
  setAgentMode: SetState<boolean>
  setAllowCommands: SetState<boolean>
  setYolo: SetState<boolean>
  setWebSearch: SetState<boolean>
  setRagEnabled: SetState<boolean>
  setIndexedChunks: SetState<number>
  setChangeCount: SetState<number>
  setImageActive: SetState<boolean>
  setVideoActive: SetState<boolean>
  setCommandApproval: SetState<boolean>
  setSavedPrompts: SetState<SavedPrompt[]>
  setActiveCharacter: SetState<CharacterProfile | null>
  setMessages: SetState<ChatMessage[]>
  setFiles: SetState<string[]>
  setLoadedFiles: SetState<LoadedFile[]>
}

export function useInitialBackendLoad(options: InitialBackendLoadOptions) {
  const {
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
  } = options
  return useCallback(async () => {
    try {
      setError('')
      await bridge.ready()
      const [state, modelList, settings, lmstudio, codex, claude, grok] = await Promise.all([
        bridge.getState(),
        bridge.listModels(),
        bridge.getSettings(),
        refreshLmStudioCatalog(),
        refreshCodexCatalog(),
        refreshClaudeCatalog(),
        refreshGrokCatalog(),
      ])
      setAssistantName(state.assistant_name || 'Assistant')
      setConversations(state.conversations || [])
      setActiveId(state.active_conversation_id || '')
      setWorkdir(state.workdir || '')
      setModels(modelList.models || [])
      setSelectedModel(modelList.models?.length ? modelList.selected || state.model || '' : '')
      const lmStudioReady = Boolean(lmstudio?.catalog.models?.length)
      const codexReady = Boolean(codex?.status.authenticated && codex.catalog.models?.length)
      const claudeReady = Boolean(claude?.status.authenticated && claude.catalog.models?.length)
      const grokReady = Boolean(grok?.status.authenticated && grok.catalog.models?.length)
      const configuredProvider: ChatProvider =
        settings.chat_provider === 'lmstudio' ||
        settings.chat_provider === 'codex' ||
        settings.chat_provider === 'claude' ||
        settings.chat_provider === 'grok'
          ? settings.chat_provider
          : 'ollama'
      // Honour the configured provider; otherwise, with no local model, fall back
      // to whichever CLI provider is ready (Codex, then Claude, then Grok).
      let resolvedProvider: ChatProvider = configuredProvider
      if (configuredProvider === 'ollama' && !modelList.models?.length) {
        resolvedProvider = lmStudioReady
          ? 'lmstudio'
          : codexReady
            ? 'codex'
            : claudeReady
              ? 'claude'
              : grokReady
                ? 'grok'
                : 'ollama'
      }
      setChatProvider(resolvedProvider)
      setLmStudioModel(settings.lmstudio_model || lmstudio?.catalog.default_model || '')
      setCodexModel(settings.codex_model || '')
      setCodexReasoningEffort(settings.codex_reasoning_effort || 'medium')
      setClaudeModel(settings.claude_model || '')
      setClaudeEffort(settings.claude_effort || 'medium')
      setGrokModel(settings.grok_model || '')
      setAgentMode(Boolean(state.agent_mode))
      setAllowCommands(Boolean(state.allow_commands))
      setYolo(Boolean(state.yolo))
      setWebSearch(Boolean(state.web_search))
      setRagEnabled(Boolean(state.rag_enabled))
      setIndexedChunks(state.indexed_chunks ?? 0)
      setChangeCount(state.change_count ?? 0)
      setImageActive(Boolean(state.image_active))
      setVideoActive(Boolean(state.video_active))
      setCommandApproval(Boolean(state.command_approval))
      setSavedPrompts(settings.saved_prompts || [])
      setActiveCharacter(state.active_character ?? null)
      if (modelList.error) {
        setError(`Ollama : ${modelList.error}`)
      } else if (
        (modelList.models || []).length === 0 &&
        !lmStudioReady &&
        !codexReady &&
        !claudeReady &&
        !grokReady
      ) {
        setError(
          'Aucun modèle disponible. Démarre Ollama ou LM Studio, ou connecte Codex / Claude / Grok dans Réglages, puis clique sur Réessayer.',
        )
      }
      const initialMessages = await bridge.getMessages()
      setMessages(initialMessages || [])

      const workspace = await bridge.listWorkspaceFiles()
      setFiles(workspace.files || [])
      setWorkdir(workspace.workdir || state.workdir || '')
      setLoadedFiles((await bridge.getLoadedFiles()) || [])
    } catch (loadError) {
      setError(`Connexion au backend impossible : ${String(loadError)}`)
    }
  }, [
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
  ])
}

export type { CodexCatalogState, ClaudeCatalogState, GrokCatalogState, LmStudioCatalogState }
