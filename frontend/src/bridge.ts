import type {
  ApplyResult,
  AppSettings,
  AppState,
  AudioStatus,
  CharacterProfile,
  CharactersResult,
  ChatMessage,
  ClaudeLoginResult,
  ClaudeModelsResult,
  ClaudeStatus,
  CliUsage,
  GrokLoginResult,
  GrokModelsResult,
  GrokStatus,
  LmStudioModelsResult,
  ConversationMeta,
  CodexLoginResult,
  CodexModelsResult,
  CodexStatus,
  CreateBlock,
  EditBlock,
  GitStatus,
  HealthItem,
  LoadedFile,
  MemoryData,
  ModelDetail,
  ModelRecommendations,
  ModelsResult,
  ModelSuggestion,
  PullStatus,
  SendResult,
  SkillMutation,
  SkillsResult,
  VoicesState,
  WebStatus,
  WorkspaceFiles,
} from './types'

type Listener = (payload: any) => void

const listeners: Record<string, Set<Listener>> = {}

export function on(event: string, fn: Listener): () => void {
  ;(listeners[event] ||= new Set()).add(fn)
  return () => {
    listeners[event]?.delete(fn)
  }
}

function dispatch(event: string, payload: any): void {
  listeners[event]?.forEach((fn) => fn(payload))
}

declare global {
  interface Window {
    pywebview?: { api: Record<string, (...args: any[]) => Promise<any>> }
    __lityBus?: (msg: { event: string; payload: any }) => void
  }
}

// Python pushes streaming events here via window.evaluate_js.
window.__lityBus = (msg) => {
  if (msg && msg.event) dispatch(msg.event, msg.payload)
}

/**
 * Resolves once the real pywebview bridge is available. There is no mock: the
 * UI always talks to the real Python backend (run via `lity --ui web`,
 * or `--ui web --dev` against the Vite dev server). pywebview injects
 * `window.pywebview.api` shortly after load, so we wait for the
 * `pywebviewready` event and poll as a backstop in case it was missed.
 */
export function ready(): Promise<void> {
  return new Promise((resolve) => {
    // Consider the bridge ready only when a concrete method is callable:
    // pywebview can set `window.pywebview.api` slightly before its methods are
    // attached, and resolving too early makes the first calls fail.
    const haveApi = () => typeof window.pywebview?.api?.get_state === 'function'
    let settled = false
    let poll: ReturnType<typeof setInterval> | undefined

    const finish = () => {
      if (settled) return
      settled = true
      if (poll) clearInterval(poll)
      resolve()
    }

    if (haveApi()) {
      finish()
      return
    }

    const tryFinish = () => {
      if (haveApi()) finish()
    }
    window.addEventListener('pywebviewready', tryFinish, { once: true })
    poll = setInterval(tryFinish, 50)
  })
}

async function call<T = any>(method: string, ...args: any[]): Promise<T> {
  const api = window.pywebview?.api
  if (api && typeof api[method] === 'function') {
    return api[method](...args)
  }
  throw new Error(`Méthode de pont indisponible : ${method}`)
}

export const bridge = {
  ready,
  on,
  getState: () => call<AppState>('get_state'),
  listModels: () => call<ModelsResult>('list_models'),
  setModel: (model: string) => call<{ selected: string }>('set_model', model),
  setWorkdir: (path: string) =>
    call<{ success: boolean; message: string; workdir: string }>('set_workdir', path),
  chooseWorkdir: () =>
    call<{ success: boolean; message: string; workdir: string }>('choose_workdir'),
  listConversations: () => call<ConversationMeta[]>('list_conversations'),
  newConversation: () => call<ConversationMeta>('new_conversation'),
  switchConversation: (id: string) =>
    call<{
      success: boolean
      active_conversation_id: string
      messages: ChatMessage[]
      workdir: string
      files: string[]
      model: string
      active_character?: CharacterProfile | null
    }>('switch_conversation', id),
  renameConversation: (id: string, title: string) =>
    call<{ success: boolean; conversations: ConversationMeta[] }>('rename_conversation', id, title),
  deleteConversation: (id: string) =>
    call<{
      active_conversation_id: string
      conversations: ConversationMeta[]
      messages: ChatMessage[]
      active_character?: CharacterProfile | null
    }>('delete_conversation', id),
  getMessages: (id?: string) => call<ChatMessage[]>('get_messages', id ?? null),
  sendMessage: (text: string, id?: string, images?: string[]) =>
    call<SendResult>('send_message', text, id ?? null, images ?? null),
  regenerate: () => call<SendResult>('regenerate'),
  editAndResend: (text: string) => call<SendResult>('edit_and_resend', text),
  stop: () => call<{ stopped: boolean }>('stop'),

  // Workspace + editing
  listWorkspaceFiles: () => call<WorkspaceFiles>('list_workspace_files'),
  getLoadedFiles: () => call<LoadedFile[]>('get_loaded_files'),
  loadContextFile: (path: string) =>
    call<{ success: boolean; message: string; loaded: LoadedFile[] }>('load_context_file', path),
  closeContextFile: (path: string) =>
    call<{ success: boolean; message: string; loaded: LoadedFile[] }>('close_context_file', path),
  applyCreate: (block: CreateBlock) => call<ApplyResult>('apply_create', block),
  applyEdit: (block: EditBlock) => call<ApplyResult>('apply_edit', block),
  undoChange: () =>
    call<{ ok: boolean; message: string; change_count: number; files: string[] }>('undo_change'),

  // Agent mode
  setAgentMode: (enabled: boolean) => call<{ agent_mode: boolean }>('set_agent_mode', enabled),
  setAllowCommands: (enabled: boolean) =>
    call<{ allow_commands: boolean }>('set_allow_commands', enabled),
  setYolo: (enabled: boolean) =>
    call<{ yolo: boolean; agent_mode: boolean; write_mode: 'reviewed' | 'autonomous' }>(
      'set_yolo',
      enabled,
    ),
  setWebSearch: (enabled: boolean) =>
    call<{ web_search: boolean; agent_mode: boolean }>('set_web_search', enabled),
  webStatus: () => call<WebStatus>('web_status'),
  lmstudioModels: () => call<LmStudioModelsResult>('lmstudio_models'),
  codexStatus: () => call<CodexStatus>('codex_status'),
  codexLogin: () => call<CodexLoginResult>('codex_login'),
  codexModels: () => call<CodexModelsResult>('codex_models'),
  claudeStatus: () => call<ClaudeStatus>('claude_status'),
  claudeLogin: () => call<ClaudeLoginResult>('claude_login'),
  claudeModels: () => call<ClaudeModelsResult>('claude_models'),
  grokStatus: () => call<GrokStatus>('grok_status'),
  grokLogin: (deviceAuth = false) => call<GrokLoginResult>('grok_login', deviceAuth),
  grokModels: () => call<GrokModelsResult>('grok_models'),
  usage: () => call<CliUsage>('usage'),
  setupSearxng: () => call<{ ok: boolean; running: boolean }>('setup_searxng'),
  markWebSetupResolved: () => call<{ ok: boolean }>('mark_web_setup_resolved'),

  // RAG + search
  indexProject: () => call<{ ok: boolean; chunks: number; message: string }>('index_project'),
  setRag: (enabled: boolean) => call<{ rag_enabled: boolean }>('set_rag', enabled),
  searchConversations: (query: string) =>
    call<ConversationMeta[]>('search_conversations', query),

  // Settings
  getSettings: () => call<AppSettings>('get_settings'),
  updateSettings: (patch: Partial<AppSettings>) => call<AppSettings>('update_settings', patch),

  // Skills (Compétences)
  listSkills: () => call<SkillsResult>('list_skills'),
  toggleSkill: (name: string, enabled: boolean) =>
    call<SkillMutation>('toggle_skill', name, enabled),
  createSkill: (name: string, description: string, body: string, whenToUse?: string, triggers?: string[]) =>
    call<SkillMutation>('create_skill', name, description, body, whenToUse ?? '', triggers ?? null),
  deleteSkill: (name: string) => call<SkillMutation>('delete_skill', name),

  // Characters
  listCharacters: () => call<CharactersResult>('list_characters'),
  createCharacter: (data: Partial<CharacterProfile>) =>
    call<CharactersResult>('create_character', data),
  updateCharacter: (id: string, patch: Partial<CharacterProfile>) =>
    call<CharactersResult>('update_character', id, patch),
  deleteCharacter: (id: string) => call<CharactersResult>('delete_character', id),
  setConversationCharacter: (id: string) =>
    call<CharactersResult>('set_conversation_character', id),
  generateCharacterEmotions: (id: string, emotions?: string[]) =>
    call<CharactersResult>('generate_character_emotions', id, emotions ?? null),

  // Health
  getHealth: () => call<HealthItem[]>('get_health'),

  // Long-term memory
  getMemory: () => call<MemoryData>('get_memory'),
  updateMemoryEntry: (category: string, key: string, value: string) =>
    call<MemoryData>('update_memory_entry', category, key, value),
  deleteMemoryEntry: (category: string, key: string) =>
    call<MemoryData>('delete_memory_entry', category, key),
  clearMemory: () => call<MemoryData>('clear_memory'),

  // Voice
  audioStatus: () => call<AudioStatus>('audio_status'),
  startRecording: () => call<{ ok: boolean; message: string }>('start_recording'),
  stopRecording: () => call<{ ok: boolean; text: string }>('stop_recording'),
  speak: (text: string) =>
    call<{ ok: boolean; message?: string; needs_voice?: boolean }>('speak', text),
  stopSpeaking: () => call<{ ok: boolean }>('stop_speaking'),
  downloadVoice: (voiceId?: string) =>
    call<{ ok: boolean; message: string }>('download_voice', voiceId ?? ''),
  listVoices: () => call<VoicesState>('list_voices'),
  setVoice: (name: string) => call<{ ok: boolean; current?: string }>('set_voice', name),

  // Image generation
  imageActive: () => call<boolean>('image_active'),
  startImageSession: () => call<any>('start_image_session'),
  pollImageLaunch: () => call<any>('poll_image_launch'),
  stopImageSession: () => call<{ active: boolean }>('stop_image_session'),
  downloadImageModel: (name: string) =>
    call<{ ok: boolean; running: boolean; message: string }>('download_image_model', name),
  cancelImageDownload: () => call<{ ok: boolean }>('cancel_image_download'),
  imagePullStatus: () => call<{ active: string | null }>('image_pull_status'),
  selectImageModel: (name: string) =>
    call<{ ok: boolean; selected: string }>('select_image_model', name),

  // Video generation
  videoActive: () => call<boolean>('video_active'),
  startVideoSession: () => call<any>('start_video_session'),
  pollVideoLaunch: () => call<any>('poll_video_launch'),
  stopVideoSession: () => call<{ active: boolean }>('stop_video_session'),
  downloadVideoModel: (name: string) =>
    call<{ ok: boolean; running: boolean; message: string }>('download_video_model', name),
  cancelVideoDownload: () => call<{ ok: boolean }>('cancel_video_download'),
  videoPullStatus: () => call<{ active: string | null }>('video_pull_status'),
  selectVideoModel: (name: string) =>
    call<{ ok: boolean; selected: string }>('select_video_model', name),

  // Model management
  listModelsDetailed: () => call<ModelDetail[]>('list_models_detailed'),
  pullModel: (name: string) => call<PullStatus>('pull_model', name),
  pullStatus: () => call<PullStatus>('pull_status'),
  cancelPull: (name?: string) => call<PullStatus>('cancel_pull', name ?? ''),
  deleteModel: (name: string) =>
    call<{ ok: boolean; message: string; models: ModelDetail[] }>('delete_model', name),
  modelInfo: (name: string) => call<Record<string, unknown>>('model_info', name),
  modelSuggestions: () => call<ModelSuggestion[]>('model_suggestions'),
  modelRecommendations: () => call<ModelRecommendations>('model_recommendations'),
  modelSupportsTools: (name?: string) =>
    call<{ model: string; supports: boolean | null }>('model_supports_tools', name ?? ''),
  extractDocument: (name: string, data: string) =>
    call<{ ok: boolean; name: string; text: string; error: string | null }>(
      'extract_document',
      name,
      data,
    ),
  fetchPage: (url: string) =>
    call<{ ok: boolean; url: string; title?: string; text: string; error: string | null }>(
      'fetch_page',
      url,
    ),
  openExternal: (url: string) => call<{ ok: boolean }>('open_external', url),
  generationStats: () =>
    call<{
      tokens_per_sec: number
      context_used: number
      context_length: number
      usage_pct: number
    }>('generation_stats'),
  getConversationInstructions: () =>
    call<{ instructions: string; temperature: number | null }>('get_conversation_instructions'),
  setConversationInstructions: (instructions: string, temperature: number | null) =>
    call<{ ok: boolean; instructions: string; temperature: number | null }>(
      'set_conversation_instructions',
      instructions,
      temperature,
    ),

  // Per-command approval
  setCommandApproval: (ask: boolean) =>
    call<{ command_approval: boolean }>('set_command_approval', ask),
  approveCommand: (id: number, allow: boolean) =>
    call<{ ok: boolean }>('approve_command', id, allow),

  // Git
  gitStatus: () => call<GitStatus>('git_status'),
  gitDiff: (path?: string) => call<{ diff: string }>('git_diff', path ?? null),
  gitBranches: () => call<{ branches: string[]; current: string }>('git_branches'),
  gitCommit: (message: string) =>
    call<{ ok: boolean; message: string; status?: GitStatus }>('git_commit', message),

  // Export / pin
  exportConversation: (id: string | null, fmt: string) =>
    call<{ ok: boolean; content: string; filename: string }>('export_conversation', id, fmt),
  setPinned: (id: string, pinned: boolean) =>
    call<{ conversations: ConversationMeta[] }>('set_pinned', id, pinned),
}
