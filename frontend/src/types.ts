export interface ConversationMeta {
  id: string
  title: string
  created_at: string
  updated_at: string
  workdir?: string
  model?: string
  pinned?: boolean
  character_id?: string
  active_character?: CharacterProfile | null
}

export interface CharacterEmotion {
  label: string
  image_path: string
  image?: string
}

export interface CharacterProfile {
  id: string
  name: string
  description: string
  gender: string
  style: string
  instructions: string
  voice: string
  image_model: string
  seed: number
  created_at: string
  updated_at: string
  emotions: Record<string, CharacterEmotion>
  thumbnail?: string
}

export interface CharactersResult {
  ok?: boolean
  message?: string
  characters: CharacterProfile[]
  active_character_id?: string
  active_character?: CharacterProfile | null
  character?: CharacterProfile | null
  generated?: string[]
  errors?: { emotion: string; message: string }[]
}

export type Role = 'user' | 'assistant' | 'system'
export type ChatProvider = 'ollama' | 'lmstudio' | 'codex' | 'claude' | 'grok'
export type CodexReasoningEffort = 'minimal' | 'low' | 'medium' | 'high' | 'xhigh'
export type ClaudeReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export interface StepEvent {
  kind: 'tool_call' | 'tool_result' | 'receipts'
  /** Tool name (absent on a 'receipts' summary event). */
  name?: string
  args?: Record<string, unknown>
  ok?: boolean
  summary?: string
  /** receipts: true when the answer is backed by ≥1 successful tool call. */
  grounded?: boolean
  /** receipts: distinct tools that ran during the turn. */
  tools_used?: string[]
}

export interface ChatMessage {
  role: Role
  content: string
  timestamp?: string
  /** True while the assistant message is still streaming. */
  pending?: boolean
  /** Agent-mode tool steps that produced this message. */
  steps?: StepEvent[]
  /** Base64 data URL of a generated image (image mode). */
  image?: string
  /** Base64 data URL of a generated video clip (video mode). */
  video?: string
  /** Data URLs of images the user attached to this message. */
  images?: string[]
  /** Wall-clock time the turn took to answer, ms (shown on hover). */
  elapsedMs?: number
}

export interface CreateBlock {
  file_path: string
  content: string
}

export interface EditBlock {
  file_path: string
  search_content: string
  replace_content: string
}

export interface LoadedFile {
  path: string
  name: string
  rel: string
}

export interface WorkspaceFiles {
  workdir: string
  files: string[]
}

export interface ApplyResult {
  success: boolean
  message: string
  files?: string[]
  change_count?: number
}

export type ChangeStatus = 'pending' | 'applied' | 'rejected' | 'error'

export type Change =
  | { id: string; kind: 'create'; block: CreateBlock; status: ChangeStatus; message?: string }
  | { id: string; kind: 'edit'; block: EditBlock; status: ChangeStatus; message?: string }

export interface SendResult {
  type: string
  content?: string
  message?: string
  action?: string
  image?: string
  video?: string
  /** Out-of-band notice for the turn (e.g. image can't be read by this model). */
  system_notification?: string
  cancelled?: boolean
  create_blocks?: CreateBlock[]
  edit_blocks?: EditBlock[]
  conversations?: ConversationMeta[]
  active_conversation_id?: string
  active_character?: CharacterProfile | null
  change_count?: number
}

export interface AppState {
  assistant_name: string
  model: string
  workdir: string
  active_conversation_id: string
  conversations: ConversationMeta[]
  agent_mode?: boolean
  allow_commands?: boolean
  command_approval?: boolean
  yolo?: boolean
  write_mode?: 'reviewed' | 'autonomous'
  web_search?: boolean
  rag_enabled?: boolean
  indexed_chunks?: number
  change_count?: number
  image_active?: boolean
  video_active?: boolean
  chat_provider?: ChatProvider
  active_character?: CharacterProfile | null
}

export interface ModelsResult {
  models: string[]
  selected: string
  error: string | null
}

export interface AppSettings {
  custom_instructions: string
  embedding_model: string
  selected_model: string
  chat_provider: ChatProvider
  lmstudio_base_url: string
  lmstudio_model: string
  codex_model: string
  codex_reasoning_effort: CodexReasoningEffort
  claude_model: string
  claude_effort: ClaudeReasoningEffort
  grok_model: string
  default_agent: boolean
  default_yolo: boolean
  saved_prompts: SavedPrompt[]
  web_search_enabled: boolean
  searxng_url: string
  cross_session_memory: boolean
  web_setup_resolved?: boolean
  skills_enabled?: boolean
  skills_semantic?: boolean
}

export interface MemoryData {
  user_profile: Record<string, string>
  assistant_profile: Record<string, string>
  facts: Record<string, string>
}

export interface HealthItem {
  name: string
  ok: boolean
  detail: string
}

export interface AudioStatus {
  stt_available: boolean
  stt_ready: boolean
  stt_error: string | null
  tts_available: boolean
  has_voice: boolean
}

export interface ModelDetail {
  name: string
  size: number
}

export interface ModelSuggestion {
  name: string
  category: string
  note: string
}

export interface HardwareInfo {
  os: string
  arch: string
  cpu_cores: number
  ram_gb: number
  gpu: string | null
  vram_gb: number | null
  accelerator: string
  budget_gb: number
}

export interface QuantEvaluation {
  name: string
  bits: number
  vram_gb: number
  disk_gb: number
  quality: string
  status: string
  grade: string
  score: number
  tokens_per_sec: number | null
  mem_pct: number | null
}

export interface ModelRecommendation {
  name: string
  display_name?: string
  provider?: string
  params_b: number
  size_gb: number
  kind: string
  installed: boolean
  verdict: 'excellent' | 'bon' | 'limite' | 'trop_lourd'
  speed: string
  recommended?: boolean
  // canirun.ai-aligned compatibility report
  grade?: 'S' | 'A' | 'B' | 'C' | 'D' | 'F' | '?'
  grade_label?: string
  score?: number
  status?: string
  tokens_per_sec?: number | null
  mem_pct?: number | null
  context_length?: number
  thinking?: boolean
  license?: string
  architecture?: string
  active_params_b?: number | null
  quants?: QuantEvaluation[]
  best_quant?: { name: string; vram_gb: number; grade: string } | null
}

export interface ImageModelRecommendation {
  name: string
  display_name: string
  provider: string
  params_b: number
  vram_gb: number
  disk_gb: number
  kind: 'image'
  backend: 'automatic1111' | 'comfyui' | 'stable-diffusion.cpp' | string
  installed: boolean
  /** True for the downloaded model the in-process engine generates with. */
  selected?: boolean
  verdict: 'excellent' | 'bon' | 'limite' | 'trop_lourd'
  speed: string
  recommended?: boolean
  grade?: 'S' | 'A' | 'B' | 'C' | 'D' | 'F' | '?'
  grade_label?: string
  score?: number
  status?: string
  tokens_per_sec?: number | null
  mem_pct?: number | null
  license: string
  model_url: string
  install_hint: string
}

export interface VideoModelRecommendation {
  name: string
  display_name: string
  provider: string
  params_b: number
  vram_gb: number
  disk_gb: number
  kind: 'video'
  backend: 'diffusers' | 'mlx' | 'comfyui' | string
  /** "text" (T2V) or "image" (I2V) input. */
  input_type: 'text' | 'image' | string
  installed: boolean
  /** True for the downloaded model the in-process engine generates with. */
  selected?: boolean
  verdict: 'excellent' | 'bon' | 'limite' | 'trop_lourd'
  speed: string
  recommended?: boolean
  grade?: 'S' | 'A' | 'B' | 'C' | 'D' | 'F' | '?'
  grade_label?: string
  score?: number
  status?: string
  tokens_per_sec?: number | null
  mem_pct?: number | null
  license: string
  model_url: string
  install_hint: string
}

export interface ModelRecommendations {
  hardware: HardwareInfo
  models: ModelRecommendation[]
  image_models?: ImageModelRecommendation[]
  video_models?: VideoModelRecommendation[]
}

export interface WebStatus {
  url: string
  reachable: boolean
  docker: boolean
  container: string
  fallback_ddg: boolean
  /** The user already made a deliberate web-setup choice (installed / fallback). */
  setup_resolved: boolean
  setup_running: boolean
}

export interface CodexStatus {
  available: boolean
  authenticated: boolean
  message: string
}

export interface CodexReasoningLevel {
  effort: CodexReasoningEffort
  description: string
}

export interface CodexModel {
  slug: string
  display_name: string
  description: string
  default_reasoning_level: CodexReasoningEffort
  supported_reasoning_levels: CodexReasoningLevel[]
  priority?: number
}

export interface CodexModelsResult {
  ok: boolean
  models: CodexModel[]
  default_model: string
  message: string
}

export interface CodexLoginResult {
  ok: boolean
  running: boolean
  message: string
  status: CodexStatus
}

export interface ClaudeStatus {
  available: boolean
  authenticated: boolean
  message: string
}

export interface ClaudeReasoningLevel {
  effort: ClaudeReasoningEffort
  description: string
}

export interface ClaudeModel {
  slug: string
  display_name: string
  description: string
  default_reasoning_level: ClaudeReasoningEffort
  supported_reasoning_levels: ClaudeReasoningLevel[]
  priority?: number
}

export interface ClaudeModelsResult {
  ok: boolean
  models: ClaudeModel[]
  default_model: string
  message: string
}

export interface ClaudeLoginResult {
  ok: boolean
  running: boolean
  message: string
  status: ClaudeStatus
}

export interface GrokStatus {
  available: boolean
  authenticated: boolean
  message: string
}

export interface GrokModel {
  slug: string
  display_name: string
  description: string
  /** Grok has no CLI reasoning-effort knob — always empty. */
  default_reasoning_level: string
  supported_reasoning_levels: { effort: string; description: string }[]
  priority?: number
}

export interface GrokModelsResult {
  ok: boolean
  models: GrokModel[]
  default_model: string
  message: string
}

export interface LmStudioModel {
  slug: string
  display_name?: string
  note?: string
}

export interface LmStudioModelsResult {
  ok: boolean
  models: LmStudioModel[]
  default_model: string
  base_url: string
  recommended: LmStudioModel[]
  message: string
}

export interface GrokLoginResult {
  ok: boolean
  running: boolean
  message: string
  status: GrokStatus
}

export interface UsageModelRow {
  model: string
  turns: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  cost_usd: number
}

export interface ProviderUsage {
  turns: number
  cost_usd: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  by_model: UsageModelRow[]
}

export interface CliUsage {
  claude: ProviderUsage
  codex: ProviderUsage
  grok: ProviderUsage
}

/** Outcome of the web-setup modal — drives whether the choice is remembered. */
export type WebSetupOutcome = 'dismissed' | 'fallback' | 'installed'

export interface PullProgress {
  status?: string
  completed?: number
  total?: number
}

export interface PullStatus {
  active: string | null
  queue: string[]
  progress: PullProgress
}

export interface VoiceCatalogItem {
  id: string
  label: string
  lang: string
}

export interface VoicesState {
  available: boolean
  installed: string[]
  current: string
  catalog: VoiceCatalogItem[]
}

export interface GitFile {
  status: string
  path: string
}

export interface GitStatus {
  is_repo: boolean
  branch: string
  files: GitFile[]
}

export interface SavedPrompt {
  title: string
  text: string
}

export interface Skill {
  name: string
  description: string
  when_to_use: string
  triggers: string[]
  allowed_tools: string[]
  source: 'builtin' | 'user'
  builtin: boolean
  path: string
  enabled: boolean
}

export interface SkillsResult {
  /** Master toggle — when false, no skill is injected at all. */
  enabled: boolean
  /** Semantic (embedding) matching is on, on top of the lexical matcher. */
  semantic: boolean
  /** User skills directory (where new skills are written). */
  dir: string
  skills: Skill[]
}

export interface SkillMutation {
  ok: boolean
  message?: string
  name?: string
  enabled?: boolean
  skill?: Skill
}
