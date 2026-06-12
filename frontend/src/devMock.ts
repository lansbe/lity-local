/**
 * Dev-only mock of the pywebview backend bridge.
 *
 * The real app talks to Python via `window.pywebview.api`, injected by pywebview
 * at runtime (`lity --ui web`). In a plain browser (Vite dev / design
 * preview) that bridge is absent, so `bridge.ready()` would hang forever and the
 * UI would never render.
 *
 * This module installs a stub `window.pywebview.api` returning representative
 * sample data so the interface can be developed and screenshotted in a browser.
 *
 * It is STRICTLY dev-only and self-disabling:
 *   - only runs under Vite dev (`import.meta.env.DEV === true`);
 *   - only runs when explicitly enabled (`?lity_mock=1` or `VITE_LITY_MOCK=1`);
 *   - never overrides a real bridge (bails if `window.pywebview` already exists);
 *   - tree-shaken out of production builds.
 */

import type {
  AppSettings,
  AppState,
  CharacterProfile,
  ChatMessage,
  ConversationMeta,
  HealthItem,
  ModelRecommendations,
  WebStatus,
} from './types'

const now = '2026-06-10T14:00:00Z'
const demoWorkdir = '/workspace/lity'

function mockPortrait(label: string, tone: string): string {
  return `data:image/svg+xml;utf8,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><rect width="128" height="128" fill="${tone}"/><circle cx="64" cy="50" r="28" fill="#f8d6c5"/><path d="M24 124c4-34 76-34 80 0" fill="#203040"/><text x="64" y="108" text-anchor="middle" font-family="Arial" font-size="16" fill="white">${label}</text></svg>`,
  )}`
}

const DEMO_CHARACTER: CharacterProfile = {
  id: 'mira',
  name: 'Mira',
  description: 'Portrait studio, regard calme, style éditorial',
  gender: 'femme',
  style: 'portrait réaliste, lumière douce',
  instructions: 'Répond avec chaleur, précision et un ton posé.',
  voice: '',
  image_model: 'sd15',
  seed: 42,
  created_at: now,
  updated_at: now,
  thumbnail: mockPortrait('Mira', '#0d78a8'),
  emotions: {
    neutral: { label: 'Neutre', image_path: '', image: mockPortrait('Neutre', '#0d78a8') },
    happy: { label: 'Heureuse', image_path: '', image: mockPortrait('Joie', '#198754') },
    thoughtful: { label: 'Réfléchie', image_path: '', image: '' },
    surprised: { label: 'Surprise', image_path: '', image: '' },
    worried: { label: 'Inquiète', image_path: '', image: '' },
    sad: { label: 'Triste', image_path: '', image: '' },
    amused: { label: 'Amusée', image_path: '', image: '' },
    focused: { label: 'Concentrée', image_path: '', image: '' },
  },
}

let characters: CharacterProfile[] = [DEMO_CHARACTER]
let activeCharacterId = DEMO_CHARACTER.id

function characterPayload(extra: Record<string, unknown> = {}) {
  const active = characters.find((character) => character.id === activeCharacterId) || null
  return {
    ok: true,
    characters,
    active_character_id: activeCharacterId,
    active_character: active,
    ...extra,
  }
}

const CONVERSATIONS: ConversationMeta[] = [
  { id: 'c1', title: 'Démo interface principale', created_at: now, updated_at: now, workdir: demoWorkdir, pinned: true, character_id: DEMO_CHARACTER.id },
  { id: 'c2', title: 'Démo analyse de fichier', created_at: now, updated_at: now, workdir: demoWorkdir },
  { id: 'c3', title: 'Démo recherche locale', created_at: now, updated_at: now, workdir: demoWorkdir },
  { id: 'c4', title: 'Démo génération image', created_at: now, updated_at: now, workdir: '/workspace/examples' },
  { id: 'c5', title: 'Démo paramètres', created_at: now, updated_at: now },
  { id: 'c6', title: 'Démo aide rapide', created_at: now, updated_at: now },
]

const MESSAGES: ChatMessage[] = [
  {
    role: 'user',
    content: 'Peux-tu me montrer un exemple de réponse structurée dans Lity ?',
  },
  {
    role: 'assistant',
    content: `Bien sûr. Ceci est une conversation fictive utilisée uniquement en développement pour tester l'interface.

## Exemple

1. Une réponse courte peut résumer le contexte.
2. Une liste peut organiser les prochaines actions.
3. Un extrait de code peut être affiché avec coloration.

\`\`\`python
def hello(name: str) -> str:
    return f"Bonjour {name}"
\`\`\`

Ces données ne viennent pas d'un historique utilisateur.`,
    steps: [
      { kind: 'tool_call', name: 'list_files', args: { path: 'examples' } },
      { kind: 'tool_result', ok: true, summary: '3 exemples' },
      { kind: 'receipts', grounded: true, tools_used: ['list_files'] },
    ],
  },
]

const SETTINGS: AppSettings = {
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
  saved_prompts: [
    { title: 'Revue de code', text: 'Relis ce code et signale les bugs, les risques et les simplifications possibles.' },
    { title: 'Explique simplement', text: 'Explique ce concept simplement, avec une analogie.' },
  ],
  web_search_enabled: false,
  searxng_url: 'http://localhost:8080',
  cross_session_memory: true,
  web_setup_resolved: true,
}

const STATE: AppState = {
  assistant_name: 'Assistant',
  model: 'qwen3:8b',
  workdir: demoWorkdir,
  active_conversation_id: 'c1',
  conversations: CONVERSATIONS,
  agent_mode: true,
  allow_commands: false,
  command_approval: false,
  yolo: false,
  web_search: false,
  rag_enabled: true,
  indexed_chunks: 1284,
  change_count: 0,
  image_active: false,
  video_active: false,
  chat_provider: 'ollama',
  active_character: DEMO_CHARACTER,
}

const HEALTH: HealthItem[] = [
  { name: 'Ollama', ok: true, detail: '5 modèles · qwen3:8b actif' },
  { name: 'Embeddings', ok: true, detail: 'nomic-embed-text' },
  { name: 'Recherche web', ok: false, detail: 'SearXNG hors ligne' },
  { name: 'Stable Diffusion', ok: false, detail: 'Serveur non détecté' },
  { name: 'Voix (Whisper + Piper)', ok: true, detail: 'fr_FR-siwis-medium' },
]

const WEB_STATUS: WebStatus = {
  url: 'http://localhost:8080',
  reachable: false,
  docker: true,
  container: '',
  fallback_ddg: true,
  setup_resolved: true,
  setup_running: false,
}

const RECOMMENDATIONS: ModelRecommendations = {
  hardware: {
    os: 'macOS', arch: 'arm64', cpu_cores: 12, ram_gb: 36, gpu: 'Apple M3 Pro',
    vram_gb: 27, accelerator: 'Metal', budget_gb: 27,
  },
  models: [
    { name: 'qwen3:8b', params_b: 8, size_gb: 5.2, kind: 'chat', installed: true, verdict: 'excellent', speed: 'rapide', recommended: true, grade: 'S', grade_label: 'Excellent', tokens_per_sec: 62, thinking: true, context_length: 32768, best_quant: { name: 'Q4_K_M', vram_gb: 5.2, grade: 'S' } },
    { name: 'llama3.1:8b', params_b: 8, size_gb: 4.9, kind: 'chat', installed: true, verdict: 'excellent', speed: 'rapide', grade: 'S', grade_label: 'Excellent', tokens_per_sec: 58, context_length: 131072, best_quant: { name: 'Q4_K_M', vram_gb: 4.9, grade: 'S' } },
    { name: 'qwen2.5-coder:14b', params_b: 14, size_gb: 9.0, kind: 'code', installed: false, verdict: 'bon', speed: 'correcte', grade: 'A', grade_label: 'Très bon', tokens_per_sec: 34, context_length: 32768, best_quant: { name: 'Q4_K_M', vram_gb: 9.0, grade: 'A' } },
    { name: 'gemma2:27b', params_b: 27, size_gb: 16.0, kind: 'chat', installed: false, verdict: 'limite', speed: 'lente', grade: 'C', grade_label: 'Limite', tokens_per_sec: 14, context_length: 8192, best_quant: { name: 'Q4_K_M', vram_gb: 16.0, grade: 'C' } },
  ],
  image_models: [
    { name: 'sdxl-base', display_name: 'Stable Diffusion XL Base 1.0', provider: 'Stability AI', params_b: 3.5, vram_gb: 8, disk_gb: 6.9, kind: 'image', backend: 'automatic1111', installed: false, verdict: 'excellent', speed: 'pret localement', recommended: true, grade: 'S', grade_label: 'Tourne parfaitement', score: 89, status: 'can-run', tokens_per_sec: null, mem_pct: 22, license: 'CreativeML Open RAIL++', model_url: 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0', install_hint: 'Checkpoint SDXL a placer dans models/Stable-diffusion.' },
    { name: 'sd15', display_name: 'Stable Diffusion 1.5', provider: 'Runway / Stability AI', params_b: 0.9, vram_gb: 4, disk_gb: 4.3, kind: 'image', backend: 'automatic1111', installed: true, verdict: 'excellent', speed: 'pret localement', grade: 'S', grade_label: 'Tourne parfaitement', score: 92, status: 'can-run', tokens_per_sec: null, mem_pct: 11, license: 'CreativeML Open RAIL-M', model_url: 'https://huggingface.co/runwayml/stable-diffusion-v1-5', install_hint: 'Checkpoint .safetensors/.ckpt a placer dans models/Stable-diffusion.' },
    { name: 'flux2-klein-4b', display_name: 'FLUX.2 klein 4B', provider: 'Black Forest Labs', params_b: 4, vram_gb: 13, disk_gb: 8, kind: 'image', backend: 'comfyui', installed: false, verdict: 'bon', speed: 'serre mais possible', grade: 'A', grade_label: 'Tourne tres bien', score: 74, status: 'tight', tokens_per_sec: null, mem_pct: 36, license: 'Apache 2.0', model_url: 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B', install_hint: 'Recommande via ComfyUI ou Diffusers; pas un checkpoint A1111 classique.' },
    { name: 'z-image-turbo', display_name: 'Z-Image-Turbo', provider: 'Tongyi-MAI', params_b: 6, vram_gb: 16, disk_gb: 12, kind: 'image', backend: 'stable-diffusion.cpp', installed: false, verdict: 'limite', speed: 'serre mais possible', grade: 'B', grade_label: 'Correct', score: 63, status: 'tight', tokens_per_sec: null, mem_pct: 44, license: 'Apache 2.0', model_url: 'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo', install_hint: 'Recommande via stable-diffusion.cpp GGUF ou ComfyUI.' },
  ],
  video_models: [
    { name: 'wan21-t2v-1.3b', display_name: 'Wan 2.1 T2V 1.3B', provider: 'Alibaba / Wan-AI', params_b: 1.3, vram_gb: 9, disk_gb: 17, kind: 'video', backend: 'diffusers', input_type: 'text', installed: false, verdict: 'bon', speed: 'serre mais possible', recommended: true, grade: 'B', grade_label: 'Correct', score: 64, status: 'tight', tokens_per_sec: null, mem_pct: 56, license: 'Apache 2.0', model_url: 'https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers', install_hint: 'Texte→vidéo le plus léger, viable sur 16 Go. Clips courts (~3 s, 480p).' },
    { name: 'wan22-ti2v-5b', display_name: 'Wan 2.2 TI2V-5B', provider: 'Alibaba / Wan-AI', params_b: 5, vram_gb: 16, disk_gb: 28, kind: 'video', backend: 'diffusers', input_type: 'text', installed: false, verdict: 'limite', speed: 'lent avec offload RAM', grade: 'C', grade_label: 'Juste', score: 41, status: 'can-run-slow', tokens_per_sec: null, mem_pct: 100, license: 'Apache 2.0', model_url: 'https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers', install_hint: 'Meilleure qualité (texte + image→vidéo), serré à 16 Go (offload mémoire).' },
    { name: 'ltx2-int4-mlx', display_name: 'LTX-2 (int4, MLX)', provider: 'Lightricks / dgrauet', params_b: 2, vram_gb: 12, disk_gb: 12, kind: 'video', backend: 'mlx', input_type: 'text', installed: false, verdict: 'limite', speed: 'serre mais possible', grade: 'C', grade_label: 'Juste', score: 48, status: 'tight', tokens_per_sec: null, mem_pct: 75, license: 'LTX Open', model_url: 'https://huggingface.co/dgrauet/ltx-2.3-mlx-q4', install_hint: 'Port MLX Metal natif, 16 Go minimum. Lity installe le runtime ltx-2-mlx au premier lancement.' },
  ],
}

/** Promise-returning stubs for everything the UI calls. */
function makeApi(): Record<string, (...args: unknown[]) => Promise<unknown>> {
  const r = <T>(value: T) => () => Promise.resolve(value)
  const api: Record<string, (...args: unknown[]) => Promise<unknown>> = {
    get_state: () => Promise.resolve({ ...STATE, active_character: characterPayload().active_character }),
    list_models: r({ models: ['qwen3:8b', 'llama3.1:8b', 'mistral:7b', 'phi4:14b', 'nomic-embed-text'].filter((m) => m !== 'nomic-embed-text'), selected: 'qwen3:8b', error: null }),
    set_model: (m) => Promise.resolve({ selected: String(m) }),
    get_settings: r(SETTINGS),
    update_settings: (patch) => Promise.resolve({ ...SETTINGS, ...(patch as object) }),
    list_skills: r({
      enabled: true,
      semantic: false,
      dir: '/Users/demo/Documents/Lity/skills',
      skills: [
        { name: 'revue-de-code', description: 'Relit du code pour trouver les bugs, les risques de sécurité et les améliorations de lisibilité.', when_to_use: '', triggers: [], allowed_tools: [], source: 'builtin', builtin: true, path: '', enabled: true },
        { name: 'explication-pas-a-pas', description: 'Explique un concept ou un bout de code de façon pédagogique et progressive.', when_to_use: '', triggers: [], allowed_tools: [], source: 'builtin', builtin: true, path: '', enabled: true },
        { name: 'synthese-de-texte', description: 'Résume un texte ou un document long en gardant les points clés.', when_to_use: '', triggers: [], allowed_tools: [], source: 'builtin', builtin: true, path: '', enabled: false },
        { name: 'traduction-fr-en', description: "Traduit un texte entre le français et l'anglais.", when_to_use: '', triggers: [], allowed_tools: [], source: 'user', builtin: false, path: '', enabled: true },
      ],
    }),
    toggle_skill: (name, enabled) => Promise.resolve({ ok: true, name: String(name), enabled: Boolean(enabled) }),
    create_skill: (name, _description, _body, _whenToUse, _triggers) =>
      Promise.resolve({ ok: true, message: `Compétence « ${String(name)} » créée.` }),
    delete_skill: (name) => Promise.resolve({ ok: true, message: `Compétence « ${String(name)} » supprimée.` }),
    codex_status: r({ available: true, authenticated: false, message: 'Non connecté' }),
    codex_models: r({ ok: true, models: [], default_model: '', message: '' }),
    claude_status: r({ available: true, authenticated: false, message: 'Non connecté' }),
    claude_models: r({ ok: true, models: [], default_model: '', message: '' }),
    grok_status: r({ available: true, authenticated: false, message: 'Non connecté' }),
    grok_models: r({ ok: true, models: [], default_model: '', message: '' }),
    usage: r({
      claude: {
        turns: 3,
        cost_usd: 0.42,
        input_tokens: 18400,
        output_tokens: 5200,
        total_tokens: 23600,
        by_model: [
          { model: 'claude-opus-4-8', turns: 2, input_tokens: 15000, output_tokens: 4200, total_tokens: 19200, cost_usd: 0.38 },
          { model: 'claude-sonnet-4-6', turns: 1, input_tokens: 3400, output_tokens: 1000, total_tokens: 4400, cost_usd: 0.04 },
        ],
      },
      codex: {
        turns: 1,
        cost_usd: 0,
        input_tokens: 5200,
        output_tokens: 900,
        total_tokens: 6100,
        by_model: [
          { model: 'gpt-5.5', turns: 1, input_tokens: 5200, output_tokens: 900, total_tokens: 6100, cost_usd: 0 },
        ],
      },
      grok: {
        turns: 1,
        cost_usd: 0,
        input_tokens: 3100,
        output_tokens: 600,
        total_tokens: 3700,
        by_model: [
          { model: 'grok-build', turns: 1, input_tokens: 3100, output_tokens: 600, total_tokens: 3700, cost_usd: 0 },
        ],
      },
    }),
    get_messages: r(MESSAGES),
    list_conversations: r(CONVERSATIONS),
    new_conversation: () => Promise.resolve({ id: 'cnew', title: 'Nouvelle conversation', created_at: now, updated_at: now, workdir: STATE.workdir, active_character: characterPayload().active_character }),
    switch_conversation: () => Promise.resolve({ success: true, active_conversation_id: 'c1', messages: MESSAGES, workdir: STATE.workdir, files: [], model: 'qwen3:8b', active_character: characterPayload().active_character }),
    list_workspace_files: r({ workdir: STATE.workdir, files: ['src/lity/services/ai/agent.py', 'src/lity/services/ai/prompts.py', 'src/lity/app/controller.py', 'README.md', 'pyproject.toml'] }),
    get_loaded_files: r([]),
    get_health: r(HEALTH),
    web_status: r(WEB_STATUS),
    lmstudio_models: r({
      ok: true,
      models: [
        {
          slug: 'qwen2.5-coder-14b-instruct-mlx-4bit',
          display_name: 'Qwen2.5 Coder 14B MLX 4-bit',
        },
        { slug: 'qwen3-8b-4bit-dwq', display_name: 'Qwen3 8B MLX DWQ' },
      ],
      default_model: 'qwen2.5-coder-14b-instruct-mlx-4bit',
      base_url: 'http://127.0.0.1:1234/v1',
      recommended: [],
      message: 'LM Studio connecté.',
    }),
    audio_status: r({ stt_available: true, stt_ready: true, stt_error: null, tts_available: true, has_voice: true }),
    image_active: r(false),
    video_active: r(false),
    video_pull_status: r({ active: null }),
    download_video_model: (name) => Promise.resolve({ ok: true, running: true, message: `Téléchargement de ${String(name)}…` }),
    select_video_model: (name) => Promise.resolve({ ok: true, selected: String(name) }),
    pull_status: r({ active: null, queue: [], progress: {} }),
    generation_stats: r({ tokens_per_sec: 62, context_used: 8420, context_length: 32768, usage_pct: 26 }),
    get_conversation_instructions: r({ instructions: '', temperature: null }),
    set_conversation_instructions: r({ ok: true, instructions: '', temperature: null }),
    list_characters: () => Promise.resolve(characterPayload()),
    create_character: (data) => {
      const raw = data as Partial<CharacterProfile>
      const id = String(raw.name || 'personnage').toLowerCase().replace(/[^a-z0-9]+/g, '-')
      const profile: CharacterProfile = {
        ...DEMO_CHARACTER,
        id,
        name: raw.name || 'Personnage',
        description: raw.description || '',
        gender: raw.gender || '',
        style: raw.style || '',
        instructions: raw.instructions || '',
        voice: raw.voice || '',
        image_model: raw.image_model || '',
        seed: raw.seed ?? -1,
        thumbnail: '',
        emotions: Object.fromEntries(
          Object.entries(DEMO_CHARACTER.emotions).map(([key, emotion]) => [
            key,
            { label: emotion.label, image_path: '', image: '' },
          ]),
        ),
      }
      characters = [profile, ...characters]
      return Promise.resolve(characterPayload({ character: profile, message: 'Personnage créé.' }))
    },
    update_character: (id, patch) => {
      characters = characters.map((character) =>
        character.id === id ? { ...character, ...(patch as Partial<CharacterProfile>) } : character,
      )
      const character = characters.find((item) => item.id === id) || null
      return Promise.resolve(characterPayload({ character, message: 'Personnage enregistré.' }))
    },
    delete_character: (id) => {
      characters = characters.filter((character) => character.id !== id)
      if (activeCharacterId === id) activeCharacterId = ''
      return Promise.resolve(characterPayload())
    },
    set_conversation_character: (id) => {
      activeCharacterId = String(id || '')
      return Promise.resolve(characterPayload())
    },
    generate_character_emotions: (id) => {
      characters = characters.map((character) => {
        if (character.id !== id) return character
        const emotions = Object.fromEntries(
          Object.entries(character.emotions).map(([key, emotion]) => [
            key,
            { ...emotion, image: mockPortrait(emotion.label || key, '#0d78a8') },
          ]),
        )
        return { ...character, emotions, thumbnail: emotions.neutral.image }
      })
      const character = characters.find((item) => item.id === id) || null
      return Promise.resolve(characterPayload({ character, generated: Object.keys(character?.emotions || {}), message: 'Émotions générées.' }))
    },
    get_memory: r({
      user_profile: { prénom: 'Alex', rôle: 'développeur', langue: 'français' },
      assistant_profile: { ton: 'concis et direct' },
      facts: { projet: 'Lity', éditeur: 'VS Code' },
    }),
    model_recommendations: r(RECOMMENDATIONS),
    list_models_detailed: r([
      { name: 'qwen3:8b', size: 5_200_000_000 },
      { name: 'llama3.1:8b', size: 4_900_000_000 },
      { name: 'mistral:7b', size: 4_100_000_000 },
      { name: 'nomic-embed-text', size: 280_000_000 },
    ]),
    model_suggestions: r([
      { name: 'nomic-embed-text', category: 'embedding', note: 'Léger, rapide, recommandé pour le RAG' },
      { name: 'mxbai-embed-large', category: 'embedding', note: 'Meilleure qualité, plus lourd' },
    ]),
    model_supports_tools: (m) => Promise.resolve({ model: String(m || 'qwen3:8b'), supports: true }),
    list_voices: r({ available: true, installed: ['fr_FR-siwis-medium'], current: 'fr_FR-siwis-medium', catalog: [
      { id: 'fr_FR-siwis-medium', label: 'Siwis (français)', lang: 'fr_FR' },
      { id: 'en_US-amy-medium', label: 'Amy (anglais)', lang: 'en_US' },
    ] }),
    git_status: r({ is_repo: true, branch: 'main', files: [
      { status: ' M', path: 'src/lity/services/ai/agent.py' },
      { status: '??', path: 'src/lity/services/ai/context.py' },
    ] }),
    git_branches: r({ branches: ['main', 'redesign'], current: 'main' }),
    git_diff: r({ diff: '' }),
  }
  // Fallback: any method we didn't define resolves to an empty object so calls
  // never throw "méthode de pont indisponible".
  return new Proxy(api, {
    get(target, prop: string) {
      if (prop in target) return target[prop]
      return () => Promise.resolve({})
    },
  })
}

export function installDevBridge(): void {
  if (typeof window === 'undefined') return
  if (window.pywebview) return // a real backend is present — never override it
  window.pywebview = { api: makeApi() }
  // eslint-disable-next-line no-console
  console.info('[devMock] pywebview bridge stubbed for browser preview (dev only)')
}
