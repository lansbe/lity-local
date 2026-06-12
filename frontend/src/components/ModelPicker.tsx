import { Menu, MenuItem, MenuLabel, MenuSeparator } from '../ui'
import { BoxIcon, ChevronDownIcon } from './Icons'

/** A CLI-backed chat provider (Codex, Claude) shown alongside the local Ollama
 *  models. Each owns a model list, a `slug` prefix used in the selected value,
 *  and an optional reasoning-effort menu. */
export interface CliProviderMenu {
  id: string
  /** Group heading, e.g. 'OpenAI · Codex' or 'Anthropic · Claude'. */
  label: string
  /** Prefix used in `selectedModelValue`, e.g. 'codex:' or 'claude:'. */
  prefix: string
  connected: boolean
  models: { slug: string; display_name?: string }[]
  reasoning: {
    effort: string
    levels: { effort: string; description?: string }[]
    defaultEffort?: string
    onSelect: (effort: string) => void
  }
}

interface ModelPickerProps {
  localModels: string[]
  cliProviders: CliProviderMenu[]
  selectedModelValue: string
  onSelectModel: (model: string) => void
  onOpenModels: () => void
  /** Open the menus upward (e.g. when anchored at the bottom composer). */
  up?: boolean
  align?: 'left' | 'right'
}

const REASONING_LABEL: Record<string, string> = {
  minimal: 'Minimal',
  low: 'Faible',
  medium: 'Moyen',
  high: 'Élevé',
  xhigh: 'Très élevé',
  max: 'Maximal',
}

const reasoningLabel = (effort: string) => REASONING_LABEL[effort] || effort

/** Per-conversation model (and CLI-provider reasoning) selector — a real menu,
 *  not a native <select>. Used in the composer so each chat can pick its own
 *  model across Ollama and any connected CLI provider (Codex, Claude). */
export function ModelPicker({
  localModels,
  cliProviders,
  selectedModelValue,
  onSelectModel,
  onOpenModels,
  up = false,
  align = 'left',
}: ModelPickerProps) {
  const connectedProviders = cliProviders.filter(
    (provider) => provider.connected && provider.models.length > 0,
  )
  const hasModels = localModels.length > 0 || connectedProviders.length > 0
  const activeProvider = cliProviders.find((provider) =>
    selectedModelValue.startsWith(provider.prefix),
  )

  const currentModelLabel = (() => {
    if (!hasModels) return 'Aucun modèle'
    if (activeProvider) {
      const slug = selectedModelValue.slice(activeProvider.prefix.length)
      return activeProvider.models.find((model) => model.slug === slug)?.display_name || slug
    }
    return selectedModelValue.replace(/^ollama:/, '') || 'Choisir un modèle'
  })()

  return (
    <div className="flex min-w-0 items-center gap-1">
      <Menu
        up={up}
        align={align}
        trigger={({ open, onClick }) => (
          <button
            type="button"
            onClick={onClick}
            disabled={!hasModels}
            aria-expanded={open}
            title="Modèle de cette conversation"
            className="flex h-8 max-w-[180px] items-center gap-1.5 rounded-md px-2 text-footnote text-secondary transition-colors hover:bg-surface-2 hover:text-primary disabled:opacity-50"
          >
            <span className="truncate font-medium">{currentModelLabel}</span>
            <ChevronDownIcon className="h-3.5 w-3.5 flex-none text-tertiary" />
          </button>
        )}
      >
        {localModels.length > 0 && <MenuLabel>Ollama local</MenuLabel>}
        {localModels.map((model) => (
          <MenuItem
            key={`ollama:${model}`}
            active={selectedModelValue === `ollama:${model}`}
            onClick={() => onSelectModel(`ollama:${model}`)}
          >
            {model}
          </MenuItem>
        ))}
        {connectedProviders.map((provider, index) => (
          <div key={provider.id}>
            {(localModels.length > 0 || index > 0) && <MenuSeparator />}
            <MenuLabel>{provider.label}</MenuLabel>
            {provider.models.map((model) => (
              <MenuItem
                key={`${provider.prefix}${model.slug}`}
                active={selectedModelValue === `${provider.prefix}${model.slug}`}
                onClick={() => onSelectModel(`${provider.prefix}${model.slug}`)}
              >
                {model.display_name || model.slug}
              </MenuItem>
            ))}
          </div>
        ))}
        <MenuSeparator />
        <MenuItem icon={<BoxIcon className="h-4 w-4" />} onClick={onOpenModels}>
          Gérer les modèles…
        </MenuItem>
      </Menu>

      {activeProvider && activeProvider.reasoning.levels.length > 0 && (
        <Menu
          up={up}
          align={align}
          trigger={({ open, onClick }) => (
            <button
              type="button"
              onClick={onClick}
              aria-expanded={open}
              title="Effort de raisonnement"
              className="flex h-8 items-center gap-1 rounded-md px-2 text-footnote text-secondary transition-colors hover:bg-surface-2 hover:text-primary"
            >
              <span className="font-medium">{reasoningLabel(activeProvider.reasoning.effort)}</span>
              <ChevronDownIcon className="h-3.5 w-3.5 flex-none text-tertiary" />
            </button>
          )}
        >
          <MenuLabel>Raisonnement</MenuLabel>
          {activeProvider.reasoning.levels.map((level) => (
            <MenuItem
              key={level.effort}
              active={activeProvider.reasoning.effort === level.effort}
              onClick={() => activeProvider.reasoning.onSelect(level.effort)}
            >
              {reasoningLabel(level.effort)}
              {activeProvider.reasoning.defaultEffort === level.effort ? ' · défaut' : ''}
            </MenuItem>
          ))}
        </Menu>
      )}
    </div>
  )
}
